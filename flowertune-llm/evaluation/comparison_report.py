"""DDP vs FSDP four-category comparison report.

Covers FedScale evaluation requirement category 4 (model accuracy) plus
the cross-method comparison requirement:

    "比较 DDP 与 FSDP 在相同 global batch size、训练步数和随机种子下的
     最终模型质量。"

This module reads aggregated results from the other evaluation modules
(training_performance, resource_usage, federated_timing, and model accuracy
metrics) and produces a structured side-by-side comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.training_performance import (
    TrainingPerformanceSummary,
    aggregate_training_performance,
)
from evaluation.resource_usage import (
    ResourceUsageSummary,
    aggregate_resource_usage,
)
from evaluation.federated_timing import (
    FederatedTimingSummary,
    aggregate_federated_timing,
)


@dataclass(frozen=True)
class ModelAccuracyMetrics:
    """Model quality metrics for one method."""

    val_loss: float | None
    perplexity: float | None
    rouge_l_f1: float | None
    bertscore_f1: float | None
    token_overlap_accuracy: float | None
    macro_f1: float | None
    exact_match: float | None
    source: str | None = None


@dataclass(frozen=True)
class ComparisonReport:
    """Full four-category DDP vs FSDP comparison report."""

    # Category 1: Training performance
    training: dict[str, TrainingPerformanceSummary]
    # Category 2: Resource usage
    resources: dict[str, ResourceUsageSummary]
    # Category 3: Federated timing
    federated: dict[str, FederatedTimingSummary]
    # Category 4: Model accuracy
    accuracy: dict[str, ModelAccuracyMetrics]
    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "category_1_training_performance": {
                k: v.to_dict() for k, v in self.training.items()
            },
            "category_2_resource_usage": {
                k: v.to_dict() for k, v in self.resources.items()
            },
            "category_3_federated_timing": {
                k: v.to_dict() for k, v in self.federated.items()
            },
            "category_4_model_accuracy": {
                k: v.__dict__ for k, v in self.accuracy.items()
            },
            "metadata": self.metadata,
        }


def _load_accuracy_from_summary(
    results_dir: str | Path,
) -> ModelAccuracyMetrics:
    """Load model accuracy metrics from experiment_summary.json or evaluation results."""
    results_dir = Path(results_dir)

    # Try experiment_summary.json first (server-side final evaluation)
    summary_path = results_dir / "experiment_summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        final_metrics = summary.get("final_evaluation_metrics", {})
        if final_metrics:
            assistant = final_metrics.get("assistant_only", {})
            rouge = final_metrics.get("rouge_l", {})
            bert = final_metrics.get("bertscore", {})
            gen_q = final_metrics.get("generation_quality", {})
            return ModelAccuracyMetrics(
                val_loss=assistant.get("loss"),
                perplexity=assistant.get("ppl"),
                rouge_l_f1=rouge.get("f1"),
                bertscore_f1=bert.get("f1"),
                token_overlap_accuracy=gen_q.get("accuracy"),
                macro_f1=gen_q.get("macro_f1"),
                exact_match=gen_q.get("exact_match"),
                source=str(summary_path),
            )

    # Try evaluation.json (client-side final evaluation)
    for eval_path in results_dir.glob("**/evaluation.json"):
        with eval_path.open(encoding="utf-8") as f:
            result = json.load(f)
        validation = result.get("validation", {})
        downstream = result.get("downstream", {})
        return ModelAccuracyMetrics(
            val_loss=validation.get("val_loss"),
            perplexity=validation.get("perplexity"),
            rouge_l_f1=downstream.get("rouge_l"),
            bertscore_f1=downstream.get("bertscore_f1"),
            token_overlap_accuracy=downstream.get("accuracy"),
            macro_f1=downstream.get("macro_f1"),
            exact_match=downstream.get("exact_match"),
            source=str(eval_path),
        )

    # Try summary.json (from run_evaluation.py CLI)
    cli_summary = results_dir / "summary.json"
    if cli_summary.exists():
        with cli_summary.open(encoding="utf-8") as f:
            summary = json.load(f)
        results = summary.get("results", {})
        fed = results.get("federated", {})
        fed_val = fed.get("validation", {}) if isinstance(fed, dict) else {}
        fed_gen = fed.get("generation", {}) if isinstance(fed, dict) else {}
        delta = results.get("delta", {}) if isinstance(results, dict) else {}
        return ModelAccuracyMetrics(
            val_loss=fed_val.get("val_loss"),
            perplexity=fed_val.get("perplexity"),
            rouge_l_f1=fed_gen.get("rouge_l", {}).get("f1") if isinstance(fed_gen, dict) else None,
            bertscore_f1=fed_gen.get("bertscore", {}).get("f1") if isinstance(fed_gen, dict) else None,
            token_overlap_accuracy=fed_gen.get("token_overlap_accuracy") if isinstance(fed_gen, dict) else None,
            macro_f1=fed_gen.get("macro_f1") if isinstance(fed_gen, dict) else None,
            exact_match=fed_gen.get("exact_match") if isinstance(fed_gen, dict) else None,
            source=str(cli_summary),
        )

    return ModelAccuracyMetrics(
        val_loss=None, perplexity=None, rouge_l_f1=None,
        bertscore_f1=None, token_overlap_accuracy=None,
        macro_f1=None, exact_match=None, source=None,
    )


def generate_comparison_report(
    ddp_results_dir: str | Path,
    fsdp_results_dir: str | Path,
) -> ComparisonReport:
    """Generate a four-category DDP vs FSDP comparison report.

    Each results directory should contain the artifacts produced by a
    complete training experiment:

    - ``metrics_detailed.json`` — training + resource metrics
    - ``federated_metrics_round_*.json`` — per-round federated timing
    - ``federated_timings.json`` — cumulative round timings
    - ``experiment_summary.json`` — final evaluation metrics

    Args:
        ddp_results_dir: Path to the DDP experiment results directory.
        fsdp_results_dir: Path to the FSDP experiment results directory.

    Returns:
        ``ComparisonReport`` with all four categories.
    """
    ddp_dir = Path(ddp_results_dir)
    fsdp_dir = Path(fsdp_results_dir)

    # Category 1: Training performance
    training: dict[str, TrainingPerformanceSummary] = {}
    for label, dir_path in [("ddp", ddp_dir), ("fsdp", fsdp_dir)]:
        detailed = dir_path / "metrics_detailed.json"
        if detailed.exists():
            training[label] = aggregate_training_performance(detailed)

    # Category 2: Resource usage
    resources: dict[str, ResourceUsageSummary] = {}
    for label, dir_path in [("ddp", ddp_dir), ("fsdp", fsdp_dir)]:
        detailed = dir_path / "metrics_detailed.json"
        if detailed.exists():
            resources[label] = aggregate_resource_usage(detailed)

    # Category 3: Federated timing
    federated: dict[str, FederatedTimingSummary] = {}
    for label, dir_path in [("ddp", ddp_dir), ("fsdp", fsdp_dir)]:
        if dir_path.exists():
            federated[label] = aggregate_federated_timing(dir_path)

    # Category 4: Model accuracy
    accuracy: dict[str, ModelAccuracyMetrics] = {}
    for label, dir_path in [("ddp", ddp_dir), ("fsdp", fsdp_dir)]:
        if dir_path.exists():
            accuracy[label] = _load_accuracy_from_summary(dir_path)

    return ComparisonReport(
        training=training,
        resources=resources,
        federated=federated,
        accuracy=accuracy,
        metadata={
            "ddp_results_dir": str(ddp_dir),
            "fsdp_results_dir": str(fsdp_dir),
        },
    )


def format_comparison_report(report: ComparisonReport) -> str:
    """Format a ComparisonReport as a readable Markdown string."""
    lines: list[str] = []
    lines.append("# DDP vs FSDP Comparison Report")
    lines.append("")

    # Category 1: Training performance
    lines.append("## 1. Training Performance")
    lines.append("")
    lines.append("| Metric | DDP | FSDP |")
    lines.append("|---|---|---|")
    ddp_t = report.training.get("ddp")
    fsdp_t = report.training.get("fsdp")
    for field_name, label in [
        ("avg_step_ms", "Avg step (ms)"),
        ("avg_forward_ms", "Avg forward (ms)"),
        ("avg_backward_ms", "Avg backward (ms)"),
        ("avg_comm_ms", "Avg comm (ms)"),
        ("avg_all_reduce_ms", "Avg All-Reduce (ms)"),
        ("avg_all_gather_ms", "Avg All-Gather (ms)"),
        ("avg_reduce_scatter_ms", "Avg Reduce-Scatter (ms)"),
        ("avg_optimizer_ms", "Avg optimizer (ms)"),
        ("total_train_time_s", "Total train time (s)"),
        ("throughput_tokens_per_s", "Throughput (tokens/s)"),
        ("full_state_export_s", "Full-state export (s)"),
        ("sharded_state_export_s", "Sharded-state export (s)"),
        ("checkpoint_save_s", "Checkpoint save (s)"),
        ("checkpoint_restore_s", "Checkpoint restore (s)"),
    ]:
        ddp_val = getattr(ddp_t, field_name, None) if ddp_t else None
        fsdp_val = getattr(fsdp_t, field_name, None) if fsdp_t else None
        lines.append(f"| {label} | {ddp_val} | {fsdp_val} |")
    lines.append("")

    # Category 2: Resource usage
    lines.append("## 2. Resource Usage")
    lines.append("")
    lines.append("| Metric | DDP | FSDP |")
    lines.append("|---|---|---|")
    ddp_r = report.resources.get("ddp")
    fsdp_r = report.resources.get("fsdp")
    for field_name, label in [
        ("gpu_memory_peak_mb", "GPU mem peak (MB)"),
        ("gpu_utilization_avg_pct", "GPU util avg (%)"),
        ("cpu_utilization_avg_pct", "CPU util avg (%)"),
        ("cpu_memory_peak_mb", "CPU mem peak (MB)"),
        ("network_total_bytes", "Network total (bytes)"),
        ("total_nccl_bytes", "NCCL total (bytes)"),
        ("nccl_collective_calls", "NCCL calls"),
        ("avg_nccl_comm_ms", "Avg NCCL comm (ms)"),
        ("state_bytes", "State bytes"),
    ]:
        ddp_val = getattr(ddp_r, field_name, None) if ddp_r else None
        fsdp_val = getattr(fsdp_r, field_name, None) if fsdp_r else None
        lines.append(f"| {label} | {ddp_val} | {fsdp_val} |")
    lines.append("")

    # Category 3: Federated timing
    lines.append("## 3. Federated Update Timing")
    lines.append("")
    lines.append("| Metric | DDP | FSDP |")
    lines.append("|---|---|---|")
    ddp_f = report.federated.get("ddp")
    fsdp_f = report.federated.get("fsdp")
    for field_name, label in [
        ("num_rounds", "Rounds"),
        ("total_federated_time_s", "Total federated (s)"),
        ("avg_round_s", "Avg round (s)"),
        ("avg_model_delta_export_s", "Avg delta export (s)"),
        ("avg_wan_transfer_s", "Avg WAN transfer (s)"),
        ("avg_fedavg_aggregation_s", "Avg FedAvg aggregation (s)"),
        ("avg_client_training_s", "Avg client training (s)"),
        ("total_model_delta_bytes", "Total delta bytes"),
    ]:
        ddp_val = getattr(ddp_f, field_name, None) if ddp_f else None
        fsdp_val = getattr(fsdp_f, field_name, None) if fsdp_f else None
        lines.append(f"| {label} | {ddp_val} | {fsdp_val} |")
    lines.append("")

    # Category 4: Model accuracy
    lines.append("## 4. Model Accuracy")
    lines.append("")
    lines.append("| Metric | DDP | FSDP | Delta (FSDP−DDP) |")
    lines.append("|---|---|---|---|")
    ddp_a = report.accuracy.get("ddp")
    fsdp_a = report.accuracy.get("fsdp")
    for field_name, label in [
        ("val_loss", "Val Loss ↓"),
        ("perplexity", "Perplexity ↓"),
        ("rouge_l_f1", "ROUGE-L F1 ↑"),
        ("bertscore_f1", "BERTScore F1 ↑"),
        ("token_overlap_accuracy", "Token Acc ↑"),
        ("macro_f1", "Macro F1 ↑"),
        ("exact_match", "Exact Match ↑"),
    ]:
        ddp_val = getattr(ddp_a, field_name, None) if ddp_a else None
        fsdp_val = getattr(fsdp_a, field_name, None) if fsdp_a else None
        delta = None
        if ddp_val is not None and fsdp_val is not None:
            delta = round(fsdp_val - ddp_val, 4)
        lines.append(f"| {label} | {ddp_val} | {fsdp_val} | {delta} |")
    lines.append("")

    return "\n".join(lines)
