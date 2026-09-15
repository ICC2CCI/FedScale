"""Unified evaluation suite for FedScale federated LLM fine-tuning.

This package consolidates all evaluation functionality across four
FedScale evaluation requirement categories:

1. **Training performance**: forward/backward/comm/optimizer timing, throughput,
   FSDP state export, checkpoint save/restore
   → ``training_performance``
2. **Resource usage**: GPU/CPU memory and utilization, network traffic, NCCL
   overhead, state export overhead
   → ``resource_usage``
3. **Federated timing**: model delta export, WAN transfer, FedAvg aggregation,
   per-round total time
   → ``federated_timing``
4. **Model accuracy**: teacher-forced loss/PPL, ROUGE-L, BERTScore, Accuracy,
   Macro-F1, Exact Match, MMLU, baseline comparison, DDP vs FSDP comparison
   → ``validation_metrics``, ``generation_metrics``, ``baseline_comparison``,
     ``mmlu_evaluator``, ``comparison_report``

Additional experiment extensions:
- **Compression report**: bytes/round and quantization error (E3)
  → ``compression_report``
- **DP report**: RDP accounting and DP-utility curves (E4)
  → ``dp_report``
- **Data splitting**: three-way train/validation/test with stable hash
  → ``data_split``
- **Unified report generation**: CLI for all four categories
  → ``generate_report``

Modules with heavy dependencies (torch, transformers, datasets) are imported
lazily so that lightweight utilities (e.g. ``stable_sample_key``,
``compute_rdp_epsilon``, aggregation functions) can be used without installing
the full training stack.
"""

# --- Lightweight: no torch/datasets/transformers required ---

from evaluation.data_split import (
    stable_sample_key,
    create_train_val_test_split,
    reconstruct_held_out_test_set,
)
from evaluation.dp_report import (
    compute_rdp_epsilon,
    dp_utility_curve,
    format_dp_report,
)
from evaluation.compression_report import (
    compute_compression_report,
    measure_int8_quantization,
    measure_mask_compression,
)
from evaluation.training_performance import (
    TrainingPerformanceSummary,
    aggregate_training_performance,
)
from evaluation.resource_usage import (
    ResourceUsageSummary,
    aggregate_resource_usage,
)
from evaluation.federated_timing import (
    FederatedRoundTiming,
    FederatedTimingSummary,
    aggregate_federated_timing,
)
from evaluation.comparison_report import (
    ComparisonReport,
    ModelAccuracyMetrics,
    generate_comparison_report,
    format_comparison_report,
)


# --- Heavy: requires torch + transformers ---

def compute_validation_metrics(*args, **kwargs):
    from evaluation.validation_metrics import compute_validation_metrics as _impl
    return _impl(*args, **kwargs)


def compute_generation_metrics(*args, **kwargs):
    from evaluation.generation_metrics import compute_generation_metrics as _impl
    return _impl(*args, **kwargs)


def evaluate_with_baseline(*args, **kwargs):
    from evaluation.baseline_comparison import evaluate_with_baseline as _impl
    return _impl(*args, **kwargs)


def compute_delta(*args, **kwargs):
    from evaluation.baseline_comparison import compute_delta as _impl
    return _impl(*args, **kwargs)


def evaluate_mmlu(*args, **kwargs):
    from evaluation.mmlu_evaluator import evaluate_mmlu as _impl
    return _impl(*args, **kwargs)


def measure_sharded_state_export(*args, **kwargs):
    from evaluation.training_performance import measure_sharded_state_export as _impl
    return _impl(*args, **kwargs)


def measure_checkpoint_restore(*args, **kwargs):
    from evaluation.training_performance import measure_checkpoint_restore as _impl
    return _impl(*args, **kwargs)


__all__ = [
    # Data splitting
    "stable_sample_key",
    "create_train_val_test_split",
    "reconstruct_held_out_test_set",
    # Model accuracy (lazy)
    "compute_validation_metrics",
    "compute_generation_metrics",
    "evaluate_with_baseline",
    "compute_delta",
    "evaluate_mmlu",
    # Compression report (E3)
    "compute_compression_report",
    "measure_int8_quantization",
    "measure_mask_compression",
    # DP report (E4)
    "compute_rdp_epsilon",
    "dp_utility_curve",
    "format_dp_report",
    # Training performance
    "TrainingPerformanceSummary",
    "aggregate_training_performance",
    "measure_sharded_state_export",
    "measure_checkpoint_restore",
    # Resource usage
    "ResourceUsageSummary",
    "aggregate_resource_usage",
    # Federated timing
    "FederatedRoundTiming",
    "FederatedTimingSummary",
    "aggregate_federated_timing",
    # Comparison report
    "ComparisonReport",
    "ModelAccuracyMetrics",
    "generate_comparison_report",
    "format_comparison_report",
]
