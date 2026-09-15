"""CLI for unified evaluation report generation.

Generates a comprehensive report from experiment artifacts, covering all
four FedScale evaluation requirement categories:

1. Cluster-internal training performance
2. Cluster-internal resource usage
3. Cross-ICC federated update timing
4. Model fine-tuning accuracy

Usage examples:

    # Single-run report (one experiment directory)
    python -m evaluation.generate_report \
        --results-dir /app/results/experiment-001 \
        --output report.json

    # DDP vs FSDP comparison report
    python -m evaluation.generate_report \
        --ddp-dir /app/results/ddp-experiment \
        --fsdp-dir /app/results/fsdp-experiment \
        --output comparison_report.md \
        --format markdown

    # Both single and comparison in one invocation
    python -m evaluation.generate_report \
        --ddp-dir /app/results/ddp-experiment \
        --fsdp-dir /app/results/fsdp-experiment \
        --output-dir ./reports \
        --format both
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.training_performance import aggregate_training_performance
from evaluation.resource_usage import aggregate_resource_usage
from evaluation.federated_timing import aggregate_federated_timing
from evaluation.comparison_report import (
    generate_comparison_report,
    format_comparison_report,
)


def _generate_single_run_report(results_dir: Path) -> dict:
    """Generate a report for a single experiment run."""
    detailed = results_dir / "metrics_detailed.json"

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
    }

    if detailed.exists():
        report["training_performance"] = aggregate_training_performance(detailed).to_dict()
        report["resource_usage"] = aggregate_resource_usage(detailed).to_dict()
    else:
        report["training_performance"] = None
        report["resource_usage"] = None
        report["_warnings"] = [f"metrics_detailed.json not found at {detailed}"]

    report["federated_timing"] = aggregate_federated_timing(results_dir).to_dict()

    # Model accuracy from comparison_report's loader
    from evaluation.comparison_report import _load_accuracy_from_summary
    accuracy = _load_accuracy_from_summary(results_dir)
    report["model_accuracy"] = accuracy.__dict__

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unified evaluation report from experiment artifacts."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--results-dir", help="Single experiment results directory")
    mode.add_argument("--ddp-dir", help="DDP experiment results directory (comparison mode)")

    parser.add_argument("--fsdp-dir", help="FSDP experiment results directory (comparison mode)")
    parser.add_argument("--output", help="Output file path (single or comparison)")
    parser.add_argument("--output-dir", help="Output directory (when --format both)")
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="json",
        help="Output format (default: json)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.results_dir:
        # Single-run mode
        results_dir = Path(args.results_dir)
        report = _generate_single_run_report(results_dir)

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / "report.json"
        else:
            json_path = Path(args.output or "report.json")

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Report saved to {json_path}")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    else:
        # Comparison mode
        if not args.fsdp_dir:
            raise ValueError("--fsdp-dir is required when --ddp-dir is used")

        ddp_dir = Path(args.ddp_dir)
        fsdp_dir = Path(args.fsdp_dir)
        report = generate_comparison_report(ddp_dir, fsdp_dir)
        report_dict = report.to_dict()

        if args.format in ("json", "both"):
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / "comparison_report.json"
            else:
                json_path = Path(args.output or "comparison_report.json")
                if json_path.suffix == ".md":
                    json_path = json_path.with_suffix(".json")

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2)
            print(f"JSON report saved to {json_path}")

        if args.format in ("markdown", "both"):
            md_text = format_comparison_report(report)

            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                md_path = out_dir / "comparison_report.md"
            else:
                md_path = Path(args.output or "comparison_report.md")
                if md_path.suffix == ".json":
                    md_path = md_path.with_suffix(".md")

            with md_path.open("w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"Markdown report saved to {md_path}")

        if args.format == "json":
            print(json.dumps(report_dict, ensure_ascii=False, indent=2))
        elif args.format == "markdown":
            print(md_text)


if __name__ == "__main__":
    main()
