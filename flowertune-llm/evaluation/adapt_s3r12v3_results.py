"""Adapt S3R12v3 round_log.json to evaluation-compatible JSON files.

The S3R12v3 dual-cluster path produces ``round_log.json`` and ``metrics.jsonl``
in ``results/<run_id>/``.  The evaluation package expects:

- ``federated_metrics_round_N.json`` — per-round server-side metrics
- ``experiment_summary.json`` — overall summary with round timings and
  aggregated client metrics
- ``metrics_detailed.json`` — per-step training performance and resource usage

This script reads a results directory and writes the missing files so that
``evaluation/generate_report.py`` and ``evaluation/federated_timing.py`` can
consume S3R12v3 experiment data without modification.

Usage::

    python -m evaluation.adapt_s3r12v3_results \
        --results-dir results/round_logs/s3r12v3-fsdp-rerun-20260909-095550 \
        --client-metrics-dir /path/to/client/metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def adapt_results(results_dir: str | Path, client_metrics_dir: str | Path | None = None) -> None:
    """Generate evaluation-compatible JSON files from S3R12v3 results.

    Args:
        results_dir: Path to the S3R12v3 results directory (contains round_log.json).
        client_metrics_dir: Path to client-side metrics_detailed.json files.
    """
    results_dir = Path(results_dir)
    round_log_path = results_dir / "round_log.json"
    if not round_log_path.exists():
        raise FileNotFoundError(f"round_log.json not found in {results_dir}")

    with round_log_path.open(encoding="utf-8") as f:
        round_log = json.load(f)

    # Generate federated_metrics_round_N.json for each round
    for entry in round_log:
        round_num = int(entry.get("round", 0))
        timing_s = entry.get("timing_s", {})
        clients = timing_s.get("clients", {})

        # Aggregate client WAN timing
        wan_downloads = [
            _safe_float(c.get("wan_download_s"))
            for c in clients.values()
            if c.get("wan_download_s") is not None
        ]
        wan_uploads = [
            _safe_float(c.get("wan_upload_s"))
            for c in clients.values()
            if c.get("wan_upload_s") is not None
        ]
        model_delta_bytes_list = [
            _safe_int(c.get("model_delta_bytes"))
            for c in clients.values()
            if c.get("model_delta_bytes") is not None
        ]

        federated_metrics = {
            "server_round": round_num,
            "timestamp": timing_s.get("finished_at"),
            "server_fedavg_aggregation_s": _safe_float(timing_s.get("agg_apply_s")),
            "federated_cycle_s": _safe_float(timing_s.get("round_wall_s")),
            "server_post_aggregation_s": _safe_float(timing_s.get("agg_total_s")),
        }
        if wan_downloads:
            federated_metrics["wan_download_s"] = sum(wan_downloads) / len(wan_downloads)
        if wan_uploads:
            federated_metrics["wan_upload_s"] = sum(wan_uploads) / len(wan_uploads)
        if model_delta_bytes_list:
            federated_metrics["model_delta_bytes"] = sum(model_delta_bytes_list)

        checkpoint_s = _safe_float(timing_s.get("agg_upload_global_s"))
        if checkpoint_s is not None:
            federated_metrics["checkpoint_save_s"] = checkpoint_s

        out_path = results_dir / f"federated_metrics_round_{round_num}.json"
        out_path.write_text(json.dumps(federated_metrics, indent=2), encoding="utf-8")

    # Generate experiment_summary.json
    round_timings = []
    aggregated_client_train_metrics = {}
    for entry in round_log:
        round_num = int(entry.get("round", 0))
        timing_s = entry.get("timing_s", {})
        clients = timing_s.get("clients", {})

        round_entry = {
            "round": round_num,
            "federated_cycle_s": _safe_float(timing_s.get("round_wall_s")),
            "server_post_aggregation_s": _safe_float(timing_s.get("agg_total_s")),
        }
        if _safe_float(timing_s.get("agg_apply_s")) is not None:
            round_entry["server_fedavg_aggregation_s"] = _safe_float(timing_s.get("agg_apply_s"))
        round_timings.append(round_entry)

        # Aggregate client metrics
        for cid, ct in clients.items():
            if not aggregated_client_train_metrics:
                aggregated_client_train_metrics = {
                    "client_training_seconds": _safe_float(ct.get("train_local_s"), 0.0),
                    "client_evaluation_seconds": _safe_float(ct.get("eval_local_s"), 0.0),
                    "client_round_seconds": _safe_float(ct.get("round_total_s"), 0.0),
                    "wan_download_seconds": _safe_float(ct.get("wan_download_s"), 0.0),
                    "wan_upload_seconds": _safe_float(ct.get("wan_upload_s"), 0.0),
                    "model_delta_bytes": _safe_int(ct.get("model_delta_bytes"), 0),
                }

    # Copy metrics_detailed.json if available
    metrics_detailed_path = results_dir / "metrics_detailed.json"
    if client_metrics_dir:
        client_md = Path(client_metrics_dir) / "metrics_detailed.json"
        if client_md.exists():
            import shutil
            shutil.copy2(client_md, metrics_detailed_path)

    summary = {
        "status": "completed",
        "round_timings": round_timings,
        "aggregated_client_train_metrics": aggregated_client_train_metrics,
        "final_evaluation_metrics": {},
    }

    summary_path = results_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write federated_timings.json
    timings_path = results_dir / "federated_timings.json"
    timings_path.write_text(json.dumps(round_timings, indent=2), encoding="utf-8")

    print(f"Generated evaluation-compatible files in {results_dir}:")
    print(f"  - {len(round_log)} federated_metrics_round_N.json files")
    print(f"  - experiment_summary.json")
    print(f"  - federated_timings.json")
    if metrics_detailed_path.exists():
        print(f"  - metrics_detailed.json (copied from client)")


def main():
    parser = argparse.ArgumentParser(
        description="Adapt S3R12v3 results to evaluation-compatible format"
    )
    parser.add_argument("--results-dir", required=True, help="S3R12v3 results directory")
    parser.add_argument(
        "--client-metrics-dir",
        default=None,
        help="Directory containing client-side metrics_detailed.json",
    )
    args = parser.parse_args()
    adapt_results(args.results_dir, args.client_metrics_dir)


if __name__ == "__main__":
    main()
