"""Cross-ICC federated update timing aggregation.

Covers FedScale evaluation requirement category 3:

    - Local model delta export time
    - Cross-center model update transmission time (WAN download + upload)
    - Plaintext FedAvg aggregation time (server-side)
    - Per-federated-round total time

Raw timing data is produced by:

- ``client_app.py`` — ``t_model_delta_export_s``, ``t_full_update_compression_s``,
  ``t_total_round_s``, ``model_delta_bytes``, plus the new
  ``wan_download_s`` / ``wan_upload_s`` fields.
- ``server_app.py`` — ``server_fedavg_aggregation_s``, ``federated_cycle_s``,
  ``server_post_aggregation_s``, ``checkpoint_save_s`` in
  ``federated_metrics_round_N.json`` and ``federated_timings.json``.
- ``distributed_trainer.py`` — ``full_state_export_s``,
  ``state_dict_conversion_s``, ``state_serialization_s`` in
  ``metrics_detailed.json``.

This module reads those scattered artifacts and produces a unified
per-round timing breakdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FederatedRoundTiming:
    """Timing breakdown for a single federated round."""

    round: int
    # Client-side
    model_delta_export_s: float | None
    full_update_compression_s: float | None
    wan_download_s: float | None
    wan_upload_s: float | None
    client_training_s: float | None
    client_evaluation_s: float | None
    client_round_s: float | None
    client_non_training_s: float | None
    # Server-side
    server_fedavg_aggregation_s: float | None
    server_post_aggregation_s: float | None
    checkpoint_save_s: float | None
    federated_cycle_s: float | None
    # Derived
    wan_transfer_total_s: float | None
    model_delta_bytes: int | None
    object_store_uploaded_bytes: int | None


@dataclass(frozen=True)
class FederatedTimingSummary:
    """Aggregated federated timing across all rounds."""

    rounds: list[FederatedRoundTiming]
    total_federated_time_s: float
    avg_round_s: float
    avg_model_delta_export_s: float
    avg_wan_transfer_s: float
    avg_fedavg_aggregation_s: float
    avg_client_training_s: float
    total_model_delta_bytes: int
    num_rounds: int
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_rounds": self.num_rounds,
            "total_federated_time_s": self.total_federated_time_s,
            "avg_round_s": self.avg_round_s,
            "avg_model_delta_export_s": self.avg_model_delta_export_s,
            "avg_wan_transfer_s": self.avg_wan_transfer_s,
            "avg_fedavg_aggregation_s": self.avg_fedavg_aggregation_s,
            "avg_client_training_s": self.avg_client_training_s,
            "total_model_delta_bytes": self.total_model_delta_bytes,
            "rounds": [r.__dict__ for r in self.rounds],
            "source_files": self.source_files,
        }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def aggregate_federated_timing(
    results_dir: str | Path,
) -> FederatedTimingSummary:
    """Aggregate federated round timings from a results directory.

    Reads the following files from *results_dir*:

    - ``federated_metrics_round_N.json`` — per-round server-side metrics
    - ``federated_timings.json`` — cumulative round timing array
    - ``experiment_summary.json`` — overall summary with round timings
    - ``metrics_detailed.json`` — client-side detailed metrics (if present)

    Args:
        results_dir: Path to the experiment results directory.

    Returns:
        ``FederatedTimingSummary`` with per-round and aggregated timings.
    """
    results_dir = Path(results_dir)

    # Collect per-round metrics files
    round_files = sorted(
        results_dir.glob("federated_metrics_round_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    # Also read federated_timings.json and experiment_summary.json
    timings_path = results_dir / "federated_timings.json"
    summary_path = results_dir / "experiment_summary.json"

    # Build a round-numbered dict of server-side metrics
    server_metrics: dict[int, dict] = {}
    for rf in round_files:
        round_num = int(rf.stem.split("_")[-1])
        with rf.open(encoding="utf-8") as f:
            server_metrics[round_num] = json.load(f)

    # Augment with federated_timings.json
    if timings_path.exists():
        with timings_path.open(encoding="utf-8") as f:
            timings_list = json.load(f)
        for entry in timings_list:
            round_num = _safe_int(entry.get("round")) or 0
            if round_num and round_num not in server_metrics:
                server_metrics[round_num] = entry
            elif round_num:
                server_metrics[round_num].update(entry)

    # Augment with experiment_summary.json
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        for entry in summary.get("round_timings", []):
            round_num = _safe_int(entry.get("round")) or 0
            if round_num and round_num not in server_metrics:
                server_metrics[round_num] = entry
            elif round_num:
                server_metrics[round_num].update(entry)

    # Read client-side metrics from metrics_detailed.json (if present)
    detailed_path = results_dir / "metrics_detailed.json"
    client_metrics: dict = {}
    if detailed_path.exists():
        with detailed_path.open(encoding="utf-8") as f:
            detailed = json.load(f)
        client_metrics = detailed.get("federated", {})

    # Read aggregated client train metrics from experiment_summary.json
    agg_client_metrics: dict = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        agg_client_metrics = summary.get("aggregated_client_train_metrics", {})

    source_files = [str(p) for p in round_files]
    if timings_path.exists():
        source_files.append(str(timings_path))
    if summary_path.exists():
        source_files.append(str(summary_path))
    if detailed_path.exists():
        source_files.append(str(detailed_path))

    # Build per-round timing records
    rounds: list[FederatedRoundTiming] = []
    for round_num in sorted(server_metrics.keys()):
        sm = server_metrics[round_num]

        wan_download = _safe_float(
            sm.get("wan_download_s")
            or client_metrics.get("wan_download_s")
            or agg_client_metrics.get("wan_download_seconds")
        )
        wan_upload = _safe_float(
            sm.get("wan_upload_s")
            or client_metrics.get("wan_upload_s")
            or agg_client_metrics.get("wan_upload_seconds")
        )
        wan_total = None
        if wan_download is not None or wan_upload is not None:
            wan_total = round((wan_download or 0.0) + (wan_upload or 0.0), 4)

        rounds.append(FederatedRoundTiming(
            round=round_num,
            model_delta_export_s=_safe_float(
                sm.get("model_delta_export_s")
                or client_metrics.get("t_model_delta_export_s")
                or agg_client_metrics.get("model_delta_export_seconds")
            ),
            full_update_compression_s=_safe_float(
                sm.get("full_update_compression_s")
                or client_metrics.get("t_full_update_compression_s")
                or agg_client_metrics.get("full_update_compression_seconds")
            ),
            wan_download_s=wan_download,
            wan_upload_s=wan_upload,
            client_training_s=_safe_float(
                sm.get("client_training_s")
                or agg_client_metrics.get("client_training_seconds")
            ),
            client_evaluation_s=_safe_float(
                sm.get("client_evaluation_s")
                or agg_client_metrics.get("client_evaluation_seconds")
            ),
            client_round_s=_safe_float(
                sm.get("client_round_s")
                or agg_client_metrics.get("client_round_seconds")
            ),
            client_non_training_s=_safe_float(
                sm.get("client_non_training_s")
                or agg_client_metrics.get("client_non_training_seconds")
            ),
            server_fedavg_aggregation_s=_safe_float(
                sm.get("server_fedavg_aggregation_s")
            ),
            server_post_aggregation_s=_safe_float(
                sm.get("server_post_aggregation_s")
            ),
            checkpoint_save_s=_safe_float(sm.get("checkpoint_save_s")),
            federated_cycle_s=_safe_float(sm.get("federated_cycle_s")),
            wan_transfer_total_s=wan_total,
            model_delta_bytes=_safe_int(
                sm.get("model_delta_bytes")
                or agg_client_metrics.get("model_delta_bytes")
            ),
            object_store_uploaded_bytes=_safe_int(
                sm.get("object_store_uploaded_bytes")
                or agg_client_metrics.get("object_store_uploaded_bytes")
            ),
        ))

    # Aggregates
    n = len(rounds)
    total_time = sum(r.federated_cycle_s or 0.0 for r in rounds)
    avg_round = total_time / n if n > 0 else 0.0
    avg_delta_export = (
        sum(r.model_delta_export_s or 0.0 for r in rounds) / n if n > 0 else 0.0
    )
    avg_wan = (
        sum(r.wan_transfer_total_s or 0.0 for r in rounds) / n if n > 0 else 0.0
    )
    avg_fedavg = (
        sum(r.server_fedavg_aggregation_s or 0.0 for r in rounds) / n if n > 0 else 0.0
    )
    avg_client_train = (
        sum(r.client_training_s or 0.0 for r in rounds) / n if n > 0 else 0.0
    )
    total_delta_bytes = sum(r.model_delta_bytes or 0 for r in rounds)

    return FederatedTimingSummary(
        rounds=rounds,
        total_federated_time_s=round(total_time, 2),
        avg_round_s=round(avg_round, 4),
        avg_model_delta_export_s=round(avg_delta_export, 4),
        avg_wan_transfer_s=round(avg_wan, 4),
        avg_fedavg_aggregation_s=round(avg_fedavg, 4),
        avg_client_training_s=round(avg_client_train, 4),
        total_model_delta_bytes=total_delta_bytes,
        num_rounds=n,
        source_files=source_files,
    )
