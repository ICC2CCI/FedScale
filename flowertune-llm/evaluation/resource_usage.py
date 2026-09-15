"""Cluster-internal resource usage aggregation.

Covers FedScale evaluation requirement category 2:

    - GPU memory peak and GPU utilization
    - CPU utilization and CPU memory peak
    - Network traffic and NCCL overhead
    - FSDP state export overhead (full-state / sharded-state)
    - Checkpoint save, restore, and state export overhead

The raw data is produced by ``ResourceMonitor`` in ``flowertune_llm/metrics.py``
and written to the ``resources`` key of ``metrics_detailed.json``.  This module
aggregates across one or more runs and provides structured access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceUsageSummary:
    """Aggregated resource-usage metrics for one run."""

    gpu_memory_peak_mb: float
    gpu_utilization_avg_pct: float | None
    cpu_utilization_avg_pct: float
    cpu_memory_peak_mb: float
    # Network (container interface counters, excludes loopback)
    network_rx_bytes: int | None
    network_tx_bytes: int | None
    network_total_bytes: int | None
    # NCCL
    total_nccl_bytes: int
    nccl_collective_calls: int
    avg_nccl_comm_ms: float
    # State export
    full_state_export_s: float | None
    sharded_state_export_s: float | None
    state_serialization_s: float | None
    state_bytes: int | None
    # Checkpoint
    checkpoint_save_s: float | None
    checkpoint_bytes: int | None
    checkpoint_restore_s: float | None
    # Source
    source_files: list[str]

    def to_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def aggregate_resource_usage(
    metrics_paths: str | Path | list[str | Path],
) -> ResourceUsageSummary:
    """Aggregate resource usage from one or more metrics_detailed.json files.

    Args:
        metrics_paths: Path or list of paths to ``metrics_detailed.json``.

    Returns:
        ``ResourceUsageSummary`` with the worst-case (peak) values across
        all provided files, since resource usage is max-bounded not averaged.
    """
    if isinstance(metrics_paths, (str, Path)):
        metrics_paths = [metrics_paths]

    gpu_mem_peak = 0.0
    gpu_util_values: list[float] = []
    cpu_util_values: list[float] = []
    cpu_mem_peak = 0.0
    net_rx: int | None = None
    net_tx: int | None = None
    net_total: int | None = None
    total_nccl_bytes = 0
    nccl_calls = 0
    avg_nccl_ms = 0.0
    full_export_s: float | None = None
    sharded_export_s: float | None = None
    state_serial_s: float | None = None
    state_bytes: int | None = None
    ckpt_save_s: float | None = None
    ckpt_bytes: int | None = None
    ckpt_restore_s: float | None = None
    source_files: list[str] = []

    for path in metrics_paths:
        path = Path(path)
        if not path.exists():
            continue
        source_files.append(str(path))
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        resources = data.get("resources", {})
        gpu_mem_peak = max(gpu_mem_peak, _safe_float(resources.get("gpu_memory_peak_mb")))
        gpu_util = resources.get("gpu_utilization_avg_pct")
        if gpu_util is not None and gpu_util >= 0:
            gpu_util_values.append(_safe_float(gpu_util))
        cpu_util_values.append(_safe_float(resources.get("cpu_utilization_avg_pct")))
        cpu_mem_peak = max(cpu_mem_peak, _safe_float(resources.get("cpu_memory_peak_mb")))

        rx = resources.get("network_rx_bytes")
        tx = resources.get("network_tx_bytes")
        tot = resources.get("network_total_bytes")
        if rx is not None:
            net_rx = (net_rx or 0) + _safe_int(rx)
        if tx is not None:
            net_tx = (net_tx or 0) + _safe_int(tx)
        if tot is not None:
            net_total = (net_total or 0) + _safe_int(tot)

        training = data.get("training", {})
        total_nccl_bytes += _safe_int(training.get("total_nccl_bytes"))
        nccl_calls += _safe_int(training.get("nccl_collective_calls"))
        avg_nccl_ms = max(avg_nccl_ms, _safe_float(training.get("avg_nccl_comm_ms")))

        federated = data.get("federated", {})
        for key, attr in [
            ("full_state_export_s", "full_export_s"),
            ("sharded_state_export_s", "sharded_export_s"),
            ("state_serialization_s", "state_serial_s"),
            ("checkpoint_save_s", "ckpt_save_s"),
            ("checkpoint_restore_s", "ckpt_restore_s"),
        ]:
            val = federated.get(key)
            if val is not None:
                val = _safe_float(val)
                current = locals().get(attr)
                if current is None or val > current:
                    locals()[attr]  # noqa — keep linter happy
        # Direct assignment for clarity
        if federated.get("full_state_export_s") is not None:
            full_export_s = _safe_float(federated["full_state_export_s"])
        if federated.get("sharded_state_export_s") is not None:
            sharded_export_s = _safe_float(federated["sharded_state_export_s"])
        if federated.get("state_serialization_s") is not None:
            state_serial_s = _safe_float(federated["state_serialization_s"])
        if federated.get("checkpoint_save_s") is not None:
            ckpt_save_s = _safe_float(federated["checkpoint_save_s"])
        if federated.get("checkpoint_restore_s") is not None:
            ckpt_restore_s = _safe_float(federated["checkpoint_restore_s"])
        sb = federated.get("state_bytes")
        if sb is not None:
            state_bytes = _safe_int(sb)
        cb = federated.get("checkpoint_bytes")
        if cb is not None:
            ckpt_bytes = _safe_int(cb)

    gpu_util_avg = (
        sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None
    )

    return ResourceUsageSummary(
        gpu_memory_peak_mb=round(gpu_mem_peak, 2),
        gpu_utilization_avg_pct=round(gpu_util_avg, 2) if gpu_util_avg is not None else None,
        cpu_utilization_avg_pct=round(sum(cpu_util_values) / len(cpu_util_values), 2) if cpu_util_values else 0.0,
        cpu_memory_peak_mb=round(cpu_mem_peak, 2),
        network_rx_bytes=net_rx,
        network_tx_bytes=net_tx,
        network_total_bytes=net_total,
        total_nccl_bytes=total_nccl_bytes,
        nccl_collective_calls=nccl_calls,
        avg_nccl_comm_ms=round(avg_nccl_ms, 2),
        full_state_export_s=full_export_s,
        sharded_state_export_s=sharded_export_s,
        state_serialization_s=state_serial_s,
        state_bytes=state_bytes,
        checkpoint_save_s=ckpt_save_s,
        checkpoint_bytes=ckpt_bytes,
        checkpoint_restore_s=ckpt_restore_s,
        source_files=source_files,
    )
