"""Cluster-internal training performance aggregation and gap-fill utilities.

Covers FedScale evaluation requirement category 1:

    - Forward / backward propagation time
    - DDP All-Reduce time
    - FSDP All-Gather + Reduce-Scatter + reshard/prefetch time
    - Optimizer update time
    - Single step time
    - Total training time, throughput (tokens/s)
    - FSDP full-state / sharded-state export time
    - Checkpoint save / restore time

The per-step raw data is produced by ``StepMetricsCallback`` in
``flowertune_llm/metrics.py`` and written to ``metrics_detailed.json``.
This module provides:

1. ``aggregate_training_performance`` — read one or more
   ``metrics_detailed.json`` files and return a structured summary.
2. ``measure_sharded_state_export`` — time FSDP sharded-state export
   (the existing ``distributed_trainer.py`` only times full-state export).
3. ``TrainingPerformanceSummary`` — dataclass for programmatic access.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingPerformanceSummary:
    """Aggregated training-performance metrics for one run."""

    # Per-step averages
    avg_step_ms: float
    avg_forward_ms: float
    avg_backward_ms: float
    avg_comm_ms: float
    avg_optimizer_ms: float
    # Per-collective breakdown
    avg_all_reduce_ms: float
    avg_all_gather_ms: float
    avg_reduce_scatter_ms: float
    total_all_reduce_bytes: int
    total_all_gather_bytes: int
    total_reduce_scatter_bytes: int
    # Whole-run
    total_train_time_s: float
    total_tokens: int
    throughput_tokens_per_s: float
    # State export
    full_state_export_s: float | None
    sharded_state_export_s: float | None
    state_dict_conversion_s: float | None
    state_serialization_s: float | None
    state_bytes: int | None
    # Checkpoint
    checkpoint_save_s: float | None
    checkpoint_bytes: int | None
    checkpoint_restore_s: float | None
    # Raw step count
    num_steps: int
    # Source
    source_files: list[str] = field(default_factory=list)

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


def aggregate_training_performance(
    metrics_paths: str | Path | list[str | Path],
) -> TrainingPerformanceSummary:
    """Aggregate training performance from one or more metrics_detailed.json files.

    Args:
        metrics_paths: Path or list of paths to ``metrics_detailed.json`` files.
                       When multiple paths are given (e.g. DDP and FSDP runs),
                       each is read independently and the union of step records
                       is averaged.

    Returns:
        ``TrainingPerformanceSummary`` with aggregated metrics.
    """
    if isinstance(metrics_paths, (str, Path)):
        metrics_paths = [metrics_paths]

    all_steps: list[dict] = []
    training_summaries: list[dict] = []
    federated_summaries: list[dict] = []
    source_files: list[str] = []

    for path in metrics_paths:
        path = Path(path)
        if not path.exists():
            continue
        source_files.append(str(path))
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        training = data.get("training", {})
        steps = training.get("steps", [])
        all_steps.extend(steps)
        training_summaries.append(training)
        federated_summaries.append(data.get("federated", {}))

    n = len(all_steps)
    if n == 0:
        return TrainingPerformanceSummary(
            avg_step_ms=0.0, avg_forward_ms=0.0, avg_backward_ms=0.0,
            avg_comm_ms=0.0, avg_optimizer_ms=0.0,
            avg_all_reduce_ms=0.0, avg_all_gather_ms=0.0,
            avg_reduce_scatter_ms=0.0,
            total_all_reduce_bytes=0, total_all_gather_bytes=0,
            total_reduce_scatter_bytes=0,
            total_train_time_s=0.0, total_tokens=0,
            throughput_tokens_per_s=0.0,
            full_state_export_s=None, sharded_state_export_s=None,
            state_dict_conversion_s=None, state_serialization_s=None,
            state_bytes=None,
            checkpoint_save_s=None, checkpoint_bytes=None,
            checkpoint_restore_s=None,
            num_steps=0, source_files=source_files,
        )

    avg_step = sum(_safe_float(s.get("total_ms")) for s in all_steps) / n
    avg_fwd = sum(_safe_float(s.get("forward_ms")) for s in all_steps) / n
    avg_bwd = sum(_safe_float(s.get("backward_ms")) for s in all_steps) / n
    avg_comm = sum(_safe_float(s.get("comm_ms")) for s in all_steps) / n
    avg_opt = sum(_safe_float(s.get("optimizer_ms")) for s in all_steps) / n

    # Per-collective breakdown
    avg_ar = sum(_safe_float(s.get("all_reduce_ms", 0)) for s in all_steps) / n
    avg_ag = sum(_safe_float(s.get("all_gather_ms", 0)) for s in all_steps) / n
    avg_rs = sum(_safe_float(s.get("reduce_scatter_ms", 0)) for s in all_steps) / n

    total_ar_bytes = sum(_safe_int(s.get("all_reduce_bytes", 0)) for s in all_steps)
    total_ag_bytes = sum(_safe_int(s.get("all_gather_bytes", 0)) for s in all_steps)
    total_rs_bytes = sum(_safe_int(s.get("reduce_scatter_bytes", 0)) for s in all_steps)

    # Whole-run from the last training summary (most complete)
    last_training = training_summaries[-1] if training_summaries else {}
    total_time = _safe_float(last_training.get("total_train_time_s"))
    total_tokens = _safe_int(last_training.get("total_tokens"))
    throughput = _safe_float(last_training.get("throughput_tokens_per_s"))

    # State export / checkpoint from federated section
    last_fed = federated_summaries[-1] if federated_summaries else {}
    full_export = last_fed.get("full_state_export_s")
    sharded_export = last_fed.get("sharded_state_export_s")
    state_conversion = last_fed.get("state_dict_conversion_s")
    state_serial = last_fed.get("state_serialization_s")
    state_bytes = last_fed.get("state_bytes")
    ckpt_save = last_fed.get("checkpoint_save_s")
    ckpt_bytes = last_fed.get("checkpoint_bytes")
    ckpt_restore = last_fed.get("checkpoint_restore_s")

    return TrainingPerformanceSummary(
        avg_step_ms=round(avg_step, 2),
        avg_forward_ms=round(avg_fwd, 2),
        avg_backward_ms=round(avg_bwd, 2),
        avg_comm_ms=round(avg_comm, 2),
        avg_optimizer_ms=round(avg_opt, 2),
        avg_all_reduce_ms=round(avg_ar, 2),
        avg_all_gather_ms=round(avg_ag, 2),
        avg_reduce_scatter_ms=round(avg_rs, 2),
        total_all_reduce_bytes=total_ar_bytes,
        total_all_gather_bytes=total_ag_bytes,
        total_reduce_scatter_bytes=total_rs_bytes,
        total_train_time_s=round(total_time, 2),
        total_tokens=total_tokens,
        throughput_tokens_per_s=round(throughput, 2),
        full_state_export_s=_safe_float(full_export) if full_export is not None else None,
        sharded_state_export_s=_safe_float(sharded_export) if sharded_export is not None else None,
        state_dict_conversion_s=_safe_float(state_conversion) if state_conversion is not None else None,
        state_serialization_s=_safe_float(state_serial) if state_serial is not None else None,
        state_bytes=_safe_int(state_bytes) if state_bytes is not None else None,
        checkpoint_save_s=_safe_float(ckpt_save) if ckpt_save is not None else None,
        checkpoint_bytes=_safe_int(ckpt_bytes) if ckpt_bytes is not None else None,
        checkpoint_restore_s=_safe_float(ckpt_restore) if ckpt_restore is not None else None,
        num_steps=n,
        source_files=source_files,
    )


def measure_sharded_state_export(
    model,
    output_path: str | Path,
) -> tuple[Path, float]:
    """Time FSDP sharded-state-dict export and save to *output_path*.

    Unlike full-state export (already timed in ``distributed_trainer.py``),
    sharded export does not gather all parameters to rank 0 and is useful
    for checkpointing large models where the full state dict would exceed
    single-node memory.

    Args:
        model: FSDP-wrapped model.
        output_path: File path for the saved sharded state dict.

    Returns:
        ``(path, elapsed_seconds)``
    """
    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    sharded_config = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
        sharded_config,
    ):
        sharded_state = model.state_dict()
        torch.save(sharded_state, output_path)
    elapsed = time.perf_counter() - started

    return output_path, round(elapsed, 4)


def measure_checkpoint_restore(
    model,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> float:
    """Time loading a checkpoint and restoring it into *model*.

    Args:
        model: Target model (FSDP-wrapped or plain).
        checkpoint_path: Path to a saved state dict file.
        device: Device for initial load.

    Returns:
        Elapsed seconds.
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    started = time.perf_counter()
    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True,
    )
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    elapsed = time.perf_counter() - started
    return round(elapsed, 4)
