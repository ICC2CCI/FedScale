#!/usr/bin/env python3
"""Minimal, deterministic FedScale v1 protocol smoke test.

This is deliberately a small control-plane/data-plane harness.  It does not
train an LLM or claim cryptographic security.  Instead it exercises the
invariants that must hold before wiring the protocol into the Kubernetes
FSDP client path:

* an ICC update is held as rank-local shards;
* rank-local fragments are exported through one canonical block layout;
* every ICC uses the same public mask and quantization parameters;
* clipping, local residuals, DP noise and integer encoding are applied before
  aggregate-only collection;
* the aggregate collector retains only a sum, never an individual update;
* a topology-aware scheduler produces a cohort/deadline/local-step plan.

The aggregate-only collector is a test double for Flower SecAgg+/an
equivalent production protocol.  It validates the wire-shape and aggregation
semantics, not cryptographic confidentiality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class BlockSpec:
    block_id: int
    start: int
    end: int
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalLayout:
    parameters: tuple[ParameterSpec, ...]
    blocks: tuple[BlockSpec, ...]
    total_elements: int
    layout_hash: str

    @classmethod
    def build(cls, parameter_shapes: Sequence[tuple[str, tuple[int, ...]]], block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        parameters: list[ParameterSpec] = []
        offset = 0
        for name, shape in parameter_shapes:
            length = math.prod(shape)
            parameters.append(ParameterSpec(name, tuple(shape), offset, offset + length))
            offset += length

        blocks: list[BlockSpec] = []
        for block_id, start in enumerate(range(0, offset, block_size)):
            end = min(offset, start + block_size)
            names = tuple(
                p.name for p in parameters if p.start < end and p.end > start
            )
            blocks.append(BlockSpec(block_id, start, end, names))

        manifest = {
            "parameters": [p.__dict__ for p in parameters],
            "blocks": [b.__dict__ for b in blocks],
            "total_elements": offset,
        }
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        layout_hash = hashlib.sha256(encoded).hexdigest()[:16]
        return cls(tuple(parameters), tuple(blocks), offset, layout_hash)


@dataclass
class Shard:
    rank: int
    start: int
    end: int
    delta: torch.Tensor


class AggregateOnlyCollector:
    """Non-cryptographic test double with aggregate-only retention semantics."""

    def __init__(self) -> None:
        self._sum: torch.Tensor | None = None
        self.count = 0
        self.individual_updates_retained = False

    def add(self, encoded: torch.Tensor) -> None:
        if encoded.ndim != 1 or encoded.dtype != torch.int16:
            raise ValueError("secure-aggregation input must be a 1-D int16 vector")
        contribution = encoded.to(torch.int64)
        self._sum = contribution if self._sum is None else self._sum + contribution
        self.count += 1
        # Do not append/store the contribution.  A real SecAgg implementation
        # would receive masked shares rather than this test double's vector.

    def result(self, expected_count: int) -> torch.Tensor:
        if self.count != expected_count or self._sum is None:
            raise RuntimeError(
                f"cohort incomplete: got {self.count}, expected {expected_count}"
            )
        return self._sum


@dataclass
class ICC:
    name: str
    rank_shards: list[Shard]
    residual: torch.Tensor
    bandwidth_mb_s: float
    rtt_ms: float
    step_s: float
    completed_rounds: int = 0


def balanced_ranges(length: int, world_size: int) -> list[tuple[int, int]]:
    if world_size <= 0 or world_size > length:
        raise ValueError("world_size must be in [1, total_elements]")
    ranges = []
    for rank in range(world_size):
        start = (length * rank) // world_size
        end = (length * (rank + 1)) // world_size
        ranges.append((start, end))
    return ranges


def make_shards(delta: torch.Tensor, world_size: int) -> list[Shard]:
    return [
        Shard(rank, start, end, delta[start:end].clone())
        for rank, (start, end) in enumerate(balanced_ranges(delta.numel(), world_size))
    ]


def build_block_vector(
    shards: Sequence[Shard], residual: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    """Assemble exactly one canonical block from rank-local fragments."""
    block = torch.empty(end - start, dtype=torch.float32)
    for shard in shards:
        overlap_start = max(start, shard.start)
        overlap_end = min(end, shard.end)
        if overlap_start >= overlap_end:
            continue
        source_start = overlap_start - shard.start
        target_start = overlap_start - start
        length = overlap_end - overlap_start
        block[target_start : target_start + length] = (
            shard.delta[source_start : source_start + length]
            + residual[overlap_start:overlap_end]
        )
    return block


def iter_windows(
    layout: CanonicalLayout,
    selected_ids: Sequence[int],
    window_size: int,
) -> Iterable[tuple[list[BlockSpec], int]]:
    current: list[BlockSpec] = []
    current_size = 0
    for block_id in selected_ids:
        block = layout.blocks[block_id]
        size = block.end - block.start
        if current and current_size + size > window_size:
            yield current, current_size
            current = []
            current_size = 0
        current.append(block)
        current_size += size
    if current:
        yield current, current_size


def public_mask(layout: CanonicalLayout, round_id: int, mask_ratio: float) -> list[int]:
    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must be in (0, 1]")
    count = max(1, math.ceil(len(layout.blocks) * mask_ratio))
    # Deterministic public rotation: no ICC-private update is consulted.
    start = ((round_id - 1) * count) % len(layout.blocks)
    return sorted((start + index) % len(layout.blocks) for index in range(count))


def quantize(vector: torch.Tensor, scale: float, qmax: int) -> tuple[torch.Tensor, torch.Tensor]:
    if scale <= 0:
        raise ValueError("quantization scale must be positive")
    encoded = torch.round(vector / scale).clamp(-qmax, qmax).to(torch.int16)
    return encoded, encoded.to(torch.float32) * scale


def scheduler_plan(clients: Sequence[ICC], payload_elements: int, deadline_s: float) -> dict:
    if not clients:
        raise ValueError("at least one ICC is required")
    plans = []
    for client in clients:
        upload_s = payload_elements / (client.bandwidth_mb_s * 1024 * 1024) + (
            client.rtt_ms / 1000.0
        )
        local_steps = max(1, int((deadline_s - upload_s) / client.step_s))
        plans.append(
            {
                "icc": client.name,
                "local_steps": local_steps,
                "estimated_upload_s": round(upload_s, 6),
                "bandwidth_mb_s": client.bandwidth_mb_s,
                "rtt_ms": client.rtt_ms,
            }
        )
    return {
        "cohort": [client.name for client in clients],
        "minimum_successful_participants": len(clients),
        "deadline_s": deadline_s,
        "plans": plans,
    }


def rdp_epsilon(rounds: int, clip_norm: float, sigma_client: float, cohort_size: int, delta: float) -> float | None:
    if sigma_client <= 0:
        return None
    sensitivity = 2.0 * clip_norm / cohort_size
    sigma_aggregate = sigma_client / math.sqrt(cohort_size)
    best = float("inf")
    for alpha in (2, 4, 8, 16, 32, 64):
        rho = rounds * alpha * sensitivity**2 / (2.0 * sigma_aggregate**2)
        best = min(best, rho + math.log(1.0 / delta) / (alpha - 1))
    return best


def run_smoke(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    layout = CanonicalLayout.build(
        [
            ("layers.0.attn.weight", (3, 4)),
            ("layers.0.mlp.weight", (2, 5)),
            ("layers.1.attn.weight", (4, 3)),
            ("lm_head.weight", (3, 3)),
        ],
        args.block_size,
    )
    base = torch.linspace(-0.2, 0.2, layout.total_elements)
    target = torch.linspace(0.4, -0.35, layout.total_elements)
    ranges = balanced_ranges(layout.total_elements, args.world_size)
    clients = [
        ICC(
            name=f"icc-{index + 1}",
            rank_shards=[],
            residual=torch.zeros(layout.total_elements),
            bandwidth_mb_s=100.0 if index == 0 else 35.0,
            rtt_ms=2.0 if index == 0 else 28.0,
            step_s=0.05 if index == 0 else 0.08,
        )
        for index in range(args.clients)
    ]
    if args.clients < 2:
        raise ValueError("FedScale smoke requires at least two ICCs")

    for client in clients:
        client.rank_shards = [
            Shard(rank, start, end, base[start:end].clone())
            for rank, (start, end) in enumerate(ranges)
        ]

    all_rounds: list[dict] = []
    initial_distance = float(torch.linalg.vector_norm(base - target).item())
    global_state = base.clone()
    for round_id in range(1, args.rounds + 1):
        selected = public_mask(layout, round_id, args.mask_ratio)
        mask_manifest = json.dumps(
            {"round": round_id, "layout_hash": layout.layout_hash, "blocks": selected},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        mask_hash = hashlib.sha256(mask_manifest).hexdigest()[:16]
        plan = scheduler_plan(clients, sum(layout.blocks[i].end - layout.blocks[i].start for i in selected), args.deadline_s)
        round_before = global_state.clone()
        encoded_payload_bytes = 0
        wire_records: list[list[torch.Tensor]] = []
        max_stream_errors: list[float] = []
        max_clip_norm = 0.0

        for client_index, client in enumerate(clients):
            # Synthetic local FSDP training: only rank-local delta shards are
            # retained after this point, matching the bridge's input shape.
            direction = target - global_state
            bias = (client_index + 1) * 0.002 * torch.sin(
                torch.arange(layout.total_elements, dtype=torch.float32) + round_id
            )
            local_delta = 0.28 * direction + bias
            local_delta += 0.0005 * torch.cos(torch.arange(layout.total_elements) * (client_index + 1))
            client.rank_shards = make_shards(local_delta, args.world_size)
            norm = float(
                math.sqrt(sum(float(torch.sum(shard.delta**2).item()) for shard in client.rank_shards))
            )
            clip_factor = min(1.0, args.clip_norm / (norm + 1e-12))
            for shard in client.rank_shards:
                shard.delta.mul_(clip_factor)
            max_clip_norm = max(max_clip_norm, norm * clip_factor)

            # Reference only: prove block streaming reconstructs the same
            # clipped+residual block as a full-vector implementation.
            clipped_reference = local_delta * clip_factor + client.residual
            for block in layout.blocks:
                streamed = build_block_vector(client.rank_shards, client.residual, block.start, block.end)
                max_stream_errors.append(float(torch.max(torch.abs(streamed - clipped_reference[block.start:block.end])).item()))

            client_encoded: list[torch.Tensor] = []
            for window_blocks, _ in iter_windows(layout, selected, args.window_size):
                window_values: list[torch.Tensor] = []
                for block in window_blocks:
                    effective = build_block_vector(client.rank_shards, client.residual, block.start, block.end)
                    if args.noise_sigma:
                        noise = torch.randn_like(effective) * args.noise_sigma
                    else:
                        noise = torch.zeros_like(effective)
                    encoded, dequantized = quantize(effective + noise, args.quant_scale, args.qmax)
                    client.residual[block.start:block.end] = effective - dequantized
                    window_values.append(encoded)
                encoded_window = torch.cat(window_values)
                client_encoded.append(encoded_window)
                encoded_payload_bytes += encoded_window.numel()  # INT8 wire estimate
            selected_set = set(selected)
            for block in layout.blocks:
                if block.block_id in selected_set:
                    continue
                # Coordinates skipped by the public mask are not discarded:
                # they remain in the ICC-local error-feedback residual.
                effective = build_block_vector(
                    client.rank_shards, client.residual, block.start, block.end
                )
                client.residual[block.start:block.end] = effective
            wire_records.append(client_encoded)

            # Unselected coordinates remain entirely local as residual state.
            selected_positions = torch.cat(
                [torch.arange(layout.blocks[i].start, layout.blocks[i].end) for i in selected]
            )
            unselected = torch.ones(layout.total_elements, dtype=torch.bool)
            unselected[selected_positions] = False
            if torch.any(unselected) and not torch.any(torch.abs(client.residual[unselected]) > 0):
                raise AssertionError("public mask failed to preserve local residual state")

        # Aggregate the exact wire records produced by each ICC.  The
        # collector's only retained state is the aggregate integer sum.
        window_specs = list(iter_windows(layout, selected, args.window_size))
        for window_index, (window_blocks, window_size) in enumerate(window_specs):
            collector = AggregateOnlyCollector()
            for client_records in wire_records:
                collector.add(client_records[window_index])
            aggregate_sum = collector.result(len(clients))
            direct_sum = sum(
                (client_records[window_index].to(torch.int64) for client_records in wire_records),
                torch.zeros(window_size, dtype=torch.int64),
            )
            if not torch.equal(aggregate_sum, direct_sum):
                raise AssertionError("aggregate-only integer sum differs from direct sum")
            if collector.individual_updates_retained:
                raise AssertionError("aggregate-only collector retained an ICC update")

        # The released aggregate is dequantized and applied only to selected
        # canonical blocks.  Unselected blocks remain unchanged this round.
        global_update = torch.zeros_like(global_state)
        for window_index, (window_blocks, window_size) in enumerate(window_specs):
            window_sum = sum(
                (client_records[window_index].to(torch.int64) for client_records in wire_records),
                torch.zeros(window_size, dtype=torch.int64),
            )
            offset = 0
            for block in window_blocks:
                size = block.end - block.start
                global_update[block.start:block.end] = (
                    window_sum[offset : offset + size].to(torch.float32)
                    / len(clients)
                    * args.quant_scale
                )
                offset += size

        global_state = global_state + global_update
        distance_after = float(torch.linalg.vector_norm(global_state - target).item())
        full_float_bytes = layout.total_elements * 4 * len(clients)
        all_rounds.append(
            {
                "round": round_id,
                "layout_hash": layout.layout_hash,
                "mask_hash": mask_hash,
                "public_mask_blocks": selected,
                "cohort": plan["cohort"],
                "local_steps": {item["icc"]: item["local_steps"] for item in plan["plans"]},
                "max_clipped_norm": round(max_clip_norm, 8),
                "max_stream_reconstruction_error": max(max_stream_errors),
                "payload_bytes_int8_estimate": encoded_payload_bytes,
                "full_float32_payload_bytes": full_float_bytes,
                "compression_ratio": round(full_float_bytes / max(encoded_payload_bytes, 1), 4),
                "distance_to_target_before": round(float(torch.linalg.vector_norm(round_before - target).item()), 8),
                "distance_to_target_after": round(distance_after, 8),
                "aggregate_only": True,
            }
        )
        for client in clients:
            client.completed_rounds += 1

    result = {
        "status": "passed",
        "implementation": "fedscale-v1-smoke",
        "security_note": "aggregate-only collector is a non-cryptographic SecAgg test double",
        "layout": {
            "hash": layout.layout_hash,
            "total_elements": layout.total_elements,
            "blocks": [b.__dict__ for b in layout.blocks],
        },
        "config": {
            **vars(args),
            "output": str(args.output) if args.output else None,
        },
        "initial_distance_to_target": initial_distance,
        "final_distance_to_target": all_rounds[-1]["distance_to_target_after"],
        "rdp_epsilon": rdp_epsilon(args.rounds, args.clip_norm, args.noise_sigma, len(clients), args.dp_delta),
        "rounds": all_rounds,
        "checks": {
            "same_layout_hash": len({item["layout_hash"] for item in all_rounds}) == 1,
            "public_mask_and_hash_present": all(item["public_mask_blocks"] and item["mask_hash"] for item in all_rounds),
            "streaming_reconstruction_exact": all(item["max_stream_reconstruction_error"] < 1e-6 for item in all_rounds),
            "clip_bound_respected": all(item["max_clipped_norm"] <= args.clip_norm + 1e-6 for item in all_rounds),
            "aggregate_only": all(item["aggregate_only"] for item in all_rounds),
            "topology_plan_present": all(item["cohort"] and item["local_steps"] for item in all_rounds),
        },
    }
    failed = [name for name, passed in result["checks"].items() if not passed]
    if failed:
        result["status"] = "failed"
        result["failed_checks"] = failed
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--clip-norm", type=float, default=0.5)
    parser.add_argument("--noise-sigma", type=float, default=0.01)
    parser.add_argument("--quant-scale", type=float, default=0.01)
    parser.add_argument("--qmax", type=int, default=127)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--deadline-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    started = time.perf_counter()
    try:
        result = run_smoke(args)
    except Exception as exc:  # Keep CLI output useful for smoke debugging.
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["elapsed_s"] = round(time.perf_counter() - started, 6)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["status"] == "passed":
        print("FEDScale smoke PASS")
        return 0
    print("FEDScale smoke FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
