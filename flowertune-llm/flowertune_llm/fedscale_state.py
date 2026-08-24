"""Canonical-block INT8 update encoding for the FedScale federation boundary.

The module intentionally operates on CPU state-dict tensors.  A training rank
can stream one canonical block at a time into this interface; neither the
encoder nor the decoder flattens a complete model into one tensor.  This is
the shared, deterministic wire contract used by ClientApp and ServerApp.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import torch


FEDSCALE_FORMAT_KEY = "__fedscale_block_int8__"
FEDSCALE_LAYOUT_HASH_KEY = "__fedscale_layout_hash__"
FEDSCALE_MASK_HASH_KEY = "__fedscale_mask_hash__"
FEDSCALE_BLOCK_IDS_KEY = "__fedscale_block_ids__"
FEDSCALE_BLOCK_OFFSETS_KEY = "__fedscale_block_offsets__"
FEDSCALE_BLOCK_SCALES_KEY = "__fedscale_block_scales__"
FEDSCALE_BLOCK_VALUES_KEY = "__fedscale_block_values__"


@dataclass(frozen=True)
class TensorRange:
    name: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass(frozen=True)
class CanonicalBlock:
    block_id: int
    start: int
    end: int


@dataclass(frozen=True)
class CanonicalLayout:
    tensors: tuple[TensorRange, ...]
    blocks: tuple[CanonicalBlock, ...]
    total_elements: int
    block_size: int
    layout_hash: str


def _floating_tensors(state: Mapping[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, value)
        for name, value in sorted(state.items())
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    ]


def build_canonical_layout(
    state: Mapping[str, torch.Tensor], block_size: int
) -> CanonicalLayout:
    """Build a stable layout from floating-point state-dict tensor names/shapes."""
    if block_size <= 0:
        raise ValueError("fedscale block_size must be positive")
    tensors: list[TensorRange] = []
    offset = 0
    for name, value in _floating_tensors(state):
        length = value.numel()
        tensors.append(TensorRange(name, tuple(value.shape), offset, offset + length))
        offset += length
    if not tensors:
        raise ValueError("FedScale requires at least one floating-point tensor")
    blocks = tuple(
        CanonicalBlock(block_id, start, min(offset, start + block_size))
        for block_id, start in enumerate(range(0, offset, block_size))
    )
    manifest = {
        "block_size": block_size,
        "tensors": [
            {"name": tensor.name, "shape": tensor.shape, "start": tensor.start, "end": tensor.end}
            for tensor in tensors
        ],
        "total_elements": offset,
    }
    layout_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CanonicalLayout(tuple(tensors), blocks, offset, block_size, layout_hash)


def public_block_mask(
    layout: CanonicalLayout, server_round: int, ratio: float
) -> tuple[int, ...]:
    """Return a deterministic public rotating mask without client data input."""
    if server_round < 1:
        raise ValueError("server_round must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("fedscale mask ratio must be in (0, 1]")
    count = max(1, math.ceil(len(layout.blocks) * ratio))
    start = ((server_round - 1) * count) % len(layout.blocks)
    return tuple(sorted((start + index) % len(layout.blocks) for index in range(count)))


def mask_hash(layout: CanonicalLayout, block_ids: Sequence[int]) -> str:
    _validate_block_ids(layout, block_ids)
    manifest = {"layout_hash": layout.layout_hash, "block_ids": list(block_ids)}
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_block_ids(layout: CanonicalLayout, block_ids: Sequence[int]) -> None:
    if not block_ids:
        raise ValueError("FedScale update must include at least one canonical block")
    ids = tuple(int(block_id) for block_id in block_ids)
    if tuple(sorted(set(ids))) != ids:
        raise ValueError("FedScale block ids must be unique and sorted")
    if ids[0] < 0 or ids[-1] >= len(layout.blocks):
        raise ValueError("FedScale block id is outside the canonical layout")


def _validate_state(layout: CanonicalLayout, state: Mapping[str, torch.Tensor]) -> None:
    for tensor in layout.tensors:
        value = state.get(tensor.name)
        if value is None:
            raise KeyError(f"state is missing canonical tensor {tensor.name!r}")
        if tuple(value.shape) != tensor.shape:
            raise ValueError(
                f"canonical tensor shape changed for {tensor.name!r}: "
                f"expected {tensor.shape}, got {tuple(value.shape)}"
            )


def _block_delta(
    final_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    layout: CanonicalLayout,
    block: CanonicalBlock,
) -> torch.Tensor:
    fragments: list[torch.Tensor] = []
    for tensor in layout.tensors:
        overlap_start = max(block.start, tensor.start)
        overlap_end = min(block.end, tensor.end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - tensor.start
        local_end = overlap_end - tensor.start
        final_value = final_state[tensor.name].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        initial_value = initial_state[tensor.name].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        fragments.append(final_value[local_start:local_end] - initial_value[local_start:local_end])
    delta = torch.cat(fragments)
    if delta.numel() != block.end - block.start:
        raise RuntimeError(f"canonical block {block.block_id} was assembled incompletely")
    return delta


def encode_fedscale_int8_delta(
    final_state: Mapping[str, torch.Tensor],
    initial_state: Mapping[str, torch.Tensor],
    layout: CanonicalLayout,
    block_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Encode selected canonical-block deltas with one symmetric INT8 scale/block."""
    _validate_block_ids(layout, block_ids)
    _validate_state(layout, final_state)
    _validate_state(layout, initial_state)
    values: list[torch.Tensor] = []
    offsets = [0]
    scales: list[torch.Tensor] = []
    for block_id in block_ids:
        delta = _block_delta(final_state, initial_state, layout, layout.blocks[block_id])
        maximum = delta.abs().max()
        scale = maximum / 127.0 if maximum.item() > 0.0 else torch.tensor(1.0)
        values.append(torch.clamp(torch.round(delta / scale), -127, 127).to(torch.int8))
        scales.append(scale.to(torch.float32).reshape(()))
        offsets.append(offsets[-1] + delta.numel())
    return {
        FEDSCALE_FORMAT_KEY: torch.tensor([1], dtype=torch.uint8),
        FEDSCALE_LAYOUT_HASH_KEY: torch.tensor(
            list(bytes.fromhex(layout.layout_hash)), dtype=torch.uint8
        ),
        FEDSCALE_MASK_HASH_KEY: torch.tensor(
            list(bytes.fromhex(mask_hash(layout, block_ids))), dtype=torch.uint8
        ),
        FEDSCALE_BLOCK_IDS_KEY: torch.tensor(block_ids, dtype=torch.int32),
        FEDSCALE_BLOCK_OFFSETS_KEY: torch.tensor(offsets, dtype=torch.int64),
        FEDSCALE_BLOCK_SCALES_KEY: torch.stack(scales),
        FEDSCALE_BLOCK_VALUES_KEY: torch.cat(values),
    }


def is_fedscale_int8_delta(state: Mapping[str, torch.Tensor]) -> bool:
    return FEDSCALE_FORMAT_KEY in state


def encoded_block_ids(
    encoded: Mapping[str, torch.Tensor], layout: CanonicalLayout
) -> tuple[int, ...]:
    _validate_encoded(encoded, layout)
    return tuple(int(value) for value in encoded[FEDSCALE_BLOCK_IDS_KEY].tolist())


def _validate_encoded(encoded: Mapping[str, torch.Tensor], layout: CanonicalLayout) -> None:
    required = {
        FEDSCALE_FORMAT_KEY,
        FEDSCALE_LAYOUT_HASH_KEY,
        FEDSCALE_MASK_HASH_KEY,
        FEDSCALE_BLOCK_IDS_KEY,
        FEDSCALE_BLOCK_OFFSETS_KEY,
        FEDSCALE_BLOCK_SCALES_KEY,
        FEDSCALE_BLOCK_VALUES_KEY,
    }
    missing = sorted(required.difference(encoded))
    if missing:
        raise ValueError(f"FedScale encoded update missing fields: {missing}")
    received_layout_hash = bytes(
        encoded[FEDSCALE_LAYOUT_HASH_KEY].to(dtype=torch.uint8).tolist()
    ).hex()
    if received_layout_hash != layout.layout_hash:
        raise ValueError("FedScale layout hash mismatch")
    ids = tuple(int(value) for value in encoded[FEDSCALE_BLOCK_IDS_KEY].tolist())
    _validate_block_ids(layout, ids)
    received_mask_hash = bytes(
        encoded[FEDSCALE_MASK_HASH_KEY].to(dtype=torch.uint8).tolist()
    ).hex()
    if received_mask_hash != mask_hash(layout, ids):
        raise ValueError("FedScale mask hash mismatch")
    offsets = encoded[FEDSCALE_BLOCK_OFFSETS_KEY].to(dtype=torch.int64)
    if offsets.numel() != len(ids) + 1 or int(offsets[0]) != 0:
        raise ValueError("FedScale block offsets are malformed")
    expected_sizes = [layout.blocks[block_id].end - layout.blocks[block_id].start for block_id in ids]
    actual_sizes = [int(offsets[index + 1] - offsets[index]) for index in range(len(ids))]
    if actual_sizes != expected_sizes:
        raise ValueError("FedScale block offsets do not match canonical layout")
    if int(offsets[-1]) != encoded[FEDSCALE_BLOCK_VALUES_KEY].numel():
        raise ValueError("FedScale values length does not match offsets")
    if encoded[FEDSCALE_BLOCK_SCALES_KEY].numel() != len(ids):
        raise ValueError("FedScale scale count does not match block ids")


def apply_fedscale_int8_delta(
    target_state: Mapping[str, torch.Tensor],
    encoded: Mapping[str, torch.Tensor],
    layout: CanonicalLayout,
    weight: float = 1.0,
    *,
    clone: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply an encoded delta block-by-block to *target_state*.

    The default copies tensor objects so it can reconstruct a client-side global
    state safely.  Server aggregation may pass ``clone=False`` to update its
    retained global state in place without allocating a second model state.
    """
    _validate_state(layout, target_state)
    _validate_encoded(encoded, layout)
    if not math.isfinite(weight):
        raise ValueError("FedScale aggregation weight must be finite")
    result = {
        name: value.clone() if clone else value
        for name, value in target_state.items()
    }
    ids = encoded_block_ids(encoded, layout)
    offsets = encoded[FEDSCALE_BLOCK_OFFSETS_KEY].to(dtype=torch.int64)
    scales = encoded[FEDSCALE_BLOCK_SCALES_KEY].to(dtype=torch.float32)
    values = encoded[FEDSCALE_BLOCK_VALUES_KEY].to(dtype=torch.float32)
    for index, block_id in enumerate(ids):
        block = layout.blocks[block_id]
        delta = values[int(offsets[index]):int(offsets[index + 1])] * scales[index] * weight
        cursor = 0
        for tensor in layout.tensors:
            overlap_start = max(block.start, tensor.start)
            overlap_end = min(block.end, tensor.end)
            if overlap_start >= overlap_end:
                continue
            size = overlap_end - overlap_start
            local_start = overlap_start - tensor.start
            flat = result[tensor.name].reshape(-1)
            flat[local_start:local_start + size].add_(
                delta[cursor:cursor + size].to(device=flat.device, dtype=flat.dtype)
            )
            cursor += size
        if cursor != delta.numel():
            raise RuntimeError(f"FedScale block {block_id} application was incomplete")
    return result


def encoded_nbytes(encoded: Mapping[str, torch.Tensor]) -> int:
    """Return the actual serialized tensor payload estimate, including metadata."""
    return sum(value.numel() * value.element_size() for value in encoded.values())
