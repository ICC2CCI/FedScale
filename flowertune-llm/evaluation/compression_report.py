"""Compression-utility report: bytes/round and quantization error (Experiment E3).

Implements the evaluation experiment E3 from the FedScale design document §11.6:

    Compare:
        No compression
        INT8 only
        Public mask 25%
        Public mask 10%
        Public mask 5%
        Public mask + INT8
        Public mask + INT8 + residual

    Metrics:
        bytes/round
        transmission time (estimated)
        validation loss / perplexity
        downstream task metrics
        quantization error (L2)

This module provides the compression-side measurement; the utility-side
metrics (loss, ROUGE-L, etc.) are obtained by running the evaluation modules
on models trained under each compression configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# torch is imported lazily inside functions to keep this module importable
# without the full training stack installed.


@dataclass(frozen=True)
class CompressionResult:
    """Result of measuring one compression configuration."""

    config_name: str
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    quantization_error_l2: float | None
    quantization_error_relative: float | None
    mask_ratio: float | None
    int8_enabled: bool
    residual_enabled: bool
    extra: dict


def _tensor_bytes(t) -> int:
    """Return the byte size of a tensor."""
    import torch
    return t.numel() * t.element_size()


def _state_dict_bytes(state: dict) -> int:
    """Return total bytes of all tensors in a state dict."""
    return sum(_tensor_bytes(t) for t in state.values())


def measure_int8_quantization(
    state: dict,
    block_size: int = 1048576,
) -> tuple[int, float, float]:
    """Measure INT8 block-wise quantization error and compressed size.

    Quantizes each tensor in *state* into blocks of *block_size* elements,
    using per-block scale (max-abs) and INT8 values.  Returns the compressed
    byte count and the L2 / relative quantization error.

    Args:
        state: Dict of CPU tensors (the federated update or model state).
        block_size: Canonical block size in elements.

    Returns:
        ``(compressed_bytes, l2_error, relative_error)``
    """
    import torch

    total_original_bytes = 0
    total_compressed_bytes = 0
    total_sq_error = 0.0
    total_sq_norm = 0.0

    for key, tensor in state.items():
        if not tensor.is_floating_point():
            total_original_bytes += _tensor_bytes(tensor)
            total_compressed_bytes += _tensor_bytes(tensor)
            continue

        flat = tensor.detach().cpu().float().flatten()
        n = flat.numel()
        total_original_bytes += _tensor_bytes(tensor)

        # Block-wise INT8 quantization.
        num_blocks = (n + block_size - 1) // block_size
        # Compressed storage: int8 values + float32 scale per block.
        compressed_bytes = n + num_blocks * 4
        total_compressed_bytes += compressed_bytes

        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, n)
            block = flat[start:end]
            scale = block.abs().max().item()
            if scale == 0:
                continue
            quantized = torch.clamp(
                torch.round(block / scale), -127, 127
            ).to(torch.int8)
            dequantized = quantized.float() * scale
            total_sq_error += float((block - dequantized).pow(2).sum().item())
            total_sq_norm += float(block.pow(2).sum().item())

    l2_error = math.sqrt(total_sq_error) if total_sq_error > 0 else 0.0
    relative_error = (
        l2_error / math.sqrt(total_sq_norm) if total_sq_norm > 0 else 0.0
    )

    return total_compressed_bytes, l2_error, relative_error


def measure_mask_compression(
    state: dict,
    mask_ratio: float,
    block_size: int = 1048576,
) -> tuple[int, int, float]:
    """Measure the effect of public block masking (without INT8).

    Selects ``mask_ratio`` fraction of canonical blocks (by round-robin) and
    measures the compressed size and the fraction of parameters transmitted.

    Args:
        state: Dict of CPU tensors.
        mask_ratio: Fraction of blocks to transmit (0 < ratio <= 1).
        block_size: Canonical block size in elements.

    Returns:
        ``(transmitted_bytes, total_bytes, param_coverage)``
    """
    import torch

    total_elements = 0
    total_bytes = 0

    for key, tensor in state.items():
        if torch.is_floating_point(tensor):
            total_elements += tensor.numel()
            total_bytes += _tensor_bytes(tensor)
        else:
            total_bytes += _tensor_bytes(tensor)

    num_blocks = (total_elements + block_size - 1) // block_size
    selected_blocks = max(1, int(num_blocks * mask_ratio))

    # Each selected block transmits block_size float16 values.
    transmitted_bytes = selected_blocks * block_size * 2  # FP16
    param_coverage = selected_blocks / num_blocks if num_blocks > 0 else 0.0

    return transmitted_bytes, total_bytes, param_coverage


def compute_compression_report(
    state: dict,
    configurations: list[dict] | None = None,
    block_size: int = 1048576,
) -> list[CompressionResult]:
    """Compute a compression-utility report across multiple configurations.

    Each configuration dict can contain:
        ``name``: Label for the configuration.
        ``int8``: bool — enable INT8 quantization.
        ``mask_ratio``: float or None — public block mask ratio.
        ``residual``: bool — whether error feedback is enabled (affects reporting only).

    Args:
        state: Dict of CPU tensors (the update or model state).
        configurations: List of config dicts.  If None, uses the E3 default set.
        block_size: Canonical block size.

    Returns:
        List of ``CompressionResult`` for each configuration.
    """
    if configurations is None:
        configurations = [
            {"name": "no_compression", "int8": False, "mask_ratio": None, "residual": False},
            {"name": "int8_only", "int8": True, "mask_ratio": None, "residual": False},
            {"name": "mask_25pct", "int8": False, "mask_ratio": 0.25, "residual": False},
            {"name": "mask_10pct", "int8": False, "mask_ratio": 0.10, "residual": False},
            {"name": "mask_5pct", "int8": False, "mask_ratio": 0.05, "residual": False},
            {"name": "mask_25pct_int8", "int8": True, "mask_ratio": 0.25, "residual": False},
            {"name": "mask_10pct_int8", "int8": True, "mask_ratio": 0.10, "residual": False},
            {"name": "mask_5pct_int8", "int8": True, "mask_ratio": 0.05, "residual": False},
            {"name": "mask_5pct_int8_residual", "int8": True, "mask_ratio": 0.05, "residual": True},
        ]

    original_bytes = _state_dict_bytes(state)
    results: list[CompressionResult] = []

    for config in configurations:
        name = config["name"]
        use_int8 = config.get("int8", False)
        mask_ratio = config.get("mask_ratio")
        use_residual = config.get("residual", False)

        if mask_ratio is not None and mask_ratio < 1.0:
            transmitted_bytes, _, param_coverage = measure_mask_compression(
                state, mask_ratio, block_size
            )
            if use_int8:
                # INT8 reduces transmitted bytes by ~50% (FP16 → INT8 + scales).
                transmitted_bytes = transmitted_bytes // 2
        elif use_int8:
            transmitted_bytes, l2_error, rel_error = measure_int8_quantization(
                state, block_size
            )
            param_coverage = 1.0
        else:
            transmitted_bytes = original_bytes
            l2_error = 0.0
            rel_error = 0.0
            param_coverage = 1.0

        # Compute quantization error if INT8 and not already computed.
        if use_int8 and mask_ratio is None:
            pass  # Already computed above.
        elif use_int8 and mask_ratio is not None:
            _, l2_error, rel_error = measure_int8_quantization(
                state, block_size
            )
        else:
            l2_error = 0.0
            rel_error = 0.0

        compression_ratio = (
            original_bytes / transmitted_bytes
            if transmitted_bytes > 0 else 1.0
        )

        results.append(CompressionResult(
            config_name=name,
            original_bytes=original_bytes,
            compressed_bytes=transmitted_bytes,
            compression_ratio=round(compression_ratio, 2),
            quantization_error_l2=round(l2_error, 4) if l2_error else None,
            quantization_error_relative=round(rel_error, 6) if rel_error else None,
            mask_ratio=mask_ratio,
            int8_enabled=use_int8,
            residual_enabled=use_residual,
            extra={"param_coverage": round(param_coverage, 4)},
        ))

    return results
