"""Sketch 值量化。

支持两种模式：
- float16：直接转 fp16 传输（默认，对齐 Clover）
- int16 public scale：用固定 scale 量化到 int16，传输 int16 + scale

记录 bucket 值域分布用于调试（§6.5）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class QuantStats:
    mode: str
    max_abs: float
    mean_abs: float
    num_zeros: int
    total: int
    overflow: int = 0  # int16 模式下溢出个数


def quantize_fp16(sketch: torch.Tensor) -> Tuple[torch.Tensor, QuantStats]:
    """fp16 量化（直接 cast）。"""
    q = sketch.to(torch.float16)
    stats = QuantStats(
        mode="float16",
        max_abs=float(sketch.abs().max().item()) if sketch.numel() > 0 else 0.0,
        mean_abs=float(sketch.abs().mean().item()) if sketch.numel() > 0 else 0.0,
        num_zeros=int((sketch == 0).sum().item()),
        total=sketch.numel(),
    )
    return q, stats


def quantize_int16(
    sketch: torch.Tensor,
    scale: float = 1_000_000.0,
    q_max: int = 32767,
) -> Tuple[torch.Tensor, float, QuantStats]:
    """int16 public scale 量化。

    q = round(x * scale)，clip 到 [-q_max, q_max]
    反量化：x' = q / scale
    """
    scaled = sketch * scale
    q = torch.round(scaled).clamp(-q_max, q_max).to(torch.int16)
    overflow = int((scaled.abs() > q_max).sum().item())
    stats = QuantStats(
        mode="int16",
        max_abs=float(sketch.abs().max().item()) if sketch.numel() > 0 else 0.0,
        mean_abs=float(sketch.abs().mean().item()) if sketch.numel() > 0 else 0.0,
        num_zeros=int((q == 0).sum().item()),
        total=sketch.numel(),
        overflow=overflow,
    )
    return q, scale, stats


def quantize_to_zq(
    sketch: torch.Tensor,
    scale: float,
    modulus_q: int,
) -> torch.Tensor:
    """量化 sketch 到 Z_q（用于安全聚合 mask 域）。

    1. 量化值 = round(sketch * scale)，clamp 到 [-q/2, q/2-1] 保证 centered 还原正确
    2. mod q 映射到 [0, q)
    """
    q_half = modulus_q // 2
    qz = torch.round(sketch * scale).long().clamp(-q_half, q_half - 1)
    return qz % modulus_q


def zq_to_float(zq: torch.Tensor, scale: float, modulus_q: int) -> torch.Tensor:
    """从 Z_q 反量化回 float（centered 还原）。"""
    q_half = modulus_q // 2
    centered = torch.where(zq > q_half, zq - modulus_q, zq)
    return centered.float() / scale
    return q.to(torch.float32)


def dequantize_int16(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) / scale


def quantize(
    sketch: torch.Tensor,
    mode: str = "float16",
    scale: float = 1_000_000.0,
) -> Tuple[torch.Tensor, float, QuantStats]:
    """统一量化接口。

    Returns:
        q: 量化后 tensor
        scale: int16 模式返回 scale，fp16 返回 0
        stats: 统计
    """
    if mode == "float16":
        q, stats = quantize_fp16(sketch)
        return q, 0.0, stats
    elif mode == "int16":
        q, sc, stats = quantize_int16(sketch, scale=scale)
        return q, sc, stats
    else:
        raise ValueError(f"unknown quant mode: {mode}")


def dequantize_fp16(q: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32)


def dequantize_int16(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) / scale
