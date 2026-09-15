"""SecAgg 定点量化：浮点 delta ↔ Z_q 整数域转换。

核心约束（spec section 9.2）：
    N_max × Q_max < q / 2
  即 N 个 client 的量化值在 Z_q 中相加不会回绕（overflow budget）。

参数说明：
    q     = 2^modulus_bits        模数，决定传输字节和运算域
    Q_max = 2^(modulus_bits-1) - 1  量化值上界（有符号范围的一半）
    scale = cohort 共享的换算因子
            q_int = round(delta / scale)，clip 到 [-Q_max, Q_max]
            delta ≈ q_int × scale

默认配置（int16）：
    modulus_bits = 16, q = 65536, Q_max = 32767
    2-client overflow: 2 × 16383 < 32768 ✓ (Q_max 留一半给聚合)
    传输量 = 2 bytes/元素（与 fp16 相同）
    14-bit 有效精度（优于 fp16 的 10-bit mantissa）

接口全部参数化，后续切换 int8/int24/int32 只需改 yaml 配置：
    secagg_modulus_bits: 16   # 8/16/24/32
    secagg_scale: 0.0         # 0=自适应（后续实现）
    secagg_stochastic_rounding: false
"""
from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

DEFAULT_MODULUS_BITS = 16
DEFAULT_Q = 1 << DEFAULT_MODULUS_BITS  # 65536
# Q_max 留一半给 overflow：Q_max = q/4 - 1，保证 2*Q_max < q/2
DEFAULT_Q_MAX = (DEFAULT_Q // 4) - 1  # 16383
# scale 选择：可表示范围 = ±(Q_max × scale)
# delta 实际范围可达 ±2.0（embed_tokens/lm_head），用 scale=2^-12 → ±4.0 覆盖
DEFAULT_SCALE = 2.0 ** -(DEFAULT_MODULUS_BITS - 4)  # 2^-12 ≈ 2.44e-4


def compute_q_max(modulus_bits: int, n_clients: int = 2) -> int:
    """根据模数位宽和 client 数计算 Q_max。

    约束: n_clients × Q_max < q / 2
    → Q_max < q / (2 × n_clients)
    取 Q_max = q // (2 × n_clients) - 1 留余量。
    """
    q = 1 << modulus_bits
    return q // (2 * n_clients) - 1


def overflow_check(n_clients: int, q_max: int, q: int) -> bool:
    """验证 overflow budget: n_clients × q_max < q / 2。"""
    return n_clients * q_max < q // 2


# ---------------------------------------------------------------------------
# 量化 / 反量化
# ---------------------------------------------------------------------------


def quantize_to_zq(
    delta: torch.Tensor,
    scale: float,
    q_max: int,
    q: int,
    stochastic: bool = False,
) -> torch.Tensor:
    """浮点 delta → Z_q 整数（int64 tensor）。

    量化: q_int = round(delta / scale)，clip 到 [-q_max, q_max]
    映射到 Z_q: q_zq = q_int mod q （非负，[0, q)）

    stochastic=True 时用随机舍入（期望无偏）。
    """
    delta_f32 = delta.to(dtype=torch.float32)
    scaled = delta_f32 / float(scale)

    if stochastic:
        # 随机舍入: floor(x) + Bernoulli(frac(x))
        floor_val = torch.floor(scaled)
        frac = scaled - floor_val
        rand = torch.rand_like(frac)
        rounded = floor_val + (rand < frac).to(dtype=torch.float32)
    else:
        rounded = torch.round(scaled)

    # clip 到 [-q_max, q_max]
    clipped = rounded.clamp(-q_max, q_max)

    # 映射到 Z_q (非负): q_zq = (q_int % q + q) % q
    q_zq = clipped.to(dtype=torch.int64) % q
    q_zq = (q_zq + q) % q  # 确保非负
    return q_zq


def dequantize_from_zq(
    q_zq: torch.Tensor,
    scale: float,
    q: int,
) -> torch.Tensor:
    """Z_q 整数 → 浮点 delta（有符号解码）。

    1. q_int = q_zq mod q → [0, q)
    2. 有符号解码: if q_int > q/2: q_int -= q  → [-q/2, q/2)
    3. delta = q_int × scale
    """
    q_int = q_zq.to(dtype=torch.int64) % q
    # 有符号解码
    half_q = q // 2
    signed = torch.where(q_int > half_q, q_int - q, q_int)
    delta = signed.to(dtype=torch.float32) * float(scale)
    return delta


def aggregate_zq(z_list: list, q: int) -> torch.Tensor:
    """在 Z_q 中求和多个 masked 向量。"""
    result = z_list[0].to(dtype=torch.int64).clone()
    for z in z_list[1:]:
        result = (result + z.to(dtype=torch.int64)) % q
    return result


def unmask_aggregate(
    sum_z: torch.Tensor,
    sum_self_mask: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """从聚合 masked 值中移除 self mask 总和。

    sum_q = (sum_z - sum_self_mask) mod q
    """
    diff = (sum_z.to(dtype=torch.int64) - sum_self_mask.to(dtype=torch.int64)) % q
    return diff


# ---------------------------------------------------------------------------
# int16 packed 编解码（2 bytes/元素，little-endian）
# ---------------------------------------------------------------------------


def pack_int16(q_zq: torch.Tensor, q: int) -> bytes:
    """把 Z_q 整数向量打包成 2 bytes/元素的 bytes（用于传输）。

    q 必须 ≤ 2^16。每个值取低 16-bit，little-endian。
    """
    if q > (1 << 16):
        raise ValueError(f"pack_int16 requires q <= 65536, got q={q}")
    arr = q_zq.to(dtype=torch.int64).cpu().numpy().astype(np.uint16)
    return arr.tobytes()


def unpack_int16(data: bytes, length: int) -> torch.Tensor:
    """从 2 bytes/元素的 bytes 解包成 Z_q 整数向量。"""
    arr = np.frombuffer(data, dtype=np.uint16, count=length).astype(np.int64)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# 通用 packed 编解码（支持 1/2/3/4 bytes）
# ---------------------------------------------------------------------------


def pack_zq(q_zq: torch.Tensor, modulus_bits: int) -> bytes:
    """把 Z_q 整数向量打包成 bytes（根据位宽自动选择字节数）。"""
    n_bytes = (modulus_bits + 7) // 8
    arr = q_zq.to(dtype=torch.int64).cpu().numpy()
    if n_bytes == 1:
        return arr.astype(np.uint8).tobytes()
    elif n_bytes == 2:
        return arr.astype(np.uint16).tobytes()
    elif n_bytes == 3:
        # 3-byte little-endian packing
        arr32 = arr.astype(np.uint32) & 0xFFFFFF
        out = np.empty(len(arr32) * 3, dtype=np.uint8)
        out[0::3] = arr32 & 0xFF
        out[1::3] = (arr32 >> 8) & 0xFF
        out[2::3] = (arr32 >> 16) & 0xFF
        return out.tobytes()
    elif n_bytes == 4:
        return arr.astype(np.uint32).tobytes()
    else:
        raise ValueError(f"unsupported modulus_bits={modulus_bits} (n_bytes={n_bytes})")


def unpack_zq(data: bytes, length: int, modulus_bits: int) -> torch.Tensor:
    """从 bytes 解包成 Z_q 整数向量。"""
    n_bytes = (modulus_bits + 7) // 8
    if n_bytes == 1:
        arr = np.frombuffer(data, dtype=np.uint8, count=length).astype(np.int64)
    elif n_bytes == 2:
        arr = np.frombuffer(data, dtype=np.uint16, count=length).astype(np.int64)
    elif n_bytes == 3:
        raw = np.frombuffer(data, dtype=np.uint8, count=length * 3)
        arr = raw[0::3].astype(np.uint32) | (raw[1::3].astype(np.uint32) << 8) | (raw[2::3].astype(np.uint32) << 16)
        arr = arr.astype(np.int64)
    elif n_bytes == 4:
        arr = np.frombuffer(data, dtype=np.uint32, count=length).astype(np.int64)
    else:
        raise ValueError(f"unsupported modulus_bits={modulus_bits}")
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# 辅助：获取 delta 的统计信息（用于自适应 scale）
# ---------------------------------------------------------------------------


def compute_adaptive_scale(
    deltas: list,
    q_max: int,
    coverage_factor: float = 1.0,
) -> float:
    """根据 delta 的 max|value| 计算自适应 scale。

    scale = max(|delta|) / q_max × coverage_factor
    coverage_factor=1.0: 覆盖全部范围
    coverage_factor=0.95: 保留 5% 余量（极端值会被 clip）

    deltas: list of torch.Tensor（各 client 的 delta）
    """
    max_abs = 0.0
    for delta in deltas:
        if delta is not None:
            m = float(delta.abs().max().item())
            if m > max_abs:
                max_abs = m
    if max_abs < 1e-12:
        return 1.0  # 全零 delta，scale 无意义
    return max_abs / float(q_max) * float(coverage_factor)
