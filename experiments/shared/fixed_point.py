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
    secagg_scale: 0.0         # 0=per-window 公开聚合；>0=固定全局 scale
    secagg_stochastic_rounding: false
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Hadamard 旋转（量化前压低动态范围）
# ---------------------------------------------------------------------------

def generate_rademacher_signs(length: int, seed: int) -> torch.Tensor:
    """生成 ±1 随机符号向量（Rademacher 分布），确定性种子。

    所有 client 用相同种子 → 相同 D → 整数域求和兼容。
    """
    g = torch.Generator()
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    signs = torch.randint(0, 2, (length,), generator=g, dtype=torch.int32)
    return (signs * 2 - 1).to(dtype=torch.float32)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    if is_power_of_two(n):
        return n
    return 1 << (n - 1).bit_length()


def hadamard_work_length(n: int, enabled: bool) -> int:
    """Hadamard 工作长度：开启时 pad 到 2 的幂，否则保持原长。"""
    n = int(n)
    if n <= 0:
        return 0
    if not enabled:
        return n
    return next_power_of_two(n)


def pad_to_length(x: torch.Tensor, length: int) -> torch.Tensor:
    """把向量 pad/截成 length（右侧补零）。"""
    flat = x.reshape(-1).to(dtype=torch.float32)
    length = int(length)
    if length <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if int(flat.numel()) == length:
        return flat.contiguous()
    out = torch.zeros(length, dtype=torch.float32)
    n = min(int(flat.numel()), length)
    if n > 0:
        out[:n] = flat[:n]
    return out


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform，长度必须是 2 的幂。

    向量化实现：只有 log2(n) 次张量运算。双缓冲，全程只分配两块 n 长向量。

    H @ x 的快速实现。Hadamard 矩阵 H 满足 H @ H^T = n @ I，因此 H^(-1) = H / n。
    本函数返回 H @ x（不除 n），逆变换用 fwht(y) / n。不原地改写输入。
    """
    n = int(x.numel())
    if not is_power_of_two(n):
        raise ValueError(f"FWHT requires power-of-2 length, got {n}")
    src = x.to(dtype=torch.float32).reshape(n).contiguous().clone()
    if n <= 1:
        return src
    dst = torch.empty_like(src)
    h = 1
    while h < n:
        s = src.view(-1, 2, h)
        d = dst.view(-1, 2, h)
        a = s[:, 0, :]
        b = s[:, 1, :]
        d[:, 0, :] = a + b
        d[:, 1, :] = a - b
        src, dst = dst, src
        h *= 2
    return src


def hadamard_transform(
    x: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """前向 Hadamard 旋转：y = FWHT(D ⊙ x)。

    signs: ±1 随机符号向量（与 x 等长），所有 client 共享。
    返回旋转后的向量，动态范围从 O(||x||_inf) 压到 O(||x||_2 / sqrt(n))。
    """
    return fwht(x * signs)


def inverse_hadamard_transform(
    y: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """逆 Hadamard 旋转：x = D ⊙ FWHT(y) / n。

    Hadamard 矩阵性质：H^(-1) = H / n，D^(-1) = D（D 是 ±1 对角矩阵）。
    """
    n = y.numel()
    return fwht(y) * signs / float(n)


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
# coverage>1：把 amax 映射到 Q_max/coverage，给当轮最大值留余量，避免 clip。
# 旧代码用 0.9，会让 amax 自身被 clip，scale 每轮收缩。
SCALE_COVERAGE = 1.05
# 主路径：client 在 key-announce 上报当轮 per-window amax，server 取 max_k → scale。
# 下面 PUBLIC_AGG_HEADROOM / update_public_block_scales 仅作回退（client 未报 amax 时），
# 用「上一轮公开聚合 delta」估下一轮 scale，不再作为主路径。
PUBLIC_AGG_HEADROOM = 3.0
ROUND1_PUBLIC_SCALE = 2.0 ** -20  # 回退常数（client 未报 amax 时）
_ZERO_AMAX = 1e-12


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


def _round_scaled(scaled: torch.Tensor, stochastic: bool) -> torch.Tensor:
    if stochastic:
        floor_val = torch.floor(scaled)
        frac = scaled - floor_val
        rand = torch.rand_like(frac)
        return floor_val + (rand < frac).to(dtype=torch.float32)
    return torch.round(scaled)


def _to_zq(clipped: torch.Tensor, q: int) -> torch.Tensor:
    q_zq = clipped.to(dtype=torch.int64) % q
    return (q_zq + q) % q


def quantization_error_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Dict[str, float]:
    """量化误差指标：relative L2、cosine similarity、SQNR (dB)。"""
    x = original.to(dtype=torch.float32).reshape(-1)
    y = reconstructed.to(dtype=torch.float32).reshape(-1)
    if x.numel() == 0:
        return {"rel_l2": 0.0, "cosine": 1.0, "sqnr_db": float("inf")}
    diff = x - y
    x_norm = float(torch.norm(x).item())
    y_norm = float(torch.norm(y).item())
    diff_norm = float(torch.norm(diff).item())
    rel_l2 = diff_norm / max(x_norm, 1e-12)
    if x_norm <= 0.0 and y_norm <= 0.0:
        cosine = 1.0
    elif x_norm <= 0.0 or y_norm <= 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(x, y).item()) / (x_norm * y_norm)
    mse = float((diff * diff).mean().item())
    signal = float((x * x).mean().item())
    if mse <= 0.0:
        sqnr_db = float("inf")
    elif signal <= 0.0:
        sqnr_db = 0.0
    else:
        sqnr_db = 10.0 * math.log10(signal / mse)
    return {"rel_l2": rel_l2, "cosine": cosine, "sqnr_db": sqnr_db}


def quantize_to_zq_with_feedback(
    delta: torch.Tensor,
    scale: float,
    q_max: int,
    q: int,
    stochastic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """量化并返回 residual + clip/zero 统计，供 error-feedback 使用。

    residual = delta - dequant(quant(delta))，选中 block 应把 residual 写入 memory。
    """
    if float(scale) <= 0.0:
        raise ValueError(f"quantization scale must be > 0, got {scale}")
    delta_f32 = delta.to(dtype=torch.float32)
    scaled = delta_f32 / float(scale)
    rounded = _round_scaled(scaled, stochastic)

    n = max(int(rounded.numel()), 1)
    n_clip = int((rounded.abs() > float(q_max)).sum().item())
    clipped = rounded.clamp(-q_max, q_max)
    q_zq = _to_zq(clipped, q)

    reconstructed = dequantize_from_zq(q_zq, scale, q)
    residual = delta_f32 - reconstructed

    half_q = q // 2
    q_int = q_zq % q
    signed = torch.where(q_int > half_q, q_int - q, q_int)
    n_zero = int((signed == 0).sum().item())
    amax = float(delta_f32.abs().max().item()) if n else 0.0
    metrics = quantization_error_metrics(delta_f32, reconstructed)
    stats = {
        "clip_frac": float(n_clip) / float(n),
        "zero_frac": float(n_zero) / float(n),
        "amax": amax,
        "scale": float(scale),
        "rel_l2": metrics["rel_l2"],
        "cosine": metrics["cosine"],
        "sqnr_db": metrics["sqnr_db"],
    }
    return q_zq, residual, stats


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
    q_zq, _residual, _stats = quantize_to_zq_with_feedback(
        delta, scale, q_max, q, stochastic=stochastic,
    )
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


def compute_window_scale(
    amax: float,
    q_max: int,
    coverage: float = SCALE_COVERAGE,
) -> float:
    """由 max|delta| 计算定点 scale。

    scale = amax / q_max * coverage
    coverage>1：amax 映射到 Q_max/coverage，不 clip 该 amax。
    主路径：amax 来自 client 当轮上报的 per-window max|delta|（secagg_coordinator._finalize_window_scales）。
    回退路径：amax 来自上一轮公开聚合（update_public_block_scales, headroom=PUBLIC_AGG_HEADROOM）。
    """
    if amax < _ZERO_AMAX:
        return 1.0
    return float(amax) / float(q_max) * float(coverage)


def block_scale_key(key_name: str, start: int, end: int) -> str:
    """跨 round 稳定的 block 身份（window_id/gidx 每轮会变）。"""
    return f"{key_name}:{int(start)}:{int(end)}"


def update_public_block_scales(
    prev_scales: dict,
    windows,
    agg_delta: dict,
    q_max: int,
    headroom: float = PUBLIC_AGG_HEADROOM,
) -> dict:
    """[回退路径] 用本轮公开聚合 delta 更新下一轮 per-block scale。

    只读 agg_delta（所有 client 之和 / N），不读任何单 client 统计。
    全零 block 保留 prev；未见过的 block 不写入。
    主路径是 client 当轮上报 amax（见 secagg_coordinator._finalize_window_scales），
    本函数仅作回退/兼容用。
    """
    out = dict(prev_scales or {})
    for window in windows:
        key = block_scale_key(window.key_name, window.start, window.end)
        amax = 0.0
        for item in (agg_delta or {}).get(window.key_name, []):
            s, e, sd = item[0], item[1], item[2]
            if int(s) == int(window.start) and int(e) == int(window.end):
                amax = float(sd.abs().max().item())
                break
        if amax < _ZERO_AMAX:
            continue
        out[key] = compute_window_scale(amax, q_max, coverage=headroom)
    return out


def compute_adaptive_scale(
    deltas: list,
    q_max: int,
    coverage_factor: float = SCALE_COVERAGE,
) -> float:
    """根据一组 delta 的 max|value| 计算 scale（测试/调试用）。

    生产路径：client 当轮上报 per-window amax → server 取 max_k（_finalize_window_scales）。
    """
    max_abs = 0.0
    for delta in deltas:
        if delta is not None:
            m = float(delta.abs().max().item())
            if m > max_abs:
                max_abs = m
    return compute_window_scale(max_abs, q_max, coverage=coverage_factor)
