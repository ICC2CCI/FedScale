"""Global Top-k + ResidualBuffer。

FSDP-aware global Top-k（§6.7）：
- 各 rank 在本地 shard 做 Top-(k·oversample)
- all-gather 候选 (coord, value) → 全局 Top-k
- 通信量 = oversample·k·(4+4) bytes

非 FSDP（单进程）直接 torch.topk。

ResidualBuffer（§6.9）：
- r_0 = 0
- z_i = g_i + r_i
- Top-k 对 z_i 执行
- r_{i+1} = z_i - Sparse(S_i, v_i)
- 可选 clip/decay 防累积爆炸
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .count_sketch import find_layer_for_coord


def global_topk_flat(
    flat_grad: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """单进程 global Top-k。

    Args:
        flat_grad: (d,) 展平梯度
        k: Top-k 数量

    Returns:
        coords: (k,) 全模型展平坐标
        values: (k,) 对应值
    """
    k = min(k, flat_grad.numel())
    vals, idxs = torch.topk(flat_grad.abs(), k)
    return idxs, flat_grad[idxs]


def fsdp_global_topk(
    local_flat_grad: torch.Tensor,
    local_offset: int,
    k: int,
    oversample: int = 3,
    group: Optional[torch.distributed.group] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FSDP-aware global Top-k（§6.7）。

    各 rank 在本地 shard 做 Top-(k·oversample)，all-gather 候选后取全局 Top-k。

    Args:
        local_flat_grad: 本 rank 的展平梯度 shard
        local_offset: 本 shard 在全模型展平空间中的起始坐标
        k: 全局 Top-k 数量
        oversample: 过采样倍数
        group: torch.distributed group（None=world）

    Returns:
        coords: (k,) 全模型展平坐标
        values: (k,) 对应值
    """
    if not torch.distributed.is_initialized():
        # 单进程退化
        return global_topk_flat(local_flat_grad, k)

    world = torch.distributed.get_world_size(group)
    local_k = min(k * oversample, local_flat_grad.numel())
    local_vals, local_idxs = torch.topk(local_flat_grad.abs(), local_k)
    local_coords = local_idxs + local_offset
    local_pairs = torch.stack([local_coords.to(torch.float64), local_vals.to(torch.float64)], dim=1)

    # all-gather 候选
    gathered = [torch.zeros_like(local_pairs) for _ in range(world)]
    torch.distributed.all_gather(gathered, local_pairs, group=group)
    all_pairs = torch.cat(gathered, dim=0)

    # 全局 Top-k
    coords_f = all_pairs[:, 0]
    vals_f = all_pairs[:, 1]
    _, top_idxs = torch.topk(vals_f.abs(), min(k, vals_f.numel()))
    coords = coords_f[top_idxs].to(torch.long)
    values = vals_f[top_idxs].to(local_flat_grad.dtype)
    return coords, values


def flat_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[Tuple[str, int, int]]]:
    """把模型 state_dict 展平为 1D tensor。

    Returns:
        flat: (d,) 展平参数
        layer_meta: [(name, start, end), ...]
    """
    parts: List[torch.Tensor] = []
    meta: List[Tuple[str, int, int]] = []
    offset = 0
    for name, t in state_dict.items():
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        n = t.numel()
        if n == 0:
            continue
        parts.append(t.detach().reshape(-1).to(torch.float32))
        meta.append((name, offset, offset + n))
        offset += n
    flat = torch.cat(parts) if parts else torch.zeros(0)
    return flat, meta


def unflat_to_state_dict(
    flat: torch.Tensor,
    layer_meta: List[Tuple[str, int, int]],
    ref_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """把 1D tensor 还原为 state_dict 形状。"""
    out: Dict[str, torch.Tensor] = {}
    for name, start, end in layer_meta:
        ref = ref_state_dict[name]
        chunk = flat[start:end].reshape(ref.shape).to(ref.dtype)
        out[name] = chunk
    return out


def compute_delta(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """计算训练前后模型差分 g = after - before（逐层）。"""
    delta: Dict[str, torch.Tensor] = {}
    for name in after:
        if name in before and torch.is_tensor(after[name]) and after[name].is_floating_point():
            delta[name] = (after[name].to(torch.float32) - before[name].to(torch.float32))
    return delta


class ResidualBuffer:
    """残差缓冲（§6.9）。

    r_0 = 0
    z_i = g_i + r_i
    Top-k 对 z_i 执行
    r_{i+1} = z_i - Sparse(S_i, v_i)
    可选 clip_norm / decay
    """

    def __init__(
        self,
        layer_meta: List[Tuple[str, int, int]],
        clip_norm: float = 0.0,
        decay: float = 1.0,
    ):
        self.layer_meta = layer_meta
        self.clip_norm = clip_norm
        self.decay = decay
        d = layer_meta[-1][2] if layer_meta else 0
        self.buffer = torch.zeros(d, dtype=torch.float32)

    def add_and_select(
        self,
        flat_grad: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """z = grad + r；对 z 做 Top-k；更新 r = z - Sparse(S, v)。"""
        z = flat_grad.to(torch.float32) + self.buffer
        coords, values = global_topk_flat(z, k)

        # 重建稀疏向量并更新残差
        sparse = torch.zeros_like(z)
        sparse[coords] = values
        r_new = z - sparse
        if self.decay != 1.0:
            r_new = r_new * self.decay
        if self.clip_norm > 0:
            norm = r_new.norm()
            if norm > self.clip_norm:
                r_new = r_new * (self.clip_norm / norm)
        self.buffer = r_new
        return coords, values

    def reset(self) -> None:
        d = self.buffer.numel()
        self.buffer = torch.zeros(d, dtype=torch.float32)


def count_topk_per_layer(
    coords: torch.Tensor,
    layout: SketchLayout,
) -> Dict[str, int]:
    """统计各层 Top-k 命中计数（用于 calibration）。"""
    counts: Dict[str, int] = {}
    for layer in layout.layers:
        counts[layer.name] = 0
    for c in coords.tolist():
        li = find_layer_for_coord(layout, int(c))
        if li >= 0:
            counts[layout.layers[li].name] += 1
    return counts
