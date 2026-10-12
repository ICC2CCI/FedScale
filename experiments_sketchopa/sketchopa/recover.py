"""从聚合 Sketch 恢复全局 Top-K_recover 坐标。

x̂_j = median_r ξ_r(j) · U[r, h_r(j)]
K_recover = min(k · |C| · ρ, d)
recover_k_mode = active_union：对所有 active client 候选坐标并集做估计
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .count_sketch import CountSketchEncoder, find_layer_for_coord
from .layout import SketchLayout


def recover_topk(
    agg_sketch: torch.Tensor,
    encoder: CountSketchEncoder,
    layout: SketchLayout,
    candidate_coords: torch.Tensor,
    k_recover: int,
    clip_norm: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """从聚合 Sketch 恢复全局 Top-K_recover。

    Args:
        agg_sketch: (R × Σ B_l,) 聚合后的 Sketch
        encoder: CountSketchEncoder（与编码时同 layout+seed）
        layout: SketchLayout
        candidate_coords: 候选坐标并集（active_union 模式：所有 client 的 Top-k 坐标去重）
        k_recover: 恢复数量 = min(k·|C|·ρ, d)
        clip_norm: 恢复后按范数 clip（>0 生效）

    Returns:
        coords: (k_recover,) 恢复坐标
        values: (k_recover,) 恢复值
        stats: 恢复统计
    """
    candidates = candidate_coords.unique().to(torch.long)
    if candidates.numel() == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0), {"num_candidates": 0}

    # 对每个候选坐标估计值（按层批量处理）
    estimates = torch.zeros(candidates.numel(), dtype=torch.float32)
    # 找各候选所属层
    from .count_sketch import _find_layer_vec
    layer_starts = torch.tensor([l.start for l in layout.layers], dtype=torch.long)
    layer_ends = torch.tensor([l.end for l in layout.layers], dtype=torch.long)
    li = _find_layer_vec(candidates, layer_starts, layer_ends)

    for layer_idx in range(len(layout.layers)):
        mask = (li == layer_idx)
        if not mask.any():
            continue
        layer = layout.layers[layer_idx]
        if layer.is_dense:
            continue
        layer_coords = candidates[mask]
        local_coords = layer_coords - layer.start
        estimates[mask] = encoder.estimate_batch(agg_sketch, layer_idx, local_coords)

    # 按 |estimate| 取 Top-K_recover
    k = min(k_recover, candidates.numel())
    _, top_idxs = torch.topk(estimates.abs(), k)
    coords = candidates[top_idxs]
    values = estimates[top_idxs]

    # clip
    if clip_norm > 0 and values.numel() > 0:
        norm = values.norm()
        if norm > clip_norm:
            values = values * (clip_norm / norm)

    stats = {
        "num_candidates": int(candidates.numel()),
        "k_recover": int(k),
        "estimate_max_abs": float(estimates.abs().max().item()) if estimates.numel() > 0 else 0.0,
        "estimate_mean_abs": float(estimates.abs().mean().item()) if estimates.numel() > 0 else 0.0,
        "recovered_norm": float(values.norm().item()) if values.numel() > 0 else 0.0,
    }
    return coords, values, stats


def compute_k_recover(k: int, num_active_clients: int, rho: float, d: int) -> int:
    """K_recover = min(k · |C| · ρ, d)。"""
    return min(int(k * num_active_clients * rho), d)


def build_sparse_update(
    coords: torch.Tensor,
    values: torch.Tensor,
    d: int,
) -> torch.Tensor:
    """把稀疏 (coords, values) 还原为稠密 1D tensor。"""
    out = torch.zeros(d, dtype=torch.float32)
    if coords.numel() > 0:
        out[coords] = values
    return out


def apply_update_to_state_dict(
    flat_update: torch.Tensor,
    layer_meta: List[Tuple[str, int, int]],
    ref_state_dict: Dict[str, torch.Tensor],
    lr_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """把 1D 更新加到 state_dict 上。

    new_param = ref_param + lr_scale * update_chunk
    """
    out: Dict[str, torch.Tensor] = {}
    for name, start, end in layer_meta:
        ref = ref_state_dict[name]
        chunk = flat_update[start:end].reshape(ref.shape).to(ref.dtype)
        out[name] = ref + lr_scale * chunk
    return out
