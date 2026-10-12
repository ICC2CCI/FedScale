"""按层分区 CountSketch 编码（向量化实现）。

对每个节点的稀疏更新 (S_i, v_i)，按坐标所属层映射到对应层 Sketch 区域。
每层 R 行 × B_l bucket，使用 seeded bucket hash h_{l,r}(j) 和 sign hash ξ_{l,r}(j)。

flat sketch 长度 = R × Σ_l B_l，按 [row][layer_offset + bucket] 展平。

hash 用快速整数运算（mul-shift-mod）替代 SHA256，支持 torch 向量化。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .layout import SketchLayout, LayerLayout

# 大素数用于 mul-hash（用 signed int64 可表示的最大素数）
_HASH_PRIME = 0x7FFFFFFFFFFFFFE7  # < 2^63 的最大素数


def _derive_hash_seeds(sketch_seed: int, layer_idx: int, row: int) -> Tuple[int, int]:
    """为 (layer_idx, row) 派生 (bucket_seed, sign_seed)，掩码到 int64 范围。"""
    import hashlib
    h = hashlib.sha256(
        f"sketchopa|{sketch_seed}|L{layer_idx}|R{row}".encode()
    ).digest()
    # 取 63 bit 确保是正 int64
    bucket_seed = int.from_bytes(h[:8], "little") & 0x7FFFFFFFFFFFFFFF
    sign_seed = int.from_bytes(h[8:16], "little") & 0x7FFFFFFFFFFFFFFF
    return bucket_seed, sign_seed


class CountSketchEncoder:
    """按层分区 CountSketch 编码器（向量化）。

    线性可加：CS(x) + CS(y) = CS(x+y)（同一 layout + seed 下）。
    """

    def __init__(self, layout: SketchLayout, sketch_seed: int):
        self.layout = layout
        self.sketch_seed = sketch_seed

        # 预计算每层每行的 hash seeds（int64 tensor）
        # _bucket_seeds[li][r], _sign_seeds[li][r]
        self._bucket_seeds: List[List[int]] = []
        self._sign_seeds: List[List[int]] = []
        for li, layer in enumerate(layout.layers):
            if layer.is_dense:
                self._bucket_seeds.append([])
                self._sign_seeds.append([])
                continue
            bs_list, ss_list = [], []
            for r in range(layout.rows):
                bs, ss = _derive_hash_seeds(sketch_seed, li, r)
                bs_list.append(bs)
                ss_list.append(ss)
            self._bucket_seeds.append(bs_list)
            self._sign_seeds.append(ss_list)

    @property
    def sketch_len(self) -> int:
        return self.layout.sketch_len

    def _hash_bucket_vec(self, local_coords: torch.Tensor, bucket_seed: int, num_buckets: int) -> torch.Tensor:
        """向量化 bucket hash: h(coord) -> [0, num_buckets)"""
        # 用 mul-shift-mod: (coord * prime ^ seed) >> shift % num_buckets
        c = local_coords.to(torch.int64)
        mixed = (c ^ bucket_seed).mul_(_HASH_PRIME)
        # 高位取模
        return (mixed >> 32) % num_buckets

    def _hash_sign_vec(self, local_coords: torch.Tensor, sign_seed: int) -> torch.Tensor:
        """向量化 sign hash: ξ(coord) -> {-1, +1}"""
        c = local_coords.to(torch.int64)
        mixed = (c ^ sign_seed).mul_(_HASH_PRIME)
        return torch.where((mixed >> 63) & 1 == 0, torch.ones_like(c), -torch.ones_like(c))

    def encode(
        self,
        coords: torch.Tensor,   # (k,) 全模型展平坐标
        values: torch.Tensor,   # (k,) 对应值
    ) -> torch.Tensor:
        """将稀疏更新 (coords, values) 编码为 flat sketch tensor (R × Σ B_l,)。"""
        layout = self.layout
        sketch = torch.zeros(layout.sketch_len, dtype=torch.float32)
        offsets = layout.sketch_layer_offsets()  # 每层 (offset, length) 或 (-1,-1)

        # 找各坐标所属层（向量化二分）
        layer_starts = torch.tensor([l.start for l in layout.layers], dtype=torch.long)
        layer_ends = torch.tensor([l.end for l in layout.layers], dtype=torch.long)
        li = _find_layer_vec(coords, layer_starts, layer_ends)

        total_bpr = layout.total_buckets_per_row

        # 按层分组处理
        for layer_idx in range(len(layout.layers)):
            mask = (li == layer_idx)
            if not mask.any():
                continue
            layer = layout.layers[layer_idx]
            if layer.is_dense:
                continue
            local_offset = offsets[layer_idx][0]
            layer_coords = coords[mask]
            layer_vals = values[mask]
            local_coords = layer_coords - layer.start

            for r in range(layout.rows):
                bs = self._bucket_seeds[layer_idx][r]
                ss = self._sign_seeds[layer_idx][r]
                buckets = self._hash_bucket_vec(local_coords, bs, layer.buckets_per_row)
                signs = self._hash_sign_vec(local_coords, ss)
                contributions = signs.float() * layer_vals.float()
                # scatter add 到 sketch
                idx = r * total_bpr + local_offset + buckets
                sketch.scatter_add_(0, idx, contributions)
        return sketch

    def estimate_batch(
        self,
        agg_sketch: torch.Tensor,
        layer_idx: int,
        local_coords: torch.Tensor,
    ) -> torch.Tensor:
        """从聚合 Sketch 批量估计多个坐标的值（median-of-rows）。

        Returns:
            (n,) 估计值
        """
        layout = self.layout
        layer = layout.layers[layer_idx]
        if layer.is_dense:
            raise ValueError("dense layer 不进 Sketch")
        offsets = layout.sketch_layer_offsets()
        local_offset = offsets[layer_idx][0]
        total_bpr = layout.total_buckets_per_row

        n = local_coords.numel()
        estimates = torch.zeros(layout.rows, n, dtype=torch.float32)

        for r in range(layout.rows):
            bs = self._bucket_seeds[layer_idx][r]
            ss = self._sign_seeds[layer_idx][r]
            buckets = self._hash_bucket_vec(local_coords, bs, layer.buckets_per_row)
            signs = self._hash_sign_vec(local_coords, ss)
            idx = r * total_bpr + local_offset + buckets
            estimates[r] = signs.float() * agg_sketch[idx].float()

        # median over rows
        return estimates.median(dim=0).values


def _find_layer_vec(
    coords: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    """向量化查找每个坐标所属层索引。"""
    # 对每个 coord，找 starts[i] <= coord < ends[i]
    # 用 searchsorted: coord >= starts 的最大 i
    idx = torch.searchsorted(starts, coords, right=True) - 1
    idx = idx.clamp(0, len(starts) - 1)
    # 验证 coord < ends[idx]
    valid = coords < ends[idx]
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    return idx


def find_layer_for_coord(layout: SketchLayout, coord: int) -> int:
    """公开接口：查找坐标所属层索引。"""
    starts = [l.start for l in layout.layers]
    ends = [l.end for l in layout.layers]
    lo, hi = 0, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if coord < starts[mid]:
            hi = mid - 1
        elif coord >= ends[mid]:
            lo = mid + 1
        else:
            return mid
    return -1
