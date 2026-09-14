"""RoundPlan / block mask 调度（从 S3R12v3 提取；ALG-2 支持多 slot）。"""
from __future__ import annotations

import logging
from typing import Dict, List, Set, Tuple

import torch

logger = logging.getLogger(__name__)

from shared.block_selection import (
    build_group_blocks,
    build_permutations,
    build_selected_blocks,
    count_selected_elems,
)
from shared.block_vote import (
    COMPRESSORS,
    flatten_group_blocks,
    select_topk_indices,
    selected_from_flat,
)
from shared.protocol import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_COVERAGE_H,
    DEFAULT_SEED,
    RoundPlan,
    SelectedByKey,
    selected_to_jsonable,
)
from shared.state_dict_utils import floating_elem_count


class BlockScheduler:
    def __init__(
        self,
        reference_state: Dict[str, torch.Tensor],
        seed: int = DEFAULT_SEED,
        coverage_h: int = DEFAULT_COVERAGE_H,
        block_size: int = DEFAULT_BLOCK_SIZE,
        slots_per_round: int = 1,
        compressor: str = "public_random",
        rho: float = 0.0,
    ) -> None:
        self.seed = seed
        self.coverage_h = max(1, coverage_h)
        self.slots_per_round = max(1, int(slots_per_round))
        self.block_size = block_size
        compressor = (compressor or "public_random").strip()
        if compressor not in COMPRESSORS:
            raise ValueError(f"unknown compressor {compressor}")
        self.compressor = compressor
        self.rho = float(rho) if rho and rho > 0 else 1.0 / self.coverage_h
        self.group_blocks = build_group_blocks(reference_state, block_size=block_size)
        self.flat_blocks = flatten_group_blocks(self.group_blocks)
        self.total_elems = floating_elem_count(reference_state)
        self._perm_cache: Dict[int, Dict[str, list]] = {}
        self._energy = None
        # SEC-1：gidx -> (key_name, start, end) 全局映射表（所有 block，不仅是选中的）
        # 从 flat_blocks 构建顺序 gidx 映射（flat_blocks 已按 group 顺序排列）
        self.gidx_layout: List[Tuple[str, int, int]] = list(self.flat_blocks)
        self.gidx_map: Dict[int, Tuple[str, int, int]] = {
            i: (kn, s, e) for i, (kn, s, e) in enumerate(self.gidx_layout)
        }

    def active_slots_for_round(self, round_idx: int) -> Set[int]:
        """ALG-2：返回本轮激活的 slot 集合（覆盖周期 = ceil(H / slots_per_round) 轮）。"""
        slot = (round_idx - 1) % self.coverage_h
        active: Set[int] = set()
        for i in range(self.slots_per_round):
            active.add((slot + i) % self.coverage_h)
        return active

    def ingest_energy(self, energy: torch.Tensor) -> None:
        if energy.numel() != len(self.flat_blocks):
            logger.warning(
                "drop energy vector len=%s expected=%s",
                int(energy.numel()),
                len(self.flat_blocks),
            )
            return
        vec = energy.detach().to(dtype=torch.float32).reshape(-1)
        if self._energy is None:
            self._energy = vec.clone()
        else:
            self._energy = 0.5 * self._energy + 0.5 * vec

    def _public_selected(self, round_idx: int):
        slot = (round_idx - 1) % self.coverage_h
        epoch = (round_idx - 1) // self.coverage_h
        if epoch not in self._perm_cache:
            self._perm_cache[epoch] = build_permutations(self.group_blocks, epoch=epoch, seed=self.seed)
        permutations = self._perm_cache[epoch]
        if self.slots_per_round <= 1:
            selected = build_selected_blocks(
                self.group_blocks, permutations, slot=slot, coverage_h=self.coverage_h
            )
        else:
            active_slots = self.active_slots_for_round(round_idx)
            selected = build_selected_blocks_multi(
                self.group_blocks, permutations, active_slots=active_slots, coverage_h=self.coverage_h
            )
        return selected, epoch, slot

    def plan_for_round(self, round_idx: int) -> RoundPlan:
        if round_idx < 1:
            raise ValueError("round_idx must be >= 1")
        if self.compressor == "dense":
            selected = selected_from_flat(self.flat_blocks, range(len(self.flat_blocks)))
            epoch = (round_idx - 1) // self.coverage_h
            slot = (round_idx - 1) % self.coverage_h
        elif self.compressor == "block_vote_lag" and self._energy is not None:
            ids = select_topk_indices(self._energy.tolist(), self.rho)
            selected = selected_from_flat(self.flat_blocks, ids)
            epoch = (round_idx - 1) // self.coverage_h
            slot = (round_idx - 1) % self.coverage_h
        else:
            selected, epoch, slot = self._public_selected(round_idx)
        selected_elems = count_selected_elems(selected)
        n_blocks = sum(len(v) for v in selected.values())
        block_list: List[List[int]] = []
        gidx = 0
        for key_name, slices in selected.items():
            for s, e in slices:
                block_list.append([gidx, key_name, s, e])
                gidx += 1
        return RoundPlan(
            round=round_idx,
            epoch=epoch,
            slot=slot,
            coverage_h=self.coverage_h,
            seed=self.seed,
            selected_by_key=selected_to_jsonable(selected),
            n_selected_blocks=n_blocks,
            selected_elems=selected_elems,
            total_elems=self.total_elems,
            block_list=block_list,
        )


def build_selected_blocks_multi(
    group_blocks,
    permutations,
    *,
    active_slots: Set[int],
    coverage_h: int = DEFAULT_COVERAGE_H,
) -> SelectedByKey:
    """ALG-2：选中 pos % H in active_slots 的所有 block。"""
    selected_by_key: SelectedByKey = {}
    for gid, blocks in group_blocks.items():
        perm = permutations[gid]
        for pos, block_idx in enumerate(perm):
            if pos % coverage_h in active_slots:
                key_name, start, end = blocks[block_idx]
                selected_by_key.setdefault(key_name, []).append((start, end))
    return selected_by_key
