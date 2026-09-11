"""RoundPlan / block mask 调度（从 S3R12v3 提取；ALG-2 支持多 slot）。"""
from __future__ import annotations

from typing import Dict, Set

import torch

from shared.block_selection import (
    build_group_blocks,
    build_permutations,
    build_selected_blocks,
    count_selected_elems,
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
    ) -> None:
        self.seed = seed
        self.coverage_h = max(1, coverage_h)
        self.slots_per_round = max(1, int(slots_per_round))
        self.block_size = block_size
        self.group_blocks = build_group_blocks(reference_state, block_size=block_size)
        self.total_elems = floating_elem_count(reference_state)
        self._perm_cache: Dict[int, Dict[str, list]] = {}

    def active_slots_for_round(self, round_idx: int) -> Set[int]:
        """ALG-2：返回本轮激活的 slot 集合（覆盖周期 = ceil(H / slots_per_round) 轮）。"""
        slot = (round_idx - 1) % self.coverage_h
        active: Set[int] = set()
        for i in range(self.slots_per_round):
            active.add((slot + i) % self.coverage_h)
        return active

    def plan_for_round(self, round_idx: int) -> RoundPlan:
        if round_idx < 1:
            raise ValueError("round_idx must be >= 1")
        slot = (round_idx - 1) % self.coverage_h
        epoch = (round_idx - 1) // self.coverage_h
        if epoch not in self._perm_cache:
            self._perm_cache[epoch] = build_permutations(self.group_blocks, epoch=epoch, seed=self.seed)
        permutations = self._perm_cache[epoch]
        # ALG-2：单 slot 用原路径；多 slot 选 active_slots 中所有 pos % H
        if self.slots_per_round <= 1:
            selected = build_selected_blocks(
                self.group_blocks, permutations, slot=slot, coverage_h=self.coverage_h
            )
        else:
            active_slots = self.active_slots_for_round(round_idx)
            selected = build_selected_blocks_multi(
                self.group_blocks, permutations, active_slots=active_slots, coverage_h=self.coverage_h
            )
        selected_elems = count_selected_elems(selected)
        n_blocks = sum(len(v) for v in selected.values())
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
