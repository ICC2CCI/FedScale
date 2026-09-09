"""RoundPlan / block mask 调度（从 S3R12v3 提取）。"""
from __future__ import annotations

from typing import Dict

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
    ) -> None:
        self.seed = seed
        self.coverage_h = coverage_h
        self.block_size = block_size
        self.group_blocks = build_group_blocks(reference_state, block_size=block_size)
        self.total_elems = floating_elem_count(reference_state)
        self._perm_cache: Dict[int, Dict[str, list]] = {}

    def plan_for_round(self, round_idx: int) -> RoundPlan:
        if round_idx < 1:
            raise ValueError("round_idx must be >= 1")
        slot = (round_idx - 1) % self.coverage_h
        epoch = (round_idx - 1) // self.coverage_h
        if epoch not in self._perm_cache:
            self._perm_cache[epoch] = build_permutations(self.group_blocks, epoch=epoch, seed=self.seed)
        permutations = self._perm_cache[epoch]
        selected = build_selected_blocks(
            self.group_blocks, permutations, slot=slot, coverage_h=self.coverage_h
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
