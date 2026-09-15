"""Unit tests for S3R-BlockVote energy selection."""
from __future__ import annotations

import torch

from shared.block_selection import build_group_blocks
from shared.block_vote import (
    block_energies,
    flatten_group_blocks,
    select_topk_indices,
    selected_from_flat,
)


def test_topk_picks_largest_blocks() -> None:
    energies = [0.1, 5.0, 0.2, 4.0, 0.0]
    ids = select_topk_indices(energies, rho=0.4)
    assert ids == [1, 3]


def test_flatten_and_energy_roundtrip() -> None:
    state = {
        "model.layers.0.weight": torch.ones(8, dtype=torch.float32),
        "lm_head.weight": torch.arange(8, dtype=torch.float32),
    }
    groups, _ = build_group_blocks(state, block_size=4)
    flat = flatten_group_blocks(groups)
    assert len(flat) >= 2
    energies = block_energies(state, flat)
    assert len(energies) == len(flat)
    ids = select_topk_indices(energies, rho=0.5)
    selected = selected_from_flat(flat, ids)
    n_elems = sum(e - s for slices in selected.values() for s, e in slices)
    assert n_elems > 0


if __name__ == "__main__":
    test_topk_picks_largest_blocks()
    test_flatten_and_energy_roundtrip()
    print("ok")
