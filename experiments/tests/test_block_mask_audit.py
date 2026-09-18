"""Unit tests for Phase A: Block Mask v1 audit enhancements.

Tests:
  A-1: canonical encoding determinism
  A-2: layout_hash stability + sensitivity
  A-3: mask_hash consistency (server vs client recompute)
  A-4: always_on small params
  A-5: client mask verification (tampering detection)
"""
from __future__ import annotations

import torch

from shared.block_selection import (
    DEFAULT_ALWAYS_ON_THRESHOLD,
    build_group_blocks,
    build_permutations,
    build_selected_blocks,
    count_selected_elems,
    recompute_selected_blocks,
)
from shared.canonical_encoding import (
    MASK_POLICY_ID,
    compute_layout_hash,
    compute_mask_hash,
    encode_block,
    encode_block_list,
    encode_selected_blocks,
)
from shared.protocol import RoundPlan
from server.block_scheduler import BlockScheduler


def _make_state():
    """Create a small model-like state_dict for testing."""
    state = {
        "model.layers.0.weight": torch.randn(1024, dtype=torch.float32),
        "model.layers.0.bias": torch.randn(64, dtype=torch.float32),  # small → always_on
        "model.layers.1.weight": torch.randn(1024, dtype=torch.float32),
        "model.layers.1.bias": torch.randn(64, dtype=torch.float32),  # small → always_on
        "model.norm.weight": torch.randn(32, dtype=torch.float32),    # small → always_on
        "lm_head.weight": torch.randn(2048, dtype=torch.float32),
    }
    return state


# --------------------------------------------------------------------------- #
# A-1: Canonical encoding determinism
# --------------------------------------------------------------------------- #

def test_canonical_encoding_deterministic():
    """Same input → same output bytes."""
    blocks = [("layer.0.weight", 0, 512), ("layer.1.weight", 512, 1024)]
    encoded1 = encode_block_list(blocks)
    encoded2 = encode_block_list(blocks)
    assert encoded1 == encoded2, "same input should produce same encoding"


def test_canonical_encoding_order_matters():
    """Different order → different encoding (for block_list)."""
    blocks_a = [("a", 0, 10), ("b", 0, 10)]
    blocks_b = [("b", 0, 10), ("a", 0, 10)]
    assert encode_block_list(blocks_a) != encode_block_list(blocks_b)


def test_canonical_encoding_selected_sorts():
    """encode_selected_blocks sorts by (key_name, start, end) regardless of dict order."""
    selected_a = {"b": [(0, 10)], "a": [(20, 30)]}
    selected_b = {"a": [(20, 30)], "b": [(0, 10)]}
    assert encode_selected_blocks(selected_a) == encode_selected_blocks(selected_b)


def test_mask_hash_stable():
    """mask_hash is deterministic for same selected blocks."""
    selected = {"layer.0.weight": [(0, 512)], "lm_head.weight": [(0, 256)]}
    h1 = compute_mask_hash(selected)
    h2 = compute_mask_hash(selected)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex


# --------------------------------------------------------------------------- #
# A-2: Layout hash
# --------------------------------------------------------------------------- #

def test_layout_hash_stable():
    """Same state + params → same layout_hash."""
    state = _make_state()
    gb, ao = build_group_blocks(state, block_size=512, always_on_threshold=64)
    h1 = compute_layout_hash(gb, ao, coverage_h=5, block_size=512)
    h2 = compute_layout_hash(gb, ao, coverage_h=5, block_size=512)
    assert h1 == h2


def test_layout_hash_sensitive_to_block_size():
    """Different block_size → different layout_hash."""
    state = _make_state()
    gb1, ao1 = build_group_blocks(state, block_size=512)
    gb2, ao2 = build_group_blocks(state, block_size=1024)
    h1 = compute_layout_hash(gb1, ao1, coverage_h=5, block_size=512)
    h2 = compute_layout_hash(gb2, ao2, coverage_h=5, block_size=1024)
    assert h1 != h2


def test_layout_hash_sensitive_to_coverage_h():
    """Different coverage_h → different layout_hash."""
    state = _make_state()
    gb, ao = build_group_blocks(state, block_size=512)
    h1 = compute_layout_hash(gb, ao, coverage_h=5, block_size=512)
    h2 = compute_layout_hash(gb, ao, coverage_h=10, block_size=512)
    assert h1 != h2


# --------------------------------------------------------------------------- #
# A-3: mask_hash consistency
# --------------------------------------------------------------------------- #

def test_mask_hash_server_vs_client():
    """Server plan_for_round and client recompute_selected_blocks produce same mask_hash."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan = scheduler.plan_for_round(1)
    # Client independently recomputes
    recomputed = recompute_selected_blocks(
        reference_state=state,
        round_idx=1,
        coverage_h=4,
        seed=42,
        block_size=512,
        always_on_keys=set(plan.always_on_keys),
    )
    client_hash = compute_mask_hash(recomputed)
    assert client_hash == plan.mask_hash, (
        f"server mask_hash={plan.mask_hash[:16]} != client={client_hash[:16]}"
    )


def test_mask_hash_changes_per_slot():
    """Different slots → different mask_hash."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan1 = scheduler.plan_for_round(1)  # slot 0
    plan2 = scheduler.plan_for_round(2)  # slot 1
    assert plan1.mask_hash != plan2.mask_hash


# --------------------------------------------------------------------------- #
# A-4: always_on small params
# --------------------------------------------------------------------------- #

def test_always_on_identifies_small_params():
    """Small params (numel <= threshold) are marked always_on."""
    state = _make_state()
    gb, ao = build_group_blocks(state, block_size=512, always_on_threshold=64)
    assert "model.layers.0.bias" in ao
    assert "model.layers.1.bias" in ao
    assert "model.norm.weight" in ao
    assert "model.layers.0.weight" not in ao
    assert "lm_head.weight" not in ao


def test_always_on_selected_every_round():
    """always_on blocks appear in every round's selection."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    for round_idx in range(1, 5):
        plan = scheduler.plan_for_round(round_idx)
        selected = {k: set((s, e) for s, e in v) for k, v in plan.selected_by_key.items()}
        for ao_key in plan.always_on_keys:
            assert ao_key in selected, f"always_on key {ao_key} missing in round {round_idx}"


def test_always_on_not_in_permutation():
    """always_on blocks don't participate in permutation (only rotating blocks do)."""
    state = _make_state()
    gb, ao = build_group_blocks(state, block_size=512, always_on_threshold=64)
    perms = build_permutations(gb, epoch=0, seed=42, always_on_keys=ao)
    # Check that no always_on block index appears in permutation
    for gid, blocks in gb.items():
        perm = perms.get(gid, [])
        for block_idx in perm:
            kn = blocks[block_idx][0]
            assert kn not in ao, f"always_on key {kn} found in permutation"


def test_always_on_coverage_property():
    """Rotating blocks: each appears exactly once in H rounds."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    seen = {}  # (key_name, start, end) → count
    for round_idx in range(1, 5):  # H=4
        plan = scheduler.plan_for_round(round_idx)
        for kn, slices in plan.selected_by_key.items():
            if kn in plan.always_on_keys:
                continue  # skip always_on
            for s, e in slices:
                key = (kn, s, e)
                seen[key] = seen.get(key, 0) + 1
    # Every rotating block should appear exactly once
    for key, count in seen.items():
        assert count == 1, f"rotating block {key} appeared {count} times (expected 1)"


def test_always_on_disabled_when_threshold_zero():
    """always_on_threshold=0 → no always_on keys."""
    state = _make_state()
    gb, ao = build_group_blocks(state, block_size=512, always_on_threshold=0)
    assert len(ao) == 0


# --------------------------------------------------------------------------- #
# A-5: Client mask verification (tampering detection)
# --------------------------------------------------------------------------- #

def test_tampered_mask_hash_detected():
    """If server sends a different mask_hash, client verification fails."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan = scheduler.plan_for_round(1)
    # Tamper: change mask_hash
    tampered_hash = "0" * 64
    assert tampered_hash != plan.mask_hash
    # Client recomputes
    recomputed = recompute_selected_blocks(
        reference_state=state,
        round_idx=1,
        coverage_h=4,
        seed=42,
        block_size=512,
        always_on_keys=set(plan.always_on_keys),
    )
    client_hash = compute_mask_hash(recomputed)
    assert client_hash != tampered_hash, "tampered hash should not match client recomputation"


def test_tampered_selected_blocks_detected():
    """If server sends different selected blocks, mask_hash won't match."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan = scheduler.plan_for_round(1)
    # Client recomputes correctly
    recomputed = recompute_selected_blocks(
        reference_state=state,
        round_idx=1,
        coverage_h=4,
        seed=42,
        block_size=512,
        always_on_keys=set(plan.always_on_keys),
    )
    client_hash = compute_mask_hash(recomputed)
    # Server sends a different selection (e.g. from a different seed)
    wrong_selected = recompute_selected_blocks(
        reference_state=state,
        round_idx=1,
        coverage_h=4,
        seed=999,  # different seed
        block_size=512,
        always_on_keys=set(plan.always_on_keys),
    )
    wrong_hash = compute_mask_hash(wrong_selected)
    assert client_hash != wrong_hash, "different seed should produce different mask_hash"


def test_round_plan_serialization_preserves_audit_fields():
    """RoundPlan to_dict/from_dict preserves mask_hash, layout_hash, always_on_keys."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan = scheduler.plan_for_round(1)
    d = plan.to_dict()
    restored = RoundPlan.from_dict(d)
    assert restored.mask_hash == plan.mask_hash
    assert restored.layout_hash == plan.layout_hash
    assert restored.mask_policy_id == plan.mask_policy_id
    assert set(restored.always_on_keys) == set(plan.always_on_keys)


# --------------------------------------------------------------------------- #
# Integration: scheduler with always_on produces correct plans
# --------------------------------------------------------------------------- #

def test_scheduler_plan_has_audit_fields():
    """BlockScheduler.plan_for_round includes all audit fields."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan = scheduler.plan_for_round(1)
    assert plan.mask_hash, "mask_hash should be non-empty"
    assert plan.layout_hash, "layout_hash should be non-empty"
    assert plan.mask_policy_id == MASK_POLICY_ID
    assert len(plan.always_on_keys) > 0, "should have always_on keys"


def test_scheduler_layout_hash_consistent_across_rounds():
    """layout_hash is the same for all rounds (layout doesn't change)."""
    state = _make_state()
    scheduler = BlockScheduler(state, seed=42, coverage_h=4, block_size=512, always_on_threshold=64)
    plan1 = scheduler.plan_for_round(1)
    plan2 = scheduler.plan_for_round(3)
    assert plan1.layout_hash == plan2.layout_hash


if __name__ == "__main__":
    test_canonical_encoding_deterministic()
    test_canonical_encoding_order_matters()
    test_canonical_encoding_selected_sorts()
    test_mask_hash_stable()
    test_layout_hash_stable()
    test_layout_hash_sensitive_to_block_size()
    test_layout_hash_sensitive_to_coverage_h()
    test_mask_hash_server_vs_client()
    test_mask_hash_changes_per_slot()
    test_always_on_identifies_small_params()
    test_always_on_selected_every_round()
    test_always_on_not_in_permutation()
    test_always_on_coverage_property()
    test_always_on_disabled_when_threshold_zero()
    test_tampered_mask_hash_detected()
    test_tampered_selected_blocks_detected()
    test_round_plan_serialization_preserves_audit_fields()
    test_scheduler_plan_has_audit_fields()
    test_scheduler_layout_hash_consistent_across_rounds()
    print("All Phase A tests passed!")
