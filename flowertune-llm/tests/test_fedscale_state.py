import torch

from flowertune_llm.fedscale_state import (
    apply_fedscale_int8_delta,
    build_canonical_layout,
    encode_fedscale_int8_delta,
    encoded_block_ids,
    encoded_nbytes,
    mask_hash,
    public_block_mask,
)


def _state():
    return {
        "z.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "a.weight": torch.linspace(-1, 1, 9, dtype=torch.float32).reshape(3, 3),
        "counter": torch.tensor(1, dtype=torch.int64),
    }


def test_canonical_layout_is_name_sorted_and_public_mask_rotates():
    layout = build_canonical_layout(_state(), block_size=5)
    assert [tensor.name for tensor in layout.tensors] == ["a.weight", "z.weight"]
    assert public_block_mask(layout, 1, 0.5) == (0, 1, 2)
    assert public_block_mask(layout, 2, 0.5) == (0, 3, 4)
    assert len(mask_hash(layout, (0, 1, 2))) == 64


def test_block_delta_round_trip_only_touches_public_blocks():
    base = _state()
    final = {name: value.clone() for name, value in base.items()}
    final["a.weight"].add_(0.03)
    final["z.weight"].sub_(0.02)
    layout = build_canonical_layout(base, block_size=5)
    selected = (0, 2, 4)
    encoded = encode_fedscale_int8_delta(final, base, layout, selected)
    decoded = apply_fedscale_int8_delta(base, encoded, layout)
    assert encoded_block_ids(encoded, layout) == selected
    assert encoded_nbytes(encoded) > 0
    for block in layout.blocks:
        for tensor in layout.tensors:
            left, right = max(block.start, tensor.start), min(block.end, tensor.end)
            if left >= right:
                continue
            local_start, local_end = left - tensor.start, right - tensor.start
            actual = decoded[tensor.name].reshape(-1)[local_start:local_end]
            expected = final[tensor.name].reshape(-1)[local_start:local_end]
            if block.block_id in selected:
                assert torch.max(torch.abs(actual - expected)) < 5e-4
            else:
                assert torch.equal(actual, base[tensor.name].reshape(-1)[local_start:local_end])
    assert torch.equal(decoded["counter"], base["counter"])


def test_mismatched_mask_is_rejected():
    base = _state()
    layout = build_canonical_layout(base, block_size=5)
    encoded = encode_fedscale_int8_delta(base, base, layout, (0,))
    encoded["__fedscale_mask_hash__"][0] ^= 1
    try:
        apply_fedscale_int8_delta(base, encoded, layout)
    except ValueError as error:
        assert "mask hash mismatch" in str(error)
    else:
        raise AssertionError("mismatched public mask was accepted")


def test_equal_weighted_icc_updates_match_blockwise_fedavg():
    base = _state()
    left = {name: value.clone() for name, value in base.items()}
    right = {name: value.clone() for name, value in base.items()}
    left["a.weight"].add_(0.02)
    right["a.weight"].sub_(0.01)
    left["z.weight"].add_(0.03)
    right["z.weight"].add_(0.01)
    layout = build_canonical_layout(base, block_size=5)
    selected = tuple(range(len(layout.blocks)))
    left_encoded = encode_fedscale_int8_delta(left, base, layout, selected)
    right_encoded = encode_fedscale_int8_delta(right, base, layout, selected)
    aggregate = {name: value.clone() for name, value in base.items()}
    apply_fedscale_int8_delta(aggregate, left_encoded, layout, weight=0.5, clone=False)
    apply_fedscale_int8_delta(aggregate, right_encoded, layout, weight=0.5, clone=False)
    for name in ("a.weight", "z.weight"):
        expected = (left[name] + right[name]) / 2
        assert torch.max(torch.abs(aggregate[name] - expected)) < 5e-4


def test_sparse_checkpoint_reconstructs_global_state_and_block_union():
    base = _state()
    global_state = {name: value.clone() for name, value in base.items()}
    global_state["a.weight"].add_(0.04)
    global_state["z.weight"].sub_(0.02)
    layout = build_canonical_layout(base, block_size=5)
    updated = (0, 2, 4)
    checkpoint = encode_fedscale_int8_delta(global_state, base, layout, updated)

    reconstructed = apply_fedscale_int8_delta(base, checkpoint, layout)
    assert encoded_block_ids(checkpoint, layout) == updated
    for block in layout.blocks:
        for tensor in layout.tensors:
            left, right = max(block.start, tensor.start), min(block.end, tensor.end)
            if left >= right:
                continue
            local_start, local_end = left - tensor.start, right - tensor.start
            actual = reconstructed[tensor.name].reshape(-1)[local_start:local_end]
            if block.block_id in updated:
                expected = global_state[tensor.name].reshape(-1)[local_start:local_end]
                assert torch.max(torch.abs(actual - expected)) < 5e-4
            else:
                expected = base[tensor.name].reshape(-1)[local_start:local_end]
                assert torch.equal(actual, expected)
