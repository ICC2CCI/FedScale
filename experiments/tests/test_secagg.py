"""Unit tests for Phase B: Windowed SecAgg.

Tests:
    B-1: fixed_point quantization/dequantization precision + overflow
    B-2: secagg_crypto DH consistency, mask generation, PRG determinism
    B-3: SecAggPlan / WindowDescriptor serialization
    B-4/B-5: end-to-end 2-client SecAgg simulation (mask cancellation)
    Edge cases: tampering, missing windows, q_min abort
"""
from __future__ import annotations

import torch

from shared.fixed_point import (
    DEFAULT_MODULUS_BITS,
    DEFAULT_Q,
    DEFAULT_Q_MAX,
    DEFAULT_SCALE,
    compute_q_max,
    compute_adaptive_scale,
    quantize_to_zq,
    dequantize_from_zq,
    aggregate_zq,
    unmask_aggregate,
    pack_zq,
    unpack_zq,
    overflow_check,
)
from shared.secagg_crypto import (
    generate_keypair,
    dh_shared_secret,
    generate_self_master,
    derive_pair_seed,
    derive_self_seed,
    derive_mask_vector,
    compute_pairwise_mask,
    compute_self_mask,
    apply_mask,
    compute_session_id,
    compute_cohort_hash,
    build_pair_domain_info,
    build_self_domain_info,
    prg_stream,
    hkdf_sha256,
)
from shared.protocol import (
    SecAggPlan,
    WindowDescriptor,
    build_window_descriptors,
    compute_window_layout_hash,
)
from shared.secagg_client import SecAggClient
from server.secagg_coordinator import SecAggCoordinator


# --------------------------------------------------------------------------- #
# B-1: Fixed-point quantization
# --------------------------------------------------------------------------- #

def test_quantize_dequantize_precision():
    """量化→反量化的误差在 scale 范围内。"""
    delta = torch.randn(1000, dtype=torch.float32) * 0.01
    q_zq = quantize_to_zq(delta, DEFAULT_SCALE, DEFAULT_Q_MAX, DEFAULT_Q)
    reconstructed = dequantize_from_zq(q_zq, DEFAULT_SCALE, DEFAULT_Q)
    error = (delta - reconstructed).abs().max().item()
    assert error < DEFAULT_SCALE * 1.5, f"quantization error {error} > {DEFAULT_SCALE * 1.5}"


def test_quantize_clip():
    """超出 Q_max 的值被 clip。"""
    delta = torch.tensor([100.0, -100.0], dtype=torch.float32)
    q_zq = quantize_to_zq(delta, DEFAULT_SCALE, DEFAULT_Q_MAX, DEFAULT_Q)
    # 反量化应该得到 ±Q_max * scale
    reconstructed = dequantize_from_zq(q_zq, DEFAULT_SCALE, DEFAULT_Q)
    assert abs(reconstructed[0].item()) <= DEFAULT_Q_MAX * DEFAULT_SCALE + DEFAULT_SCALE
    assert abs(reconstructed[1].item()) <= DEFAULT_Q_MAX * DEFAULT_SCALE + DEFAULT_SCALE


def test_overflow_check():
    """overflow budget 验证。"""
    assert overflow_check(2, DEFAULT_Q_MAX, DEFAULT_Q)
    assert not overflow_check(100, DEFAULT_Q_MAX, DEFAULT_Q)  # 100 * 16383 > 32768


def test_aggregate_zq():
    """Z_q 中求和正确。"""
    a = torch.tensor([1, 2, 3], dtype=torch.int64)
    b = torch.tensor([4, 5, 6], dtype=torch.int64)
    result = aggregate_zq([a, b], DEFAULT_Q)
    expected = (a + b) % DEFAULT_Q
    assert torch.equal(result, expected)


def test_pack_unpack_roundtrip():
    """pack/unpack 往返一致。"""
    q_zq = torch.randint(0, DEFAULT_Q, (1000,), dtype=torch.int64)
    packed = pack_zq(q_zq, DEFAULT_MODULUS_BITS)
    unpacked = unpack_zq(packed, len(q_zq), DEFAULT_MODULUS_BITS)
    assert torch.equal(q_zq, unpacked)
    # int16 = 2 bytes per element
    assert len(packed) == 1000 * 2


def test_stochastic_rounding():
    """随机舍入的期望值无偏。"""
    delta = torch.full((10000,), 0.5 * DEFAULT_SCALE, dtype=torch.float32)
    # 0.5 * scale → 量化值应在 0 和 1 之间各 50%
    results = []
    for _ in range(100):
        q_zq = quantize_to_zq(delta, DEFAULT_SCALE, DEFAULT_Q_MAX, DEFAULT_Q, stochastic=True)
        # 解码回整数（去掉 mod q）
        q_int = q_zq % DEFAULT_Q
        signed = torch.where(q_int > DEFAULT_Q // 2, q_int - DEFAULT_Q, q_int)
        results.append(signed.float().mean().item())
    avg = sum(results) / len(results)
    # 期望应该接近 0.5
    assert abs(avg - 0.5) < 0.1, f"stochastic rounding biased: avg={avg}"


# --------------------------------------------------------------------------- #
# B-2: SecAgg crypto
# --------------------------------------------------------------------------- #

def test_dh_shared_secret_consistency():
    """双方独立计算 DH 得到相同共享秘密。"""
    sk_a, pk_a = generate_keypair()
    sk_b, pk_b = generate_keypair()
    s_ab = dh_shared_secret(sk_a, pk_b)
    s_ba = dh_shared_secret(sk_b, pk_a)
    assert s_ab == s_ba
    assert len(s_ab) == 32


def test_prg_determinism():
    """相同 seed → 相同随机流。"""
    stream1 = prg_stream(b"test_seed", 100)
    stream2 = prg_stream(b"test_seed", 100)
    assert stream1 == stream2
    # 不同 seed → 不同流
    stream3 = prg_stream(b"different_seed", 100)
    assert stream1 != stream3


def test_mask_vector_in_range():
    """mask 向量值在 [0, q) 范围内。"""
    seed = b"test_mask_seed"
    vec = derive_mask_vector(seed, 1000, DEFAULT_Q)
    assert vec.shape == (1000,)
    assert vec.min().item() >= 0
    assert vec.max().item() < DEFAULT_Q


def test_pairwise_mask_identical():
    """双方独立计算 pairwise mask 得到相同结果。"""
    sk_a, pk_a = generate_keypair()
    sk_b, pk_b = generate_keypair()
    s = dh_shared_secret(sk_a, pk_b)

    session = "test-session"
    attempt = 0
    cohort = "test-cohort"
    wid = 0
    wlh = "test-window-hash"

    R_a = compute_pairwise_mask(s, session, attempt, cohort, wid, wlh, 0, 1, 100, DEFAULT_Q)
    R_b = compute_pairwise_mask(s, session, attempt, cohort, wid, wlh, 0, 1, 100, DEFAULT_Q)
    assert torch.equal(R_a, R_b)


def test_self_mask_different_per_client():
    """不同 client 的 self mask 不同。"""
    sm_a = generate_self_master()
    sm_b = generate_self_master()
    assert sm_a != sm_b

    B_a = compute_self_mask(sm_a, "session", 0, 0, 0, "wlh", 100, DEFAULT_Q)
    B_b = compute_self_mask(sm_b, "session", 0, 1, 0, "wlh", 100, DEFAULT_Q)
    assert not torch.equal(B_a, B_b)


def test_mask_domain_separation():
    """不同 window 的 mask 不同（域分离）。"""
    seed = b"test_seed"
    R0 = derive_mask_vector(seed, 100, DEFAULT_Q)
    R1 = derive_mask_vector(seed, 100, DEFAULT_Q)
    assert torch.equal(R0, R1)  # same seed → same mask

    # 不同 domain_info → 不同 mask
    sm = generate_self_master()
    B0 = compute_self_mask(sm, "session", 0, 0, 0, "wlh0", 100, DEFAULT_Q)
    B1 = compute_self_mask(sm, "session", 0, 0, 1, "wlh1", 100, DEFAULT_Q)
    assert not torch.equal(B0, B1)


# --------------------------------------------------------------------------- #
# B-3: Protocol objects
# --------------------------------------------------------------------------- #

def test_secagg_plan_serialization():
    """SecAggPlan 序列化往返。"""
    plan = SecAggPlan(
        secagg_session_id="test-session",
        attempt_id=3,
        cohort_hash="test-cohort",
        quantization_scale=0.001,
    )
    d = plan.to_dict()
    restored = SecAggPlan.from_dict(d)
    assert restored.secagg_session_id == "test-session"
    assert restored.attempt_id == 3
    assert restored.quantization_scale == 0.001
    assert restored.modulus_q == 65536


def test_window_descriptor():
    """WindowDescriptor 构建 + 序列化。"""
    block_list = [[0, "layer.0.weight", 0, 512], [1, "layer.1.weight", 512, 1024]]
    windows = build_window_descriptors(block_list)
    assert len(windows) == 2
    assert windows[0].window_id == 0
    assert windows[0].vector_length == 512
    assert len(windows[0].window_layout_hash) == 64

    d = windows[0].to_dict()
    restored = WindowDescriptor.from_dict(d)
    assert restored.window_id == 0
    assert restored.key_name == "layer.0.weight"


# --------------------------------------------------------------------------- #
# B-4/B-5: End-to-end 2-client SecAgg
# --------------------------------------------------------------------------- #

def _run_e2e_secagg(n_windows=3, vector_len=512, delta_scale=0.01):
    """运行完整的 2-client SecAgg 模拟，返回 per-window error。"""
    client_ids = [0, 1]
    block_list = [[i, f"layer.{i}.weight", 0, vector_len] for i in range(n_windows)]
    windows = build_window_descriptors(block_list)
    plan = SecAggPlan(
        secagg_session_id="test-session",
        modulus_q=DEFAULT_Q,
        q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE,
    )

    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=client_ids, windows=windows,
        layout_hash="test-layout", mask_hash="test-mask",
        secagg_plan=plan,
    )

    client_a = SecAggClient(0, plan, windows)
    client_b = SecAggClient(1, plan, windows)

    # DH
    coord.submit_public_key(0, client_a.public_key)
    coord.submit_public_key(1, client_b.public_key)
    peer_keys = coord.get_peer_keys()["public_keys"]
    client_a.setup_dh(peer_keys)
    client_b.setup_dh(peer_keys)

    # Mask + upload
    delta_a = {w.window_id: torch.randn(vector_len) * delta_scale for w in windows}
    delta_b = {w.window_id: torch.randn(vector_len) * delta_scale for w in windows}

    z_a = client_a.mask_all_windows(delta_a)
    z_b = client_b.mask_all_windows(delta_b)

    for wid in z_a:
        coord.submit_masked_window(0, wid, z_a[wid])
        coord.submit_masked_window(1, wid, z_b[wid])

    # Self masters
    coord.submit_self_master(0, client_a.self_master)
    coord.submit_self_master(1, client_b.self_master)

    # Aggregate
    coord.freeze_survivors()
    agg_delta = coord.aggregate_and_unmask()

    # Verify
    errors = []
    for window in windows:
        wid = window.window_id
        expected = (delta_a[wid] + delta_b[wid]) / 2.0
        actual = None
        for s, e, slice_data in agg_delta.get(window.key_name, []):
            if s == window.start and e == window.end:
                actual = slice_data
                break
        assert actual is not None, f"window {wid} not found in agg_delta"
        error = (actual - expected).abs().max().item()
        errors.append(error)
    return errors


def test_e2e_mask_cancellation():
    """2-client SecAgg: pairwise mask 在聚合中精确抵消。"""
    errors = _run_e2e_secagg(n_windows=3, vector_len=512)
    for i, err in enumerate(errors):
        assert err < 1e-3, f"window {i} error {err} too large"
    print(f"  E2E errors: {[f'{e:.6f}' for e in errors]}")


def test_e2e_different_delta_scales():
    """不同 delta 量级下 SecAgg 仍然正确。"""
    for dscale in [0.001, 0.01, 0.1]:
        errors = _run_e2e_secagg(n_windows=2, vector_len=256, delta_scale=dscale)
        max_err = max(errors)
        assert max_err < dscale * 0.1, f"delta_scale={dscale} error={max_err}"


def test_e2e_larger_vector():
    """大向量（实际 block 大小）SecAgg 正确。"""
    errors = _run_e2e_secagg(n_windows=1, vector_len=524288, delta_scale=0.01)
    assert errors[0] < 1e-3


def test_q_min_abort():
    """survivors < q_min 时 abort。"""
    client_ids = [0, 1]
    windows = build_window_descriptors([[0, "layer.0.weight", 0, 64]])
    plan = SecAggPlan(q_min=2, modulus_q=DEFAULT_Q, q_max=DEFAULT_Q_MAX)

    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=client_ids, windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=plan,
    )

    client_a = SecAggClient(0, plan, windows)

    # 只有 client 0 上传
    coord.submit_public_key(0, client_a.public_key)
    z_a = client_a.mask_all_windows({0: torch.randn(64) * 0.01})
    coord.submit_masked_window(0, 0, z_a[0])
    coord.submit_self_master(0, client_a.self_master)

    survivors = coord.freeze_survivors()
    assert len(survivors) < plan.q_min

    try:
        coord.aggregate_and_unmask()
        assert False, "should have raised RuntimeError"
    except RuntimeError as e:
        assert "q_min" in str(e) or "abort" in str(e)


def test_session_id_deterministic():
    """相同输入 → 相同 session_id。"""
    sid1 = compute_session_id("job", 1, 0, "v0", "ch", "lh", "mh")
    sid2 = compute_session_id("job", 1, 0, "v0", "ch", "lh", "mh")
    assert sid1 == sid2
    # 不同 round → 不同 session
    sid3 = compute_session_id("job", 2, 0, "v0", "ch", "lh", "mh")
    assert sid1 != sid3


def test_cohort_hash_deterministic():
    """cohort_hash 确定性 + 顺序无关。"""
    h1 = compute_cohort_hash([0, 1])
    h2 = compute_cohort_hash([1, 0])  # 顺序不同
    assert h1 == h2  # 内部排序
    h3 = compute_cohort_hash([0, 1, 2])
    assert h1 != h3


def test_hkdf_domain_separation():
    """不同 info → 不同密钥。"""
    k1 = hkdf_sha256(b"ikm", info=b"info1")
    k2 = hkdf_sha256(b"ikm", info=b"info2")
    assert k1 != k2


if __name__ == "__main__":
    test_quantize_dequantize_precision()
    test_quantize_clip()
    test_overflow_check()
    test_aggregate_zq()
    test_pack_unpack_roundtrip()
    test_stochastic_rounding()
    test_dh_shared_secret_consistency()
    test_prg_determinism()
    test_mask_vector_in_range()
    test_pairwise_mask_identical()
    test_self_mask_different_per_client()
    test_mask_domain_separation()
    test_secagg_plan_serialization()
    test_window_descriptor()
    test_e2e_mask_cancellation()
    test_e2e_different_delta_scales()
    test_e2e_larger_vector()
    test_q_min_abort()
    test_session_id_deterministic()
    test_cohort_hash_deterministic()
    test_hkdf_domain_separation()
    print("All Phase B tests passed!")
