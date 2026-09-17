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
    SCALE_COVERAGE,
    compute_q_max,
    compute_adaptive_scale,
    compute_window_scale,
    fwht,
    hadamard_transform,
    inverse_hadamard_transform,
    generate_rademacher_signs,
    quantize_to_zq,
    quantize_to_zq_with_feedback,
    quantization_error_metrics,
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
from shared.secagg_client import (
    SecAggClient,
    extract_window_to_send_fp32,
    global_amax_from_slices,
    window_amax_payload,
)
from server.secagg_coordinator import SecAggCoordinator
from shared.block_selection import (
    merge_quant_residual_memory,
    update_block_memory_from_states,
    update_block_memory_with_quant_residual,
)
from shared.state_dict_utils import add_state, sub_state


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


def test_window_scale_does_not_clip_amax():
    """coverage=1.05 时，本轮 amax 量化后 clip_frac=0。"""
    amax = 3e-4
    scale = compute_window_scale(amax, DEFAULT_Q_MAX, coverage=SCALE_COVERAGE)
    delta = torch.tensor([amax, -amax, amax * 0.5], dtype=torch.float32)
    _q, residual, stats = quantize_to_zq_with_feedback(
        delta, scale, DEFAULT_Q_MAX, DEFAULT_Q, stochastic=False,
    )
    assert stats["clip_frac"] == 0.0
    assert residual.abs().max().item() < scale * 1.5


def test_old_coverage_clips_amax():
    """coverage=0.9 会 clip 本轮最大值（回归：不要再用这个系数）。"""
    amax = 3e-4
    scale = compute_window_scale(amax, DEFAULT_Q_MAX, coverage=0.9)
    delta = torch.tensor([amax, -amax], dtype=torch.float32)
    _q, _res, stats = quantize_to_zq_with_feedback(
        delta, scale, DEFAULT_Q_MAX, DEFAULT_Q, stochastic=False,
    )
    assert stats["clip_frac"] == 1.0


def test_quant_error_feedback_identity():
    """residual + reconstructed == 原始 delta。"""
    delta = torch.tensor([0.001, 1e-8, -0.002, 0.0], dtype=torch.float32)
    scale = compute_window_scale(float(delta.abs().max()), DEFAULT_Q_MAX)
    q_zq, residual, _stats = quantize_to_zq_with_feedback(
        delta, scale, DEFAULT_Q_MAX, DEFAULT_Q,
    )
    reconstructed = dequantize_from_zq(q_zq, scale, DEFAULT_Q)
    assert torch.allclose(delta, reconstructed + residual, atol=1e-7)


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
        window_scales={"0": 1e-8, "1": 2e-5},
    )
    d = plan.to_dict()
    restored = SecAggPlan.from_dict(d)
    assert restored.secagg_session_id == "test-session"
    assert restored.attempt_id == 3
    assert restored.quantization_scale == 0.001
    assert restored.get_window_scale(0) == 1e-8
    assert restored.get_window_scale(1) == 2e-5
    assert restored.get_window_scale(99) == 0.001
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


def test_public_scales_use_current_round_amax():
    """收齐 client amax 后，scale = max_k(amax) / Q_max * SCALE_COVERAGE，且所有 client 一致。"""
    client_ids = [0, 1]
    windows = build_window_descriptors([[0, "tiny.weight", 0, 8], [1, "big.weight", 0, 8]])
    plan = SecAggPlan(
        modulus_q=DEFAULT_Q,
        q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE,
        window_scales={},  # per-window 模式：由 amax 填充
    )
    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=client_ids, windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=plan,
    )
    # 创建时未就绪，只有 fallback
    assert coord.scales_ready is False
    client_a = SecAggClient(0, plan, windows)
    client_b = SecAggClient(1, plan, windows)
    # 两个 client 上报不同 amax，server 取 max_k
    coord.submit_public_key(0, client_a.public_key, {"0": 1e-6, "1": 1e-2})
    result = coord.submit_public_key(1, client_b.public_key, {"0": 2e-6, "1": 5e-3})
    assert result["status"] == "ready"
    assert result["scales_ready"] is True
    expected_tiny = compute_window_scale(2e-6, DEFAULT_Q_MAX, SCALE_COVERAGE)
    expected_big = compute_window_scale(1e-2, DEFAULT_Q_MAX, SCALE_COVERAGE)
    assert abs(float(result["window_scales"]["0"]) - expected_tiny) / expected_tiny < 1e-9
    assert abs(float(result["window_scales"]["1"]) - expected_big) / expected_big < 1e-9


def test_update_public_block_scales_from_agg():
    """coordinator 收齐 amax 后，per-window scale 量级差与 amax 一致（>1e4）。"""
    windows = build_window_descriptors([[0, "tiny.weight", 0, 8], [1, "big.weight", 0, 8]])
    client_ids = [0, 1]
    plan = SecAggPlan(
        modulus_q=DEFAULT_Q, q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE, window_scales={},
    )
    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=client_ids, windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=plan,
    )
    client_a = SecAggClient(0, plan, windows)
    client_b = SecAggClient(1, plan, windows)
    coord.submit_public_key(0, client_a.public_key, {"0": 1e-6, "1": 1e-1})
    coord.submit_public_key(1, client_b.public_key, {"0": 1e-6, "1": 1e-1})
    scale_tiny = coord.window_scales[0]
    scale_big = coord.window_scales[1]
    expected_tiny = compute_window_scale(1e-6, DEFAULT_Q_MAX, SCALE_COVERAGE)
    expected_big = compute_window_scale(1e-1, DEFAULT_Q_MAX, SCALE_COVERAGE)
    assert abs(scale_tiny - expected_tiny) / expected_tiny < 1e-9
    assert abs(scale_big - expected_big) / expected_big < 1e-9
    assert scale_big / scale_tiny > 1e4


def test_e2e_per_window_mixed_magnitude():
    """两个 window 量级差 1e5 时，公开 per-window scale 仍能还原小更新。"""
    client_ids = [0, 1]
    windows = build_window_descriptors([[0, "tiny.weight", 0, 64], [1, "big.weight", 0, 64]])
    tiny = 1e-6
    big = 5e-2
    plan = SecAggPlan(
        modulus_q=DEFAULT_Q,
        q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE,
        window_scales={},  # per-window 模式：由 client amax 填充
    )
    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=client_ids, windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=plan,
    )
    client_a = SecAggClient(0, plan, windows)
    client_b = SecAggClient(1, plan, windows)

    delta_a = {
        0: torch.full((64,), tiny, dtype=torch.float32),
        1: torch.full((64,), big, dtype=torch.float32),
    }
    delta_b = {
        0: torch.full((64,), tiny * 0.5, dtype=torch.float32),
        1: torch.full((64,), big * 0.5, dtype=torch.float32),
    }
    # 上报当轮 amax（client_a 用自己 delta 的 max）
    coord.submit_public_key(0, client_a.public_key, {"0": tiny, "1": big})
    result = coord.submit_public_key(1, client_b.public_key, {"0": tiny * 0.5, "1": big * 0.5})
    # 把 server 下发的 scale 同步到 client plan（真实 pipeline 通过 peer-keys 拿到）
    for k, v in result["window_scales"].items():
        plan.window_scales[str(k)] = float(v)
    peer_keys = coord.get_peer_keys()["public_keys"]
    client_a.setup_dh(peer_keys)
    client_b.setup_dh(peer_keys)

    z_a = client_a.mask_all_windows(delta_a)
    z_b = client_b.mask_all_windows(delta_b)
    for wid in z_a:
        coord.submit_masked_window(0, wid, z_a[wid])
        coord.submit_masked_window(1, wid, z_b[wid])
    coord.submit_self_master(0, client_a.self_master)
    coord.submit_self_master(1, client_b.self_master)
    coord.freeze_survivors()
    agg = coord.aggregate_and_unmask()

    expected_tiny = (tiny + tiny * 0.5) / 2.0
    expected_big = (big + big * 0.5) / 2.0
    actual_tiny = agg["tiny.weight"][0][2]
    actual_big = agg["big.weight"][0][2]
    tiny_err = (actual_tiny - expected_tiny).abs().max().item()
    big_err = (actual_big - expected_big).abs().max().item()
    assert tiny_err < tiny * 0.05, f"tiny window lost: err={tiny_err}"
    assert big_err < big * 0.05, f"big window err={big_err}"


def test_update_block_memory_keeps_quant_residual():
    """选中 block 保留量化 residual（不乘 memory_decay）；未选中 block 保留原 delta * decay。"""
    to_send = {
        "w": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    }
    selected = {"w": [(0, 2)]}
    residual = {"w": [(0, 2, torch.tensor([0.01, -0.02], dtype=torch.float32))]}
    mem = update_block_memory_with_quant_residual(to_send, selected, residual, decay=0.9)
    got = mem["w"]
    assert abs(got[0].item() - 0.01) < 1e-6
    assert abs(got[1].item() + 0.02) < 1e-6
    assert abs(got[2].item() - 3.0 * 0.9) < 1e-6
    assert abs(got[3].item() - 4.0 * 0.9) < 1e-6
    assert got.dtype == torch.float32


# --------------------------------------------------------------------------- #
# B-4/B-5: End-to-end 2-client SecAgg
# --------------------------------------------------------------------------- #

def _fresh_client_plan(*, hadamard: bool = False) -> SecAggPlan:
    """生产路径：client 本地 plan 没有 session，必须从 peer-keys 写入。"""
    return SecAggPlan(
        modulus_q=DEFAULT_Q,
        q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE,
        hadamard_enabled=hadamard,
        hadamard_seed=1,
    )


def _apply_peer_to_clients(clients, payload: dict) -> None:
    for client in clients:
        client.plan.apply_session_from_server(payload)
        window_scales = payload.get("window_scales") or {}
        if window_scales:
            client.plan.window_scales = {str(k): float(v) for k, v in window_scales.items()}
        if payload.get("global_scale") is not None:
            client.plan.quantization_scale = float(payload["global_scale"])


def _run_e2e_secagg(n_windows=3, vector_len=512, delta_scale=0.01, seed_public_scales=False, hadamard=False):
    """运行完整的 2-client SecAgg 模拟，返回 per-window error。

    client 与 coordinator 使用独立 plan；session 只通过 peer-keys 同步。
    """
    client_ids = [0, 1]
    block_list = [[i, f"layer.{i}.weight", 0, vector_len] for i in range(n_windows)]
    windows = build_window_descriptors(block_list)
    server_plan = SecAggPlan(
        modulus_q=DEFAULT_Q,
        q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE,
        hadamard_enabled=hadamard,
        hadamard_seed=1,
    )

    delta_a = {w.window_id: torch.randn(vector_len) * delta_scale for w in windows}
    delta_b = {w.window_id: torch.randn(vector_len) * delta_scale for w in windows}

    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=client_ids, windows=windows,
        layout_hash="test-layout", mask_hash="test-mask",
        secagg_plan=server_plan,
    )

    client_a = SecAggClient(0, _fresh_client_plan(hadamard=hadamard), windows)
    client_b = SecAggClient(1, _fresh_client_plan(hadamard=hadamard), windows)

    if seed_public_scales:
        coord.submit_public_key(0, client_a.public_key)
        coord.submit_public_key(1, client_b.public_key)
    elif hadamard:
        coord.submit_public_key(
            0, client_a.public_key,
            global_amax=global_amax_from_slices(delta_a, windows, client_a.plan),
        )
        result = coord.submit_public_key(
            1, client_b.public_key,
            global_amax=global_amax_from_slices(delta_b, windows, client_b.plan),
        )
        assert result.get("global_scale") is not None
        assert result.get("secagg_session_id")
        assert len(set(result["window_scales"].values())) == 1
        assert coord.window_amax == {}
    else:
        amax_a = window_amax_payload(delta_a, windows=windows, plan=client_a.plan)
        amax_b = window_amax_payload(delta_b, windows=windows, plan=client_b.plan)
        coord.submit_public_key(0, client_a.public_key, amax_a)
        coord.submit_public_key(1, client_b.public_key, amax_b)

    payload = coord.get_peer_keys()
    assert payload.get("status") == "ready"
    _apply_peer_to_clients([client_a, client_b], payload)
    peer_keys = payload["public_keys"]
    client_a.setup_dh(peer_keys)
    client_b.setup_dh(peer_keys)

    z_a = client_a.mask_all_windows(delta_a)
    z_b = client_b.mask_all_windows(delta_b)

    for wid in z_a:
        coord.submit_masked_window(0, wid, z_a[wid])
        coord.submit_masked_window(1, wid, z_b[wid])

    coord.submit_self_master(0, client_a.self_master)
    coord.submit_self_master(1, client_b.self_master)

    coord.freeze_survivors()
    agg_delta = coord.aggregate_and_unmask()

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
        assert int(actual.numel()) == vector_len
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
    """不同 delta 量级下，当轮 per-window scale 使相对误差保持很小。"""
    for dscale in [0.001, 0.01, 0.1]:
        errors = _run_e2e_secagg(n_windows=2, vector_len=256, delta_scale=dscale, seed_public_scales=False)
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


def test_sub_state_keep_fp32_avoids_fp16_roundtrip():
    """keep_fp32=True 时 delta 不再被 round 回 FP16。"""
    a = {"w": torch.tensor([1.0 + 1e-5], dtype=torch.float16)}
    b = {"w": torch.tensor([1.0], dtype=torch.float16)}
    rounded = sub_state(a, b, keep_fp32=False)
    kept = sub_state(a, b, keep_fp32=True)
    assert rounded["w"].dtype == torch.float16
    assert kept["w"].dtype == torch.float32
    # FP16 减法后再还原会丢掉一部分尾数；FP32 差值应更接近真值
    true_diff = a["w"].to(torch.float32) - b["w"].to(torch.float32)
    assert torch.allclose(kept["w"], true_diff, atol=0.0, rtol=0.0)
    added = add_state(kept, {"w": torch.tensor([3e-6], dtype=torch.float32)}, keep_fp32=True)
    assert added["w"].dtype == torch.float32


def test_extract_window_to_send_fp32_uses_split_residual():
    local = {"w": torch.tensor([1.5, 2.5, 3.5, 4.5], dtype=torch.float16)}
    glob = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)}
    block_mem = {"w": torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.float32)}
    quant_res = {"w": [(0, 2, torch.tensor([0.001, -0.002], dtype=torch.float32))]}
    windows = build_window_descriptors([[0, "w", 0, 2], [1, "w", 2, 4]])
    slices = extract_window_to_send_fp32(local, glob, block_mem, quant_res, windows)
    # window 0: (1.5-1.0)+0.01+0.001 , (2.5-2.0)+0.02-0.002
    assert abs(slices[0][0].item() - (0.5 + 0.01 + 0.001)) < 1e-5
    assert abs(slices[0][1].item() - (0.5 + 0.02 - 0.002)) < 1e-5
    # window 1 无 quant residual
    assert abs(slices[1][0].item() - (0.5 + 0.03)) < 1e-5
    assert slices[0].dtype == torch.float32


def test_split_error_feedback_independent_decay():
    local = {"w": torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32)}
    glob = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)}
    block_mem = {"w": torch.zeros(4, dtype=torch.float32)}
    old_res = {"w": [(2, 4, torch.tensor([0.7, 0.8], dtype=torch.float32))]}
    selected = {"w": [(0, 2)]}
    new_res = {"w": [(0, 2, torch.tensor([0.01, -0.02], dtype=torch.float32))]}
    new_block = update_block_memory_from_states(local, glob, block_mem, selected, decay=0.9)
    merged = merge_quant_residual_memory(old_res, selected, new_res, quant_decay=1.0)
    # 选中 [0:2] 的 block memory 置 0；未选中 [2:4] = 0.9 * delta
    assert abs(new_block["w"][0].item()) < 1e-8
    assert abs(new_block["w"][2].item() - 0.9) < 1e-6
    # 新 residual 不衰减；未选中旧 residual 原样保留
    sel = [it for it in merged["w"] if it[0] == 0]
    kept = [it for it in merged["w"] if it[0] == 2]
    assert abs(sel[0][2][0].item() - 0.01) < 1e-6
    assert abs(kept[0][2][0].item() - 0.7) < 1e-6


def test_fwht_roundtrip():
    x = torch.randn(1024, dtype=torch.float32)
    y = fwht(x)
    recon = fwht(y) / float(x.numel())
    assert torch.allclose(recon, x, atol=1e-5, rtol=1e-5)


def test_hadamard_roundtrip():
    x = torch.randn(256, dtype=torch.float32)
    signs = generate_rademacher_signs(256, seed=7)
    y = hadamard_transform(x, signs)
    recon = inverse_hadamard_transform(y, signs)
    assert torch.allclose(recon, x, atol=1e-5, rtol=1e-5)
    # 旋转后动态范围应被摊平：amax(y) 通常小于 amax(x) * sqrt(n) / few
    assert float(y.abs().max()) > 0.0


def test_quant_error_metrics_identity():
    x = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    metrics = quantization_error_metrics(x, x.clone())
    assert metrics["rel_l2"] < 1e-7
    assert metrics["cosine"] > 0.999
    assert metrics["sqnr_db"] == float("inf") or metrics["sqnr_db"] > 80.0


def test_e2e_hadamard_mask_cancellation():
    """Hadamard 旋转后 2-client SecAgg 仍能还原平均 delta。"""
    errors = _run_e2e_secagg(n_windows=2, vector_len=512, delta_scale=0.01, hadamard=True)
    for i, err in enumerate(errors):
        assert err < 1e-3, f"hadamard window {i} error {err} too large"


def test_hadamard_global_scale_hides_per_window_amax():
    """Hadamard 只收 global_amax，所有 window scale 相同，不保存 per-window L∞。"""
    windows = build_window_descriptors([[0, "tiny.weight", 0, 64], [1, "big.weight", 0, 64]])
    plan = SecAggPlan(
        modulus_q=DEFAULT_Q, q_max=DEFAULT_Q_MAX,
        quantization_scale=DEFAULT_SCALE,
        hadamard_enabled=True, hadamard_seed=1,
    )
    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=[0, 1], windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=plan,
    )
    client_a = SecAggClient(0, plan, windows)
    client_b = SecAggClient(1, plan, windows)
    # 即使误传 per-window amax，Hadamard 路径也只应折叠成全局 max
    coord.submit_public_key(0, client_a.public_key, window_amax={"0": 1e-6, "1": 1e-2}, global_amax=1e-2)
    result = coord.submit_public_key(1, client_b.public_key, global_amax=5e-3)
    assert result["status"] == "ready"
    assert coord.window_amax == {}
    scales = [float(v) for v in result["window_scales"].values()]
    assert len(set(scales)) == 1
    expected = compute_window_scale(1e-2, DEFAULT_Q_MAX, SCALE_COVERAGE)
    assert abs(scales[0] - expected) / expected < 1e-9
    assert abs(float(result["global_scale"]) - expected) / expected < 1e-9
    assert result.get("secagg_session_id") == coord.session_id


def test_next_power_of_two_and_work_length():
    from shared.fixed_point import hadamard_work_length, next_power_of_two, pad_to_length
    assert next_power_of_two(896) == 1024
    assert next_power_of_two(1024) == 1024
    assert hadamard_work_length(896, True) == 1024
    assert hadamard_work_length(896, False) == 896
    x = torch.arange(3, dtype=torch.float32)
    padded = pad_to_length(x, 8)
    assert padded[:3].tolist() == [0.0, 1.0, 2.0]
    assert float(padded[3:].abs().sum()) == 0.0


def test_mask_refuses_empty_session():
    windows = build_window_descriptors([[0, "w", 0, 8]])
    plan = SecAggPlan(modulus_q=DEFAULT_Q, q_max=DEFAULT_Q_MAX)
    client = SecAggClient(0, plan, windows)
    try:
        client.mask_window(windows[0], torch.zeros(8))
        assert False, "empty session should refuse mask"
    except RuntimeError as exc:
        assert "secagg_session_id" in str(exc)


def test_peer_keys_includes_session():
    windows = build_window_descriptors([[0, "w", 0, 8]])
    plan = SecAggPlan(modulus_q=DEFAULT_Q, q_max=DEFAULT_Q_MAX)
    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=[0, 1], windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=plan,
    )
    waiting = coord.get_peer_keys()
    assert waiting["status"] == "waiting"
    assert waiting["secagg_session_id"] == coord.session_id
    client_a = SecAggClient(0, _fresh_client_plan(), windows)
    client_b = SecAggClient(1, _fresh_client_plan(), windows)
    coord.submit_public_key(0, client_a.public_key, {"0": 1e-4})
    result = coord.submit_public_key(1, client_b.public_key, {"0": 1e-4})
    assert result["secagg_session_id"] == coord.session_id
    ready = coord.get_peer_keys()
    assert ready["status"] == "ready"
    assert ready["secagg_session_id"] == coord.session_id
    assert ready["cohort_hash"] == coord.cohort_hash


def test_e2e_hadamard_padded_non_power_of_two():
    """LayerNorm 一类非 2 的幂窗口 pad 后仍能还原。"""
    errors = _run_e2e_secagg(n_windows=2, vector_len=896, delta_scale=0.01, hadamard=True)
    for i, err in enumerate(errors):
        assert err < 2e-3, f"padded hadamard window {i} error {err} too large"


def test_session_desync_breaks_aggregation():
    """client 用错 session 时 self-mask 无法抵消；大 scale 下泄漏可见（Hadamard 全局 scale 同理）。"""
    windows = build_window_descriptors([[0, "ln.weight", 0, 896]])
    server_plan = SecAggPlan(
        modulus_q=DEFAULT_Q, q_max=DEFAULT_Q_MAX,
        quantization_scale=2.0e-5,  # 模拟 Hadamard 全局 scale
        window_scales={},
    )
    coord = SecAggCoordinator(
        round_idx=1, successful_round_index=0,
        client_ids=[0, 1], windows=windows,
        layout_hash="lh", mask_hash="mh", secagg_plan=server_plan,
    )
    client_a = SecAggClient(0, _fresh_client_plan(), windows)
    client_b = SecAggClient(1, _fresh_client_plan(), windows)
    delta_a = {0: torch.full((896,), 1e-4, dtype=torch.float32)}
    delta_b = {0: torch.full((896,), 2e-4, dtype=torch.float32)}
    coord.submit_public_key(0, client_a.public_key)
    coord.submit_public_key(1, client_b.public_key)
    payload = coord.get_peer_keys()
    _apply_peer_to_clients([client_a, client_b], payload)
    client_a.plan.secagg_session_id = "wrong-session"
    client_b.plan.secagg_session_id = "wrong-session"
    peer_keys = payload["public_keys"]
    client_a.setup_dh(peer_keys)
    client_b.setup_dh(peer_keys)
    z_a = client_a.mask_all_windows(delta_a)
    z_b = client_b.mask_all_windows(delta_b)
    coord.submit_masked_window(0, 0, z_a[0])
    coord.submit_masked_window(1, 0, z_b[0])
    coord.submit_self_master(0, client_a.self_master)
    coord.submit_self_master(1, client_b.self_master)
    coord.freeze_survivors()
    agg = coord.aggregate_and_unmask()
    actual = agg["ln.weight"][0][2]
    expected = (delta_a[0] + delta_b[0]) / 2.0
    err = float((actual - expected).abs().max().item())
    assert err > 0.05, f"desynced session should leak self-mask, err={err}"


if __name__ == "__main__":
    test_quantize_dequantize_precision()
    test_quantize_clip()
    test_overflow_check()
    test_aggregate_zq()
    test_pack_unpack_roundtrip()
    test_stochastic_rounding()
    test_window_scale_does_not_clip_amax()
    test_old_coverage_clips_amax()
    test_quant_error_feedback_identity()
    test_dh_shared_secret_consistency()
    test_prg_determinism()
    test_mask_vector_in_range()
    test_pairwise_mask_identical()
    test_self_mask_different_per_client()
    test_mask_domain_separation()
    test_secagg_plan_serialization()
    test_window_descriptor()
    test_public_scales_use_current_round_amax()
    test_update_public_block_scales_from_agg()
    test_e2e_per_window_mixed_magnitude()
    test_update_block_memory_keeps_quant_residual()
    test_e2e_mask_cancellation()
    test_e2e_different_delta_scales()
    test_e2e_larger_vector()
    test_q_min_abort()
    test_session_id_deterministic()
    test_cohort_hash_deterministic()
    test_hkdf_domain_separation()
    test_sub_state_keep_fp32_avoids_fp16_roundtrip()
    test_extract_window_to_send_fp32_uses_split_residual()
    test_split_error_feedback_independent_decay()
    test_fwht_roundtrip()
    test_hadamard_roundtrip()
    test_quant_error_metrics_identity()
    test_e2e_hadamard_mask_cancellation()
    test_hadamard_global_scale_hides_per_window_amax()
    test_next_power_of_two_and_work_length()
    test_mask_refuses_empty_session()
    test_peer_keys_includes_session()
    test_e2e_hadamard_padded_non_power_of_two()
    test_session_desync_breaks_aggregation()
    print("All Phase B tests passed!")
