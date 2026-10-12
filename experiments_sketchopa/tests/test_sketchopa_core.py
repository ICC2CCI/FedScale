"""SketchOPA 核心库单元测试。

运行：
    cd experiments_sketchopa && python -m pytest tests/test_sketchopa_core.py -v
    或 python tests/test_sketchopa_core.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from sketchopa.layout import build_layout, save_calibration
from sketchopa.count_sketch import CountSketchEncoder
from sketchopa.topk import (
    global_topk_flat,
    ResidualBuffer,
    flat_state_dict,
    count_topk_per_layer,
)
from sketchopa.recover import recover_topk, compute_k_recover, build_sparse_update
from sketchopa.quant import quantize, dequantize_fp16, dequantize_int16
from sketchopa.shprg import SHPRG
from sketchopa.seedshare import (
    SHAMIR_PRIME,
    shamir_split,
    shamir_reconstruct,
    shamir_split_vec,
    shamir_reconstruct_vec,
    aggregate_shares_vec,
    TwoClientSeedExchange,
)


def _make_state_dict():
    """构造一个小型合成模型 state_dict（3 层）。"""
    return {
        "layer0.weight": torch.randn(100, 50),   # 5000
        "layer0.bias": torch.randn(100),         # 100
        "layer1.weight": torch.randn(200, 80),   # 16000
        "layer1.bias": torch.randn(200),         # 200
        "layer2.weight": torch.randn(50, 20),    # 1000
        "layer2.bias": torch.randn(50),          # 50
    }


def test_layout_uniform():
    """测试 uniform layout 构造 + budget 公式。"""
    sd = _make_state_dict()
    k = 500
    layout = build_layout(
        sd, k=k, rows=5, value_bytes=2, budget_ratio=1.0,
        allocation_strategy="uniform_param",
    )
    # Σ B_l = 6k / (R × 2) = 6*500 / 10 = 300
    assert layout.total_buckets_per_row == 300, f"expected 300, got {layout.total_buckets_per_row}"
    # sketch_len = R × Σ B_l
    assert layout.sketch_len == 5 * 300
    # payload = sketch_len × value_bytes = 3000
    assert layout.payload_bytes == 3000
    # hash 非空
    assert len(layout.layout_hash) == 64
    print("[OK] test_layout_uniform")


def test_layout_calibrated(tmp_path=None):
    """测试 calibrated layout（按 topk 计数分配）。"""
    import tempfile, json
    sd = _make_state_dict()
    k = 500

    # 造 calibration 文件：layer1.weight 命中最多
    counts = {
        "layer0.weight": 100,
        "layer0.bias": 5,
        "layer1.weight": 400,  # 最多
        "layer1.bias": 10,
        "layer2.weight": 50,
        "layer2.bias": 3,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(counts, f)
        cal_path = f.name

    layout = build_layout(
        sd, k=k, rows=5, value_bytes=2, budget_ratio=1.0,
        allocation_strategy="topk_count_calibrated",
        calibration_path=cal_path,
    )
    # layer1.weight 应分到最多 bucket
    bw = {l.name: l.buckets_per_row for l in layout.layers}
    assert bw["layer1.weight"] > bw["layer0.weight"], f"layer1.weight should have most buckets: {bw}"
    assert bw["layer1.weight"] > bw["layer2.weight"], f"{bw}"
    print("[OK] test_layout_calibrated", bw)

    import os
    os.unlink(cal_path)


def test_count_sketch_linearity():
    """CountSketch 线性性：CS(x) + CS(y) == CS(x+y)。"""
    sd = _make_state_dict()
    k = 500
    layout = build_layout(sd, k=k, rows=5, value_bytes=2, budget_ratio=1.0)
    encoder = CountSketchEncoder(layout, sketch_seed=42)

    # 构造两组稀疏更新
    d = sum(t.numel() for t in sd.values())
    x = torch.zeros(d)
    x[torch.randperm(d)[:k]] = torch.randn(k)
    y = torch.zeros(d)
    y[torch.randperm(d)[:k]] = torch.randn(k)

    coords_x = x.nonzero().squeeze(1)
    vals_x = x[coords_x]
    coords_y = y.nonzero().squeeze(1)
    vals_y = y[coords_y]

    cs_x = encoder.encode(coords_x, vals_x)
    cs_y = encoder.encode(coords_y, vals_y)
    cs_xy = encoder.encode(
        torch.cat([coords_x, coords_y]),
        torch.cat([vals_x, vals_y]),
    )

    diff = (cs_x + cs_y - cs_xy).abs().max().item()
    assert diff < 1e-4, f"linearity violated: max diff {diff}"
    print(f"[OK] test_count_sketch_linearity (max diff {diff:.2e})")


def test_topk_flat():
    """测试 global Top-k。"""
    d = 10000
    g = torch.randn(d)
    k = 100
    coords, vals = global_topk_flat(g, k)
    assert coords.numel() == k
    # 验证确实选的是 |g| 最大的 k 个
    expected = g.abs().topk(k).indices.sort().values
    got = coords.sort().values
    assert torch.equal(expected, got)
    print("[OK] test_topk_flat")


def test_residual_buffer():
    """测试 ResidualBuffer：r_{i+1} = z - Sparse(S, v)。"""
    d = 1000
    k = 50
    buf = ResidualBuffer(layer_meta=[("all", 0, d)])
    g = torch.randn(d)

    z = g + buf.buffer  # r_0=0
    coords, vals = buf.add_and_select(g, k)
    # 残差 = z - sparse
    sparse = torch.zeros(d)
    sparse[coords] = vals
    expected_r = z - sparse
    assert torch.allclose(buf.buffer, expected_r, atol=1e-5), "residual update wrong"
    print("[OK] test_residual_buffer")


def test_residual_clip():
    """测试残差 clip。"""
    d = 1000
    k = 50
    buf = ResidualBuffer(layer_meta=[("all", 0, d)], clip_norm=1.0)
    g = torch.randn(d) * 100  # 大梯度
    buf.add_and_select(g, k)
    norm = buf.buffer.norm().item()
    assert norm <= 1.0 + 1e-4, f"clip failed: norm={norm}"
    print(f"[OK] test_residual_clip (norm={norm:.4f})")


def test_end_to_end_recover():
    """端到端：Top-k → 编码 → 聚合 → 恢复。"""
    sd = _make_state_dict()
    d = sum(t.numel() for t in sd.values())
    k = 500
    layout = build_layout(sd, k=k, rows=5, value_bytes=2, budget_ratio=2.0)  # 大 budget 提升恢复
    encoder = CountSketchEncoder(layout, sketch_seed=42)

    # 2 个 client
    num_clients = 2
    all_coords = []
    sketches = []
    for ci in range(num_clients):
        g = torch.randn(d)
        coords, vals = global_topk_flat(g, k)
        all_coords.append(coords)
        sketches.append(encoder.encode(coords, vals))

    agg = sum(sketches)
    candidates = torch.cat(all_coords)
    k_rec = compute_k_recover(k, num_clients, rho=2.0, d=d)
    coords_r, vals_r, stats = recover_topk(agg, encoder, layout, candidates, k_rec)

    # 基本断言
    assert coords_r.numel() == min(k_rec, candidates.unique().numel())
    assert vals_r.numel() == coords_r.numel()
    print(f"[OK] test_end_to_end_recover (candidates={stats['num_candidates']}, recovered={coords_r.numel()})")


def test_quant_fp16():
    """测试 fp16 量化往返。"""
    x = torch.randn(1000) * 0.01
    q, scale, stats = quantize(x, mode="float16")
    x2 = dequantize_fp16(q)
    err = (x - x2).abs().max().item()
    assert err < 1e-3, f"fp16 quant error too large: {err}"
    assert stats.mode == "float16"
    print(f"[OK] test_quant_fp16 (max err {err:.2e})")


def test_quant_int16():
    """测试 int16 public scale 量化往返。"""
    x = torch.randn(1000) * 0.001
    q, scale, stats = quantize(x, mode="int16", scale=1e6)
    x2 = dequantize_int16(q, scale)
    err = (x - x2).abs().max().item()
    assert err < 1e-5, f"int16 quant error too large: {err}"
    assert stats.mode == "int16"
    print(f"[OK] test_quant_int16 (max err {err:.2e})")


def test_shprg_homomorphism():
    """SHPRG 同态性：SHPRG(s1) + SHPRG(s2) == SHPRG(s1+s2) (mod q)。"""
    q = 2**16
    shprg = SHPRG(length=500, ctx_hash="test_ctx", modulus_q=q, seed_dim=4)
    s1 = [123, 456, 789, 1011]
    s2 = [222, 333, 444, 555]

    m1 = shprg.expand(s1)
    m2 = shprg.expand(s2)
    s_sum = SHPRG.add_seed_vectors(s1, s2, q)
    m_sum = shprg.expand(s_sum)

    diff = ((m1 + m2 - m_sum) % q).max().item()
    assert diff == 0, f"SHPRG homomorphism violated: {diff}"
    print(f"[OK] test_shprg_homomorphism")


def test_shprg_mask_unmask():
    """SHPRG mask/unmask 往返（Z_q 域）：U = (ct - SHPRG(s)) mod q 还原 sketch_zq。"""
    torch.manual_seed(0)
    from sketchopa.quant import quantize_to_zq, zq_to_float
    q = 2**16
    shprg = SHPRG(length=500, ctx_hash="test_ctx", modulus_q=q, seed_dim=4)
    sketch_fp = torch.randn(500) * 0.05
    scale = 1e5  # 保证 |sketch*scale| < q/2 = 32768
    sketch_zq = quantize_to_zq(sketch_fp, scale, q)
    s = [111, 222, 333, 444]

    ct = shprg.mask_sketch_zq(sketch_zq, s)
    recovered_zq = shprg.unmask_sketch_zq(ct, s)
    recovered_fp = zq_to_float(recovered_zq, scale, q)
    err = (sketch_fp - recovered_fp).abs().max().item()
    assert err < 1e-4, f"mask/unmask error: {err}"
    print(f"[OK] test_shprg_mask_unmask (err {err:.2e})")


def test_shamir_scalar():
    """Shamir 标量秘密分享 + 重构。"""
    import random as rnd
    rng = rnd.Random(42)
    q = SHAMIR_PRIME
    secret = 12345
    shares = shamir_split(secret, n=5, t=3, modulus=q, rng=rng)
    # 用任意 3 份重构
    subset = shares[:3]
    recovered = shamir_reconstruct(subset, q)
    assert recovered == secret % q, f"expected {secret%q}, got {recovered}"
    # 用另一组 3 份
    subset2 = [shares[0], shares[2], shares[4]]
    recovered2 = shamir_reconstruct(subset2, q)
    assert recovered2 == secret % q
    print("[OK] test_shamir_scalar")


def test_shamir_vec():
    """Shamir 向量秘密分享 + 重构。"""
    import random as rnd
    rng = rnd.Random(42)
    q = SHAMIR_PRIME
    secret_vec = [100, 200, 300, 400]
    shares = shamir_split_vec(secret_vec, n=5, t=3, modulus=q, rng=rng)
    subset = shares[:3]
    recovered = shamir_reconstruct_vec(subset, q)
    assert recovered == [s % q for s in secret_vec], f"{recovered} != {secret_vec}"
    print("[OK] test_shamir_vec")


def test_two_client_seed_exchange():
    """2-client 退化 seed 交换：s_sum = s_0 + s_1。"""
    q = SHAMIR_PRIME
    exchange = TwoClientSeedExchange(modulus=q)
    s0 = [100, 200, 300, 400]
    s1 = [111, 222, 333, 444]

    # 双方互发
    share0 = exchange.client_pack_share(s0)
    share1 = exchange.client_pack_share(s1)
    # 辅助节点聚合
    agg0 = exchange.auxiliary_aggregate(s0, share1)  # client0 视角
    agg1 = exchange.auxiliary_aggregate(s1, share0)  # client1 视角
    # server 重构（t=1，收一份即可）
    recovered = exchange.server_reconstruct([agg0])
    expected = [(a + b) % q for a, b in zip(s0, s1)]
    assert recovered == expected, f"{recovered} != {expected}"
    # agg0 和 agg1 应相等
    assert agg0 == agg1
    print("[OK] test_two_client_seed_exchange")


def test_count_topk_per_layer():
    """测试各层 Top-k 计数统计（用于 calibration）。"""
    sd = _make_state_dict()
    k = 500
    layout = build_layout(sd, k=k, rows=5, value_bytes=2, budget_ratio=1.0)
    d = sum(t.numel() for t in sd.values())
    g = torch.randn(d)
    coords, _ = global_topk_flat(g, k)
    counts = count_topk_per_layer(coords, layout)
    assert sum(counts.values()) == k
    assert all(v >= 0 for v in counts.values())
    print("[OK] test_count_topk_per_layer", counts)


def test_secure_aggregation_pipeline():
    """安全聚合管线（Z_q 域）：2 client 量化→mask→聚合→unmask→反量化→恢复。

    验证安全层不影响聚合结果（与明文聚合一致）。
    SHPRG 和 seedshare 共用 SHAMIR_PRIME 模数。
    """
    from sketchopa.seedshare import SHAMIR_PRIME
    from sketchopa.quant import quantize_to_zq, zq_to_float
    sd = _make_state_dict()
    d = sum(t.numel() for t in sd.values())
    k = 500
    layout = build_layout(sd, k=k, rows=5, value_bytes=2, budget_ratio=2.0)
    encoder = CountSketchEncoder(layout, sketch_seed=42)

    q = SHAMIR_PRIME
    scale = 1e6  # sketch 量化 scale
    shprg = SHPRG(length=layout.sketch_len, ctx_hash="e2e_test", modulus_q=q, seed_dim=4)
    exchange = TwoClientSeedExchange(modulus=q)

    # 明文聚合（基准，量化到 Z_q 再反量化）
    all_coords = []
    plain_zq_list = []
    seeds = []
    for ci in range(2):
        g = torch.randn(d)
        coords, vals = global_topk_flat(g, k)
        all_coords.append(coords)
        sk = encoder.encode(coords, vals)
        sk_zq = quantize_to_zq(sk, scale, q)
        plain_zq_list.append(sk_zq)
        seeds.append([10 + ci*100, 20 + ci*100, 30 + ci*100, 40 + ci*100])

    plain_agg_zq = sum(plain_zq_list) % q  # 明文聚合在 Z_q
    plain_agg_fp = zq_to_float(plain_agg_zq, scale, q)

    # 安全聚合：量化→mask→求和→unmask
    cts = []
    for ci in range(2):
        ct = shprg.mask_sketch_zq(plain_zq_list[ci], seeds[ci])
        cts.append(ct)
    agg_ct = sum(cts) % q
    agg_share = exchange.auxiliary_aggregate(seeds[0], exchange.client_pack_share(seeds[1]))
    s_sum = exchange.server_reconstruct([agg_share])
    secure_agg_zq = shprg.unmask_sketch_zq(agg_ct, s_sum)
    secure_agg_fp = zq_to_float(secure_agg_zq, scale, q)

    err = (plain_agg_fp - secure_agg_fp).abs().max().item()
    assert err < 1e-4, f"secure vs plain mismatch: {err}"
    print(f"[OK] test_secure_aggregation_pipeline (err {err:.2e})")


def main():
    """运行所有测试。"""
    test_layout_uniform()
    test_layout_calibrated()
    test_count_sketch_linearity()
    test_topk_flat()
    test_residual_buffer()
    test_residual_clip()
    test_end_to_end_recover()
    test_quant_fp16()
    test_quant_int16()
    test_shprg_homomorphism()
    test_shprg_mask_unmask()
    test_shamir_scalar()
    test_shamir_vec()
    test_two_client_seed_exchange()
    test_count_topk_per_layer()
    test_secure_aggregation_pipeline()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
