# SecAgg 量化精度问题分析与优化方案

> 日期：2026-09-15
> 状态：问题已定位，优化方案已设计，待实现

## 1. 当前 SecAgg 方案

### 1.1 架构

基于远端 `docs/ai-design/FedScale_Windowed_Secure_Aggregation_Implementation_Spec_v2.docx` 规范实现的 Windowed SecAgg：

```
Client 端 (_secagg_upload_and_apply):
  1. 训练 → delta = full_state - global_state (fp32)
  2. 定点量化: q_k = round(delta / scale), clip 到 [-Q_max, Q_max], mod q → Z_q
  3. DH 密钥协商 (X25519): client 间交换公钥, 计算 shared secret
  4. 加 mask: z_k = q_k + pairwise_mask + self_mask (mod q)
  5. 上传 z_k (int16 packed, 2 bytes/元素)
  6. 提交 self_master (32 bytes)
  7. 等待 server 聚合完成
  8. 下载聚合 delta, apply 到 local_global

Server 端 (secagg_coordinator + secagg_try_finalize_round):
  1. 收集所有 client 的 z_k + self_master
  2. 聚合: sum_z = Σ z_k (mod q)
  3. 移除 self mask: sum_q = sum_z - Σ B_k (mod q)
  4. 反量化: agg_delta = dequantize(sum_q, scale) / N
  5. 写 per-block agg_block_key + global_delta_key
```

### 1.2 关键参数

| 参数 | 当前值 | 说明 |
|---|---|---|
| 模数位宽 | 16 (int16) | q = 2^16 = 65536 |
| Q_max | 16383 | 量化值上界 (q/4 - 1, 留 overflow 空间) |
| scale | 自适应 | round 1 用 2^-20, 后续根据 delta_max 调整 |
| 随机舍入 | 开启 | stochastic rounding, E[quantize(x)] = x |
| 传输量 | 2 bytes/元素 | 与 fp16 一致 |
| PRG | SHAKE-256 | 纯标准库, 确定性 |
| DH | X25519 | cryptography 库, server 看不到 shared secret |

### 1.3 安全模型

- **pairwise mask**: X25519 DH 协商, server 不知道 shared secret → server 看不到单个 client 的 q_k + R_kl
- **self mask**: client 生成 self_master, 提交给 server → server 知道 B_k, 但不知道 R_kl → 无法分离 q_k
- **2-client 无掉线**: pairwise mask 自动抵消, self mask 由 server 移除
- **掉线处理**: survivors < q_min → abort (2-client + q_min=2 → 任何掉线 abort)

## 2. 测试结果

### 2.1 协议验证 ✓

5 轮端到端测试 (202609151931, int16, adaptive scale, stochastic rounding):

```
Round 1: train=2.0414 eval=1.498291  (与 fp16 基准一致)
Round 2: train=2.2323 eval=1.732745
Round 3: train=2.7364 eval=1.960495
Round 4: train=3.2869 eval=2.244391
Round 5: train=3.5293 eval=2.361885
```

- ✓ SecAgg 协议完全正确: DH, mask, 聚合, unmask 全部正常
- ✓ 无 NaN, 无爆炸
- ✓ Round 1 与 fp16 基准完全一致
- ✓ 传输量 2B/元素 (与 fp16 一致)
- ✓ 自适应 scale 工作正常
- ✗ Round 2+ loss 缓慢上升 (量化精度问题)

### 2.2 fp16 基准对比

fp16 pipeline (无 SecAgg) 20 轮结果:
```
Round 1: train=2.0414 eval=1.4983
Round 20: train=0.9400 eval=0.9871  (收敛)
```

SecAgg 5 轮:
```
Round 1: train=2.0414 eval=1.4983  (一致)
Round 5: train=3.5293 eval=2.3619  (上升, 未收敛)
```

## 3. 问题根因

### 3.1 mask 完全抵消 (不是问题)

```
z_A = q_A + R_AB + B_A
z_B = q_B - R_AB + B_B
sum_z = (q_A + q_B) + (B_A + B_B)   ← R_AB 精确抵消 ✓
sum_q = sum_z - (B_A + B_B) = q_A + q_B   ← self mask 精确移除 ✓
```

mask 抵消是数学上精确的, 单元测试已验证。**loss 上升不是 mask 的问题**。

### 3.2 定点量化精度 (根本原因)

**fp16 vs int16 定点的精度特性差异:**

| | fp16 (浮点) | int16 定点 (SecAgg) |
|---|---|---|
| 精度类型 | **相对精度** (10-bit mantissa) | **绝对精度** (固定步长 scale) |
| delta=0.001 | 误差 1e-6 (0.1%) | 误差 = scale (可能 5%) |
| delta=1e-6 | 误差 1e-9 (0.1%) | **量化为 0, 100% 丢失** |
| 小值表现 | ✓ 精度跟随值大小 | ✗ 小值相对误差大 |

**fp16 有指数位提供动态范围**, 无论值多小都有 0.1% 的相对精度。
**int16 是固定步长**, 小值的相对误差很大甚至完全丢失。

### 3.3 delta 的实际分布

```
delta = full_state - global_state (训练30步后的权重差值)
  大部分元素: 1e-6 ~ 1e-5 (很小的更新)
  少部分元素: 1e-4 ~ 1e-3 (较大的更新)
  极少数元素: ~3e-4 (max)
```

- fp16: 所有大小的值都有 0.1% 相对误差 → 累积误差小 → 收敛
- int16: 大值 OK, 但小值 (1e-6~1e-7) 相对误差 47%~100% → 大量信息丢失 → 发散

### 3.4 核心矛盾

```
SecAgg 要求: 整数域运算 (mask 精确抵消) → 必须定点量化
2B 限制: 最多 65536 级
小 delta (1e-6): int16 量化为 0 → 有损
fp16 (2B): 小 delta 精度好, 但浮点不满足整数域 mask 抵消
```

**在 2B 限制下, 定点量化对动态范围大的 delta 有固有精度损失。**

### 3.5 各 scale 参数的测试结果

| scale | 可表示范围 | Round 1 | Round 2+ | 结论 |
|---|---|---|---|---|
| 2^-14 (6.1e-5) | ±1.0 | train=2.04 eval=1.50 | **NaN** | delta > 1.0 被 clip → 模型爆炸 |
| 2^-12 (2.4e-4) | ±4.0 | train=2.04 eval=1.50 | **NaN** | delta > 4.0 被 clip → 模型爆炸 |
| 2^-10 (9.8e-4) | ±16.0 | train=2.04 eval=1.50 | **NaN** | delta max=0.0003 量化为 0 → 模型不更新 → 下轮 delta 爆炸 |
| 2^-20 (9.5e-7) | ±0.016 | train=2.04 eval=1.50 | train=2.30→4.81 | 小 delta 精度好, 但 round 2+ delta 超出范围被 clip |
| 自适应 (2^-20→) | 动态 | train=2.04 eval=1.50 | train=2.23→3.53 | **最佳**: 自适应调整, loss 缓慢上升 |
| int24 (2^-22) | ±1.0 | - | - | **OOM** (hex 编码内存翻倍) |

## 4. 优化方案

### 4.1 目标

- ✅ 不增加传输量 (2B/元素, 和 fp16 一致)
- ✅ 无损收敛 (与 fp16 基准一致)
- ✅ SecAgg 安全性不变 (server 看不到单个 client 更新)

### 4.2 方案: per-block adaptive scale + stochastic rounding

**核心思路**: 把 fp16 的"per-value 指数"改为"per-block scale", 让每个 block 的量化精度匹配自己的动态范围。

**当前问题**: 所有 block 共用一个 global scale → 小 delta 的 block 精度差
**优化**: 每个 block 有自己的 scale → 每个 block 的精度匹配自己的 delta 范围

**实现**:
1. server 在 plan 中为每个 block 下发 scale (基于上一轮该 block 的 delta max)
2. client 用 per-block scale 量化
3. mask 仍然在 Z_q 中 (mod 2^16), 但 scale 是 per-block 的
4. server 聚合时用 per-block scale 反量化

**传输开销**: 269 个 block × 4 bytes (float32 scale) = ~1KB (可忽略)
**元素传输**: 仍然 2B/元素 (int16)
→ **不增加传输量**

**精度分析** (per-block scale = delta_max / Q_max):
```
block 内 min/max ratio = 0.01: q_min=164, 相对误差=0.6% (接近 fp16 的 0.1%)
block 内 min/max ratio = 0.1:  q_min=1638, 相对误差=0.1% (等于 fp16)
```

配合 stochastic rounding (E[quantize(x)] = x, 无偏):
- 单轮有误差, 但期望为 0
- 多轮后累积误差 ~ sqrt(N) * single_round_error (不发散)

### 4.3 预期效果

- per-block scale: 每个 block 精度匹配自己的动态范围 (类似 fp16 的指数)
- stochastic rounding: 误差无偏, 多轮不发散
- 收敛轨迹接近 fp16 基准 (可能有微小偏差但不发散)

### 4.4 备选方案 (如果 per-block scale 仍不够)

1. **int24 + 内存优化**: 用 MinIO 直接上传 raw bytes (不走 hex 编码), 避免 OOM。3B/元素, 4M 级量化, 精度远超 fp16。但传输量增加 50%。

2. **浮点 mask (非标准 SecAgg)**: 直接在 fp16 上加 mask, 接受浮点抵消的微小残余误差。mask 大 → 安全但 delta 被淹没; mask 小 → 精度好但不安全。矛盾不可调和。

3. **同态加密 (HE)**: 在密文域聚合, 无需量化。但 ciphertext 远大于 2B (通常 100+ bytes), 不满足传输量限制。

## 5. 已实现的代码

### 5.1 核心模块

| 文件 | 内容 |
|---|---|
| `experiments/shared/fixed_point.py` | 定点量化/反量化, pack/unpack, overflow check, stochastic rounding |
| `experiments/shared/secagg_crypto.py` | X25519 DH, SHAKE-256 PRG, HKDF, pairwise/self mask 生成 |
| `experiments/shared/secagg_client.py` | client 端 SecAgg 状态机 |
| `experiments/server/secagg_coordinator.py` | server 端协调器 |
| `experiments/shared/protocol.py` | SecAggPlan, WindowDescriptor |
| `experiments/server/aggregation_server.py` | REST API + pipeline 集成 |
| `experiments/run_s3r12v3_fsdp.py` | client 端 pipeline 集成 |

### 5.2 配置

```yaml
# configs/s3r12v3-fsdp-secagg-verify.yaml
security:
  secagg_enabled: true
  secagg_modulus_bits: 16       # int16 (2B)
  secagg_scale: 0.0             # 0=自适应
  secagg_stochastic_rounding: true
  secagg_q_min: 0
```

### 5.3 测试

- `experiments/tests/test_secagg.py`: 21 个单元测试 (量化精度, DH 一致性, mask 抵消, 端到端模拟)
- 5 轮端到端验证: 协议正确, 传输量 2B, loss 缓慢上升 (量化精度问题)

## 6. 下一步

1. **实现 per-block adaptive scale**: server 为每个 block 维护独立的 scale, 基于上一轮 delta max 调整
2. **重新验证 5 轮**: 检查 loss 是否不再上升
3. **如果仍不够**: 优化 int24 传输方式 (raw bytes 替代 hex), 解决 OOM
