# SecAgg 量化精度问题分析与优化方案

> 日期：2026-09-15（问题定位）/ 2026-09-16（方案实现）/ 2026-09-17（旧 20 轮）/ 2026-09-18（Hadamard 对齐 fp16，含复跑与时间优化）
> 状态：**已解决并成为默认路径**（Hadamard + Issue #1 + 单 blob + 异步 finalize）。
>
> | 阶段 | 结果目录 | R20 eval | 说明 |
> |---|---|---|---|
> | 修复前发散 | 早期 5 轮 | — | R5 飙到 2.36 |
> | per-window amax 修复 | `202609162025` | **1.328** | 不再发散，但仍落后 fp16 |
> | Hadamard + Issue #1 | `202609171809` | **0.985** | ≈ fp16 `final-clean` **0.987** |
> | 关掉 profiler 复跑 | **`202609180941`** | **0.985** | `train≈38s`，整轮≈148s；传输记账正常 |
> | 时间优化 5 轮 | **`202609181151`** | （R5=1.364，与 941 逐轮一致） | 单 blob + 并行 unmask + 异步写盘；整轮≈110–134s |
> | 时间优化 20 轮 | **`202609181406`** | **0.985** | 与 941 对齐；整轮≈113s |
>
> **当前端到端流程以 §3.1「Hadamard 默认路径」为准**；§2 / §3.1.1 的 per-window 描述是历史中间态。  
> 调研与 Phase：`docs/algorithm/2026-09-17-secagg-quantization-optimization-survey.md`。  
> 联调总览：`docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md` §3.1。  
> 时间切片（已结项）：`docs/exec-plans/completed/2026-09-18-secagg-time-efficiency.md`。

## 1. 问题概述

### 1.1 现象

SecAgg（安全聚合）5 轮端到端测试中，Round 1 与 fp16 基线一致（eval=1.498），但从 Round 2 起 eval loss **单调上升**，到 Round 5 达到 2.36（发散）。fp16 基线 20 轮 eval 从 1.498 收敛到 0.987。

```
修复前（5 轮）:
R1: eval=1.498  R2: eval=1.733  R3: eval=1.960  R4: eval=2.244  R5: eval=2.362  ← 上升

per-window amax 修复后（20 轮, 202609162025）:
R1: eval=1.498  R5: eval=1.432  R20: eval=1.328  ← 下降但仍落后 fp16

Hadamard + Issue #1（当前默认, 202609180941）:
R1: eval=1.498  R5: eval=1.364  R10: eval=1.091  R20: eval=0.985  ← 对齐 fp16≈0.987
```

### 1.2 排除项

- **mask 抵消**：数学上精确抵消（`z_A + z_B = (q_A + q_B) + (B_A + B_B)`，pairwise mask 完全消去），单元测试验证。**不是问题**。
- **协议正确性**：DH、mask、聚合、unmask 全部正常，无 NaN、无爆炸。**不是问题**。

### 1.3 根因：定点量化在「全局一个 scale」下把有效更新剪坏了

SecAgg 必须在整数域加 mask，所以 delta 只能先变成定点。fp16 是相对精度（再小的数也有约 0.1% 误差）；int16 是绝对步长 scale。一轮里大部分权重更新在 1e-6~1e-5，少数到 1e-4~1e-3。**一个全局 scale 无法同时保住这两头**：

| scale 选择 | 可表示范围 | 问题 |
|---|---|---|
| 太大（2^-10） | ±16 | 小更新（1e-6）量化为 0 → 下轮 delta 爆炸 → NaN |
| 太小（2^-20） | ±0.016 | 大更新（3e-4）被 clip → 有效梯度被砍掉 → loss 缓慢上升 |
| 自适应（旧版） | 动态 | coverage=0.9 方向反了 + 用聚合后 max 估下轮 scale → 每轮缩 10% → 慢发散 |

## 2. 修复方案

### 2.1 核心思路

**历史（§2.2）**：先把「一个全局 scale」改成 **per-window 当轮 amax**，修好 coverage / 滞后 scale / 无 residual，eval 从发散变为可下降（R20=1.328）。

**当前默认**：在此基础上再上 **Hadamard 旋转 + 全局 `global_amax`（1 个标量）+ Issue #1（FP32 delta / quant residual 不衰减）+ MinIO raw bytes**，R20 eval≈0.985。  
下面 §2.2 保留 per-window 五项修复记录；实现与配置以 §3.1 / §5.2 为准。

### 2.2 五项修复（per-window 中间态，已归档）

#### A. Per-window scale（`SecAggPlan.window_scales: Dict[window_id, float]`）

- `SecAggPlan.quantization_scale` 从标量改成 `Dict[str, float]`（`window_scales`）
- client `mask_window` 按 window 取 scale，server `dequantize_from_zq` 按 window 取 scale
- 传输开销：269 个 float ≈ 1KB（可忽略）；元素传输仍 2B/元素

#### B. 当轮 amax 对齐 scale（关键修复）

**旧逻辑（有缺陷）**：用上一轮聚合后的 `agg_delta` max 估下一轮 scale → 聚合后 max 系统性偏小 → scale 每轮缩 10% → 慢发散。

**新逻辑**：每轮 client 在 key-announce 阶段顺带上报每个 window 的 `max|delta|`（269 个 float ≈ 1KB），server 收齐后取跨 client 最大值：

```
scale_b = max_k(|delta_{k,b}|) / Q_max * 1.05
```

- `1.05`（`SCALE_COVERAGE`）= 余量：最大值映射到 `0.95 * Q_max`，**不 clip**
- 所有 client 用同一套 scale（整数域求和的前提）
- 泄露每 window 的 L∞，不是更新本身；2-client 实验完全可接受
- Round 1 也走「上报 amax → 下发 scale → 再量化」，不用全局 2^-20 兜底

#### C. clip / zero 打点

每个 window 记录 `clip_frac`（被截断的比例）和 `zero_frac`（量化为 0 的比例）。`clip_frac > 0` 说明 scale 偏小；`zero_frac` 接近 1 说明 scale 偏大。`quantize_to_zq_with_feedback` 返回 residual + stats。

#### D. 量化 error-feedback

选中 block **不再清零**，改为写入量化残差：

```
residual = true_delta - dequant(quant(true_delta))
mem = residual * memory_decay
```

被 clip / 圆成 0 的分量下一轮还能发出去。这是 QSGD 能收敛的常规做法，和现有 `memory_decay=0.9` 是同一条管线。

#### E. stochastic rounding + fp32 量化

- `stochastic_rounding: true`：范围内舍入无偏（`E[quantize(x)] = x`）
- 量化作用在 fp32 的 `to_send` 上（`extract_window_slices` 强制 fp32），不让 fp16 截断后再用过粗的定点 scale

### 2.3 旧逻辑的三个缺陷及修复对照

| 缺陷 | 旧逻辑 | 新逻辑 |
|---|---|---|
| ① coverage=0.9 方向反 | `scale = delta_max / Q_max * 0.9` → amax 被映射到 Q_max/0.9 > Q_max → 一定被 clip | `SCALE_COVERAGE=1.05` → amax 映射到 0.95*Q_max，不 clip |
| ② 用聚合后 max 估下轮 scale | `new_scale = agg_max / Q_max * 0.9 ≤ old_scale * 0.9` → 每轮缩 10% | 当轮 client 上报 amax → server 取 max_k → 无滞后、无收缩 |
| ③ 量化误差不进 residual | 选中 block 直接置 0 | `update_block_memory_with_quant_residual`：写 residual * memory_decay |

## 3. 完整流程（以当前默认路径为准）

### 3.1 每轮执行流程 — Hadamard 默认路径（`secagg_hadamard: true`）

```
┌─────────────────────────────────────────────────────────────────┐
│ Round N（当前默认：Hadamard + Issue #1 + raw bytes）              │
│                                                                  │
│  Client 端 (每个 client):                                        │
│  1. 训练 30 步 → delta = full_state - global_state（保持 FP32）  │
│  2. to_send = delta + block_memory + quant_residual（FP32）      │
│  3. 在旋转域估 global_amax（所有 window 共用；只报 1 个标量）     │
│  4. key-announce: POST pk_hex + global_amax                      │
│  5. peer-keys: 收齐后拿 public_keys + 统一 scale                 │
│     + secagg_session_id（必须写回本地 plan，禁止空 session）     │
│  6. 逐 window:                                                   │
│     a. pad 到 2 的幂（含 LayerNorm 等非 p2 长度）                 │
│     b. 符号翻转 D → FWHT → 用全局 scale 做 INT16 随机舍入        │
│     c. z_k = q_k + pairwise_mask + self_mask (mod q)             │
│     d. 全部 window 打成一个 blob PUT；HTTP 只 POST blob z_key    │
│  7. Memory 拆分更新:                                             │
│     - 选中: quant_residual = x - D(Q(x))（decay=1.0）            │
│             block_memory 置 0                                    │
│     - 未选中: block_memory = (delta+block_memory)*0.9            │
│  8. 提交 self_master（server 后台 finalize，HTTP 先回 200）      │
│  9. 等聚合完成 → 下载 global_delta → apply                       │
│                                                                  │
│  Server 端:                                                      │
│  1. key-announce: 收集 pk + global_amax                          │
│  2. scale = max_k(global_amax) / Q_max * 1.05（全局一份）        │
│  3. peer-keys: 下发 public_keys + scale + session                │
│  4. finalize 时拉 2 个 blob → unpack_zq（notify 不 GET）         │
│  5. 按窗并行：sum_z = Σ z_k (mod q)；去掉 Σ self_mask → sum_q    │
│  6. 反量化 → 逆 FWHT → 去 pad → FedAvg                           │
│  7. 写 global_delta 后立刻 done；全量 global_state 异步 PUT      │
└─────────────────────────────────────────────────────────────────┘
```

数据路径摘要：

```
FP16 模型参数
  → FP32 delta（不 round 回 FP16）          # Issue #1
  → FP32 (block_memory + quant_residual)
  → pad-p2 → D → FWHT                       # Hadamard
  → INT16 + stochastic rounding（全局 scale）
  → pairwise + self mask (mod q)
  → MinIO 单 blob + z_key 通知
  → Server unmask → 反量化 → 逆 FWHT → FedAvg
```

### 3.1.1 历史流程 — per-window 当轮 amax（`secagg_hadamard: false`）

曾用于把 eval 从发散拉回可下降（`202609162025`，R20=1.328）。Hadamard 默认开启后一般不再走这条路径；保留作对照与隐私退路说明。

```
┌─────────────────────────────────────────────────────────────────┐
│ Round N（旧：per-window 当轮 amax）                               │
│                                                                  │
│  Client:                                                         │
│  1–2. 同左（训练 + extract windows）                             │
│  3. window_amax_payload → {window_id: max|delta|}（~269 float）  │
│  4. key-announce: POST pk + window_amax                          │
│  5. peer-keys: 下发 per-window window_scales                     │
│  6. 逐 window: scale_b 量化 → mask → 上传（早期为 hex）          │
│  7. residual 写入 memory（早期曾 residual×0.9，现已拆分）         │
│                                                                  │
│  Server:                                                         │
│  scale_b = max_k(window_amax[k][b]) / Q_max * 1.05               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 关键参数（当前默认）

| 参数 | 值 | 说明 |
|---|---|---|
| 模数位宽 | 16 (int16) | q = 2^16 = 65536 |
| Q_max | 16383 | 量化值上界 (q/4 - 1) |
| scale 模式 | **全局 scale（Hadamard）** | `secagg_hadamard: true`；只收 `global_amax` |
| 非 Hadamard 退路 | per-window 当轮 amax | `secagg_hadamard: false`；`secagg_scale: 0.0` |
| SCALE_COVERAGE | 1.05 | amax 映射到 0.95*Q_max，减少 clip |
| 随机舍入 | 开启 | stochastic rounding，E[quantize(x)] = x |
| `memory_decay` | 0.9 | 仅未选中 block 的 block-mask residual |
| `quant_residual_decay` | **1.0** | 量化残差不衰减（Issue #1） |
| 传输 | MinIO 单 blob（SAW1） | ~2B/元素；控制面一次 `z_key` |
| amax 开销 | **1 float/client/轮** | Hadamard；旧路径为 ~269 float |

### 3.3 安全模型

- **pairwise mask**: X25519 DH 协商, server 不知道 shared secret → server 看不到单个 client 的 q_k + R_kl
- **self mask**: client 生成 self_master, 提交给 server → server 知道 B_k, 但不知道 R_kl → 无法分离 q_k
- **2-client 无掉线**: pairwise mask 自动抵消, self mask 由 server 移除
- **掉线处理**: survivors < q_min → abort
- **session 同步**: peer-keys 必须把 `secagg_session_id` 写回 client；空 session 会导致 self-mask 与 server 不一致（Hadamard 下尤其危险）
- **amax 泄露**:
  - **当前（Hadamard）**: 每轮只泄露 1 个 `global_amax`（旋转域 L∞），**不再**泄露 per-window L∞
  - **旧（per-window）**: 每轮泄露 ~269 个 float；2-client 实验可接受。更强隐私可退到「公开聚合 max」路径，但精度会降

### 3.4 隐私泄露分析

当前方案在 SecAgg 安全聚合之上，为对齐量化 scale，仍有少量额外泄露。以下按**当前默认（Hadamard）**列出；括号注明旧路径差异。

#### 3.4.1 泄露清单

| # | 泄露内容 | 泄露给谁 | 每轮数据量 | 泄露程度 | 引入原因 |
|---|---|---|---|---|---|
| 1 | **global_amax（旋转域 L∞）** | server | **1 个 float**（旧路径：269 个 per-window L∞） | **低–中** | scale 对齐；Hadamard 已大幅削减 |
| 2 | **masked window z_k** | server | ~115–140 MiB raw | 低（已被 mask 保护） | SecAgg 协议本身 |
| 3 | **self_master（32 bytes）** | server | 32B × 2 client | 低（只能算 B_k，无法分离 q_k） | SecAgg 协议本身 |
| 4 | **X25519 公钥** | server + 其他 client | 32B × 2 | 无（公钥设计上公开） | SecAgg 协议本身 |
| 5 | **train_loss / eval_loss** | server | 2 个 float × 2 client | **中** | upload-complete body 里携带 |
| 6 | **num_examples** | server | 1 个 int × 2 client | 低 | upload-complete body |
| 7 | **per-round timing** | server | ~15 个 float × 2 client | 低 | upload-complete body |
| 8 | **block_energies** | server | 每 block 一个 float | 低 | upload-complete body |
| 9 | **per-window z_k 的 POST 时序** | server | 旧路径 ~269 个时间戳 | 低（侧面信息） | **已改为单 blob**；旧逐窗上传才有 |

#### 3.4.2 关键泄露 #1：amax（scale 对齐引入）

**Hadamard 当前路径泄露什么**：整轮所有 window 在旋转域上的一个全局 `max|y|`。server 能看到本轮更新「总体幅度」和 client 间谁更大，但**看不到哪个 block 更新大**。

**旧 per-window 路径额外泄露**：每个 window 的 `max|delta|` → 可推断 embedding / lm_head vs LayerNorm 等块的相对活跃度（见历史分析）。Hadamard 默认开启后这条路径默认关闭。

**server 仍无法推断**：delta 方向、具体坐标值、argmax 位置；masked `z_k` 不可反推单 client `q_k`。

**风险评级**：Hadamard 下为低–中；旧 per-window 为中。2-client 实验可接受。

**更强隐私退路**：`update_public_block_scales`（公开聚合 max，headroom=3.0）——不读单 client amax，但有一轮滞后且精度较差（曾导致 R2 eval 上升）。

#### 3.4.3 关键泄露 #5：train_loss / eval_loss

**泄露什么**：每个 client 每轮的 train_loss 和 eval_loss（标量）。

**这不是 SecAgg 引入的**——fp16 基线路径也泄露同样的信息（在 `upload-complete` body 里）。但需要注意：train_loss + amax 组合后，server 能更准确地推断训练状态。

**如果需要消除**：把 train_loss / eval_loss 从 SecAgg 路径的 POST body 中移除，改用单独的加密通道上报，或在聚合完成后才上报（server 无法关联到单 client 的 z_k）。但这不是 SecAgg 量化方案的问题，是整个 pipeline 的设计选择。

#### 3.4.4 SecAgg 协议本身的保护（不泄露）

以下信息 server **无法看到**：

| 信息 | 保护机制 |
|---|---|
| 单个 client 的 delta 方向/值 | pairwise mask（DH 保护，server 不知道 R_kl） |
| 单个 client 的 q_k | pairwise mask + self mask（需同时知道 R_kl 和 B_k 才能分离） |
| delta 的具体元素（除 L∞ 外） | z_k 被 mask，server 只能看到 sum_z = Σ z_k |
| shared secret | X25519 DH（server 不参与 DH 协商） |

#### 3.4.5 安全模型总结

```
Server 能看到（Hadamard 当前默认）:
  - 每轮 1 个 global_amax（旋转域 L∞）← scale 对齐；旧路径曾是 269 个 per-window L∞
  - masked z_k（被 mask 保护，看不到 q_k）
  - self_master（能看到 B_k，但缺 R_kl 无法分离）
  - train/eval loss（pipeline 本身泄露，非 SecAgg 引入）
  - 聚合后的 agg_delta（设计上公开）

Server 看不到:
  - 单 client 的 delta 方向/具体值
  - shared secret / pairwise mask
  - q_k（需 R_kl + B_k 同时泄露）
```

**结论**：Hadamard 路径只额外泄露 1 个 `global_amax`；旧 per-window 路径泄露 269 个 L∞。详见调研文档 §3.2。

## 4. 验证结果

### 4.1 20 轮端到端验证（2026-09-17, results/202609162025）

```
R 1: train=2.0414 eval=1.498291  ← 与 fp16 基线一致
R 2: train=1.7123 eval=1.438641
R 3: train=1.7440 eval=1.441008
R 4: train=1.7654 eval=1.443681
R 5: train=1.7132 eval=1.432033
R 6: train=1.7763 eval=1.451301
R 7: train=1.7079 eval=1.427232
R 8: train=1.7740 eval=1.416400
R 9: train=1.6449 eval=1.377002
R10: train=1.7275 eval=1.396150
R11: train=1.7155 eval=1.388097
R12: train=1.7189 eval=1.409219
R13: train=1.7718 eval=1.424472
R14: train=1.6729 eval=1.391206
R15: train=1.7685 eval=1.415414
R16: train=1.6821 eval=1.388314
R17: train=1.7667 eval=1.417587
R18: train=1.6747 eval=1.400741
R19: train=1.7204 eval=1.362505
R20: train=1.6943 eval=1.327735  ← 收敛

eval: 1.4983 → 1.3277 (下降 11.4%)
全程 clip_frac=0, zero_frac≈0.0003
```

### 4.2 对比

| 方案 | R1 eval | R5 eval | R20 eval | 趋势 |
|---|---|---|---|---|
| fp16 基线（无 SecAgg） | 1.498 | — | 0.987 | 收敛 |
| 修复前 SecAgg | 1.498 | 2.362 | — | **发散** |
| 修复后 SecAgg | 1.498 | 1.432 | 1.328 | **收敛** |

### 4.3 clip / zero 监控

20 轮全部 `clip_windows=0/269 max_clip=0`，per-window scale 每轮自适应（scale_min/scale_max 随 delta 分布变化）。`zero_frac≈0.0003`（极少数极小值量化为 0，正常）。

## 4x. 与 fp16 基线的差距分析

### 4x.1 差距现状

修复后 SecAgg 不再发散，但与 fp16 基线（`results/20260911-10pct-v2`）仍有显著差距：

```
Round | fp16 train  fp16 eval | SecAgg train SecAgg eval
------+-----------------------+--------------------------
R 1   |     2.0414   1.4983   |      2.0414    1.4983    ← R1 完全一致
R 5   |     1.4200   1.3772   |      1.7132    1.4320    ← 差距开始拉大
R 9   |     1.1391   1.1064   |      1.6449    1.3770    ← fp16 快速下降，SecAgg 缓慢
R20   |     0.9400   0.9871   |      1.6943    1.3277    ← 差 35%
```

| 指标 | fp16 基线 | 修复后 SecAgg | 差距 |
|---|---|---|---|
| R20 eval loss | 0.987 | 1.328 | SecAgg 高 35% |
| R20 train loss | 0.940 | 1.694 | SecAgg 高 80% |
| eval 下降幅度 | 34.1% | 11.4% | — |
| 每轮耗时 | ~107s | ~160s | SecAgg 慢 50% |
| encode 耗时 | ~7s | ~55s | SecAgg 慢 8x |
| upload 耗时 | ~7s | ~48s | SecAgg 慢 7x |
| 传输量 | ~125 MiB | ~135 MiB | 1.08x（接近） |

### 4x.2 根因 1：int16 定点量化固有精度损失（收敛速度慢）

**这是 eval loss 差距的主要原因。**

Round 1 两路径完全一致（eval=1.4983），因为 R1 的 delta 较小且分布均匀，per-window scale 能精确覆盖。从 R2 起差距拉大，根因是 int16 定点与 fp16 浮点的本质精度差异：

- **fp16**：10-bit mantissa，每个值都有 ~0.1% 相对精度，无论值多小
- **int16 per-window scale**：14-bit 有效精度（Q_max=16383），window 内动态范围越大，小值的相对误差越大

即使 per-window scale 消除了 clip（`clip_frac=0`），**量化舍入误差**仍然存在：
- `zero_frac≈0.0003`：每轮约 0.03% 的元素量化为 0（完全丢失）
- window 内 min/max ratio 大的 block（如 embedding 层），小值的相对误差可达 1-5%
- 这些误差通过 error-feedback 部分回收，但不是 100%——`memory_decay=0.9` 意味着每轮 residual 衰减 10%

**累积效应**：单轮误差小（~0.1%），但 20 轮累积后 train loss 差距达 80%（0.94 vs 1.69），eval loss 差距 35%（0.99 vs 1.33）。fp16 基线在 R9 出现「loss cliff」（train 1.14→1.06，eval 1.11→1.09），SecAgg 没有出现这个快速下降阶段——量化噪声阻碍了模型找到那个下降路径。

### 4x.3 根因 2：encode 慢 8x（每轮多 ~48s）

**fp16 encode**：`encode_block_delta(to_send, dtype=fp16)` 直接把 fp32 tensor 转 fp16 + pack，纯内存操作，~7s。

**SecAgg encode**（~55s）包含：
1. `extract_window_slices`：fp32 slice 提取
2. 逐 window `quantize_to_zq_with_feedback`：fp32→int16 量化 + stochastic rounding + residual 计算 + stats
3. 逐 window `apply_mask`：pairwise mask + self mask（mod q）
4. `pack_zq` + hex 编码
5. `update_block_memory_with_quant_residual`：residual 写回 memory

其中 **hex 编码是瓶颈**：`pack_zq` 输出 raw bytes，然后 `.hex()` 把每个 byte 变成 2 个 ASCII 字符，内存翻倍且 CPU 密集。269 个 window × 524288 元素 × 2 bytes = ~269 MB raw bytes → hex 后 ~538 MB 字符串。

### 4x.4 根因 3：upload 慢 7x（每轮多 ~41s）

fp16 upload ~7s，SecAgg upload ~48s。传输量接近（125 vs 135 MiB），差距来自：

1. **hex 编码导致传输量翻倍**：135 MiB raw bytes → 270 MiB hex 字符串（JSON body），实际网络传输量翻倍
2. **逐 window 串行上传**：269 个 window 逐个 POST，每个请求的 HTTP/JSON 开销累积
3. fp16 路径是批量打包一次上传，SecAgg 是逐 window 上传

### 4x.5 优化方向（落地状态，2026-09-18）

| 优化 | 解决什么 | 状态 |
|---|---|---|
| **raw bytes 上传（不走 hex）** | upload 慢 + 传输量翻倍 | ✅ 已落地（`z_key` + MinIO） |
| **单 blob + notify 不 GET** | 逐窗 PUT/GET × N | ✅ `202609181151` |
| **server 并行 unmask + 异步全量写盘** | `wait_agg` 含 1GB PUT | ✅ 同跑次；wait≈10s |
| **Hadamard + 全局 scale** | window 内动态范围 / per-window L∞ | ✅ 已落地；R20≈0.985 |
| **quant residual decay=1.0** | 量化误差被 0.9 吃掉 | ✅ Issue #1 |
| **异步 finalize** | self-master HTTP 超时 | ✅ 已落地 |
| **默认关 detailed-train-metrics** | train 被 profiler 拖到 ~300s | ✅ 默认关；正式跑 `train≈38s` |
| int24 / EF21 / Kashin | 进一步压缩或精度 | 后备，当前不需要 |

> §4x.1–4x.4 描述的是 **hex 时代** 的慢因分析，请勿当成当前瓶颈清单。当前剩余开销主要是 Hadamard+量化+mask 的 encode（≈15s vs fp16 ≈7s）。上传单 blob 已快于 fp16 多 block；`wait_agg`≈10s。无 SecAgg 稳态 ~88s 不是必须打平。

## 5. 代码实现

### 5.1 核心文件

| 文件 | 当前职责 |
|---|---|
| `experiments/shared/fixed_point.py` | 向量化 FWHT、量化+feedback、误差指标 |
| `experiments/shared/protocol.py` | `SecAggPlan`（hadamard / session / scale） |
| `experiments/server/secagg_coordinator.py` | global_amax→统一 scale；pad-p2 逆变换；聚合 |
| `experiments/shared/secagg_client.py` | Hadamard amax、mask、raw 上传辅助 |
| `experiments/shared/block_selection.py` | block_memory / quant_residual 拆分 |
| `experiments/server/aggregation_server.py` | blob 登记；后台 finalize；`done` 不挡全量写盘 |
| `experiments/shared/secagg_blob.py` | SAW1 打包/解包 |
| `experiments/run_s3r12v3_fsdp.py` | SecAgg 主路径；session 同步；`post_delta_MiB`；`--detailed-train-metrics` |
| `experiments/tests/test_secagg.py` | 量化 / mask / scale / blob 单测 |

### 5.2 配置（当前默认）

```yaml
# configs/s3r12v3-fsdp-secagg-verify.yaml
federated:
  num_rounds: 20                 # 可用 NUM_ROUNDS_ENV 覆盖
  coverage_h: 10
  memory_decay: 0.9              # 仅未选中 block
  quant_residual_decay: 1.0      # 量化残差不衰减
  block_size: 524288

security:
  secagg_enabled: true
  secagg_modulus_bits: 16
  secagg_scale: 0.0              # Hadamard 下用 global_amax 定全局 scale
  secagg_stochastic_rounding: true
  secagg_hadamard: true
```

### 5.3 单元测试

```bash
PYTHONPATH=experiments python experiments/tests/test_secagg.py
```

## 6. 后续（可选）

1. ~~Hadamard + raw bytes + Issue #1~~：**已完成**（`202609180941` 复跑确认）。
2. ~~时间效率（通用，不改量化）~~：**已结项**（5 轮 `202609181151` / 20 轮 `202609181406`），见 [completed：SecAgg 时间效率](../exec-plans/completed/2026-09-18-secagg-time-efficiency.md)。
3. **int24 / EF21 / Kashin / SCAFFOLD**：仅当要再压带宽或换更大模型时再评估。
4. 更强隐私：去掉 `global_amax` 或加密上报 train/eval loss（精度/工程权衡）。

## 7. 环境注意事项

运行 SecAgg 验证时需注意：

1. **端口**：SecAgg 验证常用 **8081**（`AGGREGATION_PORT_OVERRIDE=8081`）。
2. **干扰进程**：其他目录里的 watchdog / `pkill -f aggregation_server.py` 可能误杀；开跑前确认端口与 pid。
3. **远端代码同步**：改 client 后需同步到 `~/liuchao/fedscale-icc-1` 与 `~/liuchao/fedscale-icc-2`。
4. **观测开关**：正式对照勿开 `--detailed-train-metrics`（默认关）。
