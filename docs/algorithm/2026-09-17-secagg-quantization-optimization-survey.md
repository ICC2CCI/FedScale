# SecAgg 量化优化：文献调研与技术方案

> 日期：2026-09-17
> 状态：**代码已落地 + 5 轮验证通过**（Issue #1 + 向量化 Hadamard + 全局 scale 去 L∞ + raw bytes）。R5 eval=1.364，优于旧 SecAgg（1.432）且略优于 fp16 R5（1.377）。20 轮对照进行中。
> 关联：
> - [ICC2CCI/FedScale#1](https://github.com/ICC2CCI/FedScale/issues/1)
> - `docs/algorithm/2026-09-15-secagg-quantization-precision-issue.md`

## 0. 实现对照（相对 opencode 半成品）

先前 Hadamard 已接到协议字段和量化前后变换，但有三处不能直接用于训练：

| 项 | 当时状态 | 本次 |
|---|---|---|
| FWHT | Python 双层循环，n=524288 时每窗口数十秒 | 向量化，`log2(n)` 次张量运算 |
| scale / amax | 用原始域 per-window amax | Hadamard：**只报 1 个 global_amax**，所有 window 同一 scale |
| raw bytes | 仍 `pack_zq(...).hex()` 走 JSON | MinIO raw bytes + `z_key` 通知；hex 仅兼容旧客户端 |
| Issue #1 | 未做 | FP32 delta / 拆分 residual decay / 量化误差指标 |
| 训练稳定性 | 空 session 导致 self-mask 不抵消；非 2 的幂 window 不旋转 | session 从 peer-keys 同步；Hadamard pad 到 2 的幂 |
| finalize 超时 | 最后一个 self-master HTTP 同步聚合 ~31s | 后台 finalize，client 超时 180s |

## 1. 问题回顾

当前 SecAgg 方案（per-window 当轮 amax + error-feedback + stochastic rounding）解决了 eval loss 发散问题（1.50→2.36 变为 1.50→1.33），但与 fp16 基线（R20 eval=0.987）仍有三个差距：

1. **eval loss 更大**（差 35%）：int16 均匀量化对 window 内动态范围大的 delta，小值精度差；叠加实现层额外 FP16 rounding 与 residual×0.9
2. **时间更长**（慢 50%）：hex 编码导致传输量翻倍 + 逐 window 串行上传
3. **仍有泄露**：per-window L∞（269 个标量/轮）泄露给 server

**不能把剩余收敛差距直接归因于 INT16 位宽。** 应先去掉实现层额外误差（Issue #1），再加上 Hadamard 压低动态范围，再和 fp16 基线对照。

### 1.1 Issue #1：量化前多余的精度损失

见 [ICC2CCI/FedScale#1](https://github.com/ICC2CCI/FedScale/issues/1)。两条实现错误会叠加：

1. **过早 FP16 rounding**：`sub_state` / `get_sharded_block_delta` 在 FP32 算出 delta 后又 `.to(fp16).to(fp32)`，`delta+memory` 同样先 round 再相加。丢失的尾数无法恢复。
2. **量化残差被 `memory_decay=0.9` 衰减**：标准 error-feedback 应保存 `e = x - D(Q(x))`；乘 0.9 等于每轮主动丢掉 10% 量化误差。block-mask 未发送的 stale update 可以衰减，量化误差不行。

本次数据路径：

```
FP16 模型参数
  → FP32 delta（不 round 回 FP16）
  → FP32 (block_memory + quant_residual)
  → 可选 Hadamard（旋转域 amax → scale）
  → INT16 量化 + stochastic rounding
  → MinIO raw bytes（INT16 packed）
  → SecAgg mask 在整数模域精确抵消
```

Memory 拆分：

| 状态 | 选中 block | 未选中 block | dtype |
|---|---|---|---|
| `block_memory` | 置 0 | `(delta + block_memory) * 0.9` | FP32 |
| `quant_residual` | `x - D(Q(x))`（decay=1.0） | 保留上一轮 residual | FP32 稀疏 |

量化输入：`x_t = Δ_t^{FP32} + m_t^{block} + e_t^{quant}`。

## 2. 文献调研

### 2.1 最直接相关：Google SecAgg 团队的 Hadamard 方案

**论文**：Bonawitz et al., *"Federated Learning with Autotuned Communication-Efficient Secure Aggregation"*, arXiv:1912.00131 (2019)

- **核心**：量化前对 model update 做随机 Hadamard 旋转，旋转后元素幅值趋于均匀（测度集中），一个固定 scale 的均匀量化器即可覆盖，无需 per-coordinate scaling
- **与我们的关系**：这就是 Google SecAgg 团队自己为「SecAgg 整数域求和 + 低 bit 量化」设计的方案，架构和我们一模一样，但多了旋转
- **Bit-width**：8-16 bit，SecAgg 模运算兼容
- **动态范围**：Hadamard 旋转 → ℓ∞/ℓ2 比从 O(1) 降到 O(√(log d)/√d)，512K 维向量动态范围从 100x 压到 ~3x

### 2.2 QSGD — 量化 SGD 的奠基工作

**论文**：Alistarh et al., *"QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding"*, arXiv:1610.02132 (NeurIPS 2017)

- **核心**：per-coordinate stochastic uniform quantization，用 ‖v‖₂ 归一化后量化
- **方案**：均匀 + stochastic rounding，传输 scale + quantized indices
- **局限**：对稀疏/高动态范围梯度方差大（后续工作用 Hadamard 修复）
- **启示**：「传输 scale，量化 shape」是标准模式

### 2.3 Error Feedback — 让有偏量化器收敛

**论文**：Karimireddy et al., *"Error Feedback Fixes SignSGD and other Gradient Compression Schemes"*, arXiv:1901.09847 (ICML 2019)

- **核心**：residual `e_t = g_t - Q(g_t + e_{t-1})`，量化器作用在 `g_t + e_{t-1}` 上，补偿有偏压缩器的偏差
- **关键结论**：EF 让任何 contractive 压缩器（sign、Top-k、uniform）达到 SGD 收敛速率 O(1/√T)
- **我们已实现**：`block_memory` decay=0.9（未选中块）；`quant_residual` decay=1.0（量化误差完整保留，Issue #1）

### 2.4 EF21 — 现代 Error Feedback

**论文**：Richtárik et al., *"EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback"*, arXiv:2106.05203 (NeurIPS 2021)

- **核心**：不量化原始梯度，而是量化「与 running memory 的差」`g_t - m_{t-1}`，memory 更新为 `m_t = Q(g_t - m_{t-1}) + m_{t-1}`
- **优势**：输入到量化器的信号始终接近 0，动态范围天然小，无需额外旋转
- **与 Hadamard 的关系**：EF21 + Hadamard 在 EF21+Bells&Whistles (2110.03294) 中组合使用
- **启示**：如果 Hadamard 仍不够，可进一步改为 EF21 偏差量化

### 2.5 RATQ — 旋转 + 自适应量化（理论最优）

**论文**：Mayekar & Tyagi, *"RATQ: Rotated Adaptive Tetra-iterated Quantizer"*, arXiv:1908.08200

- **核心**：Hadamard 旋转 + 自适应动态范围 uniform quantization
- **理论**：证明了「旋转 + 均匀量化」在信息论意义上接近最优
- **启示**：旋转后均匀量化是理论上正确的选择，不需要非均匀量化

### 2.6 ScionFL — Kashin 表示（SOTA）

**论文**：Ben-Itzhak et al., *"ScionFL: Efficient and Robust Secure Quantized Aggregation"*, arXiv:2210.07376 (SaTML 2024)

- **核心**：MPC-based SecAgg + 1-bit 线性量化 + **Kashin's representation**
- **Kashin vs Hadamard**：Kashin 给出更强的 ℓ∞ bound——O(1/√d) vs Hadamard 的 O(√(log d)/√d)，同 bit-width 下精度更高
- **启示**：如果 Hadamard + int16 不够，Kashin 是下一步

### 2.7 SCAFFOLD — 修正 Client Drift

**论文**：Karimireddy et al., *"SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"*, arXiv:1910.06378 (ICML 2020)

- **核心**：control variates `c_i` 修正 FedAvg 的 client drift
- **与量化的关系**：2-client 非 IID 场景下 FedAvg drift 可能严重，且 drift 和量化误差难以区分。SCAFFOLD 的修正项天然小幅度，量化友好
- **启示**：如果 eval loss 差距部分来自 client drift 而非纯量化，SCAFFOLD 能进一步改善

### 2.8 低 bit 训练的关键发现

**MXFP4 预训练研究** (arXiv:2605.09825)：
- **权重梯度（Wgrad）对量化最敏感**
- 随机 Hadamard 在 Wgrad 上不够稳定，**确定性 Hadamard** 才能恢复稳定性
- 启示：我们的 delta 是权重更新（≈ Wgrad 累积），应用确定性 Hadamard

**FP8 格式** (arXiv:2209.05433)：
- 即使 8-bit，业界也用两种格式（E4M3 forward + E5M2 gradient），因为 forward 和 gradient 的动态范围不同
- **loss scaling** 是标准技巧：量化前乘大常数扩展小值范围，反量化后除回

### 2.9 非均匀量化为什么不适合 SecAgg

文献共识：**不使用 Laplace/Normal 匹配的非均匀量化器**，原因：
1. Hadamard 旋转后坐标趋于高斯，均匀量化器 + stochastic rounding 在 ℓ2 意义下近最优（RATQ 证明）
2. 非均匀量化器需要传输非均匀 levels，**破坏 SecAgg 整数域求和**
3. Kashin 表示比非均匀量化更优雅地解决同一问题

## 3. 技术方案设计

### 3.1 方案概述

基于文献共识，最优方案组合为：

```
Hadamard 旋转 + 均匀量化（int16）+ stochastic rounding + error feedback
```

在当前已实现的 per-window scale + error-feedback + stochastic rounding 基础上，**新增 Hadamard 旋转**，并做工程优化（raw bytes 上传）。

### 3.2 Hadamard 旋转

#### 3.2.1 数学原理

```
量化前:  x = [1e-3, 1e-6, 1e-7, 2e-4, ...]   ← 动态范围 100x
旋转后:  y = H @ (D ⊙ x)                      ← 动态范围 ~3x
         y_i ≈ ||x|| / sqrt(n)，几乎所有元素

其中:
  D = diag(±1)  随机符号矩阵（每轮重新生成，所有 client 共享种子）
  H = Walsh-Hadamard 矩阵（确定性，H @ H^T = n @ I）
```

测度集中（concentration of measure）：512K 维向量旋转后，`P(|y_i| > t ||x||/sqrt(n)) < exp(-t²/2)`，99.99% 的元素在 `3 ||x||/sqrt(n)` 以内。

#### 3.2.2 流程

```
Client 端:
  1. delta_slice (fp32, n=524288)
  2. 生成随机符号 D（种子 = round_seed + window_id，所有 client 共享）
  3. y = FWHT(D ⊙ delta_slice)   ← Fast Walsh-Hadamard Transform, O(n log n)
  4. 上报 global_amax = max_b max|y_b|（1 个 float，不是 269 个 per-window L∞）
  5. server 取 max_k(global_amax) → 所有 window 共用同一 scale
  6. q = quantize(y, global_scale)
  7. z_k = q + mask (mod q)
  8. 上传 z_k raw bytes

非 Hadamard 路径仍上报 per-window amax（精度优先，泄露 L∞）。

Server 端:
  1. sum_z = Σ z_k (mod q)         ← 整数域求和（不变）
  2. sum_q = unmask(sum_z)         ← 移除 mask（不变）
  3. y_agg = dequant(sum_q, scale) / N
  4. agg_delta = D ⊙ FWHT(y_agg)   ← 逆变换（H^(-1) = H/n, D^(-1) = D）
  5. 写 agg_block_key
```

#### 3.2.3 关键约束

| 约束 | 说明 |
|---|---|
| 所有 client 用同一个 D | 整数域求和的前提：`sum(H*D*x_k) = H*D*sum(x_k)`。D 的种子必须对所有 client 一致 |
| D 的种子来源 | `round_seed + window_id`，server 在 plan 里下发 round_seed，所有 client 用相同种子生成 D |
| H 是确定性的 | Walsh-Hadamard 矩阵不需要随机化，D 提供随机性 |
| 逆变换 | `H^(-1) = H/n`（Hadamard 矩阵性质），`D^(-1) = D`（D 是 ±1 对角矩阵）|
| GPU 实现 | FWHT 在 GPU 上 O(n log n)，512K 元素 <0.1s |

#### 3.2.4 对三个问题的影响

| 问题 | 解决方式 | 预期效果 |
|---|---|---|
| eval loss 更大 | 旋转后动态范围 ~3x，int16 小值相对误差从 10% → 0.5% | eval 接近 fp16 |
| 时间更长 | Hadamard <0.1s 可忽略；时间问题靠 raw bytes 解决 | encode/upload 快 5-8x |
| 泄露 | 旋转后只上报 1 个 global_amax，所有 window 共用 scale | **已落地**：不再上报 269 个 per-window L∞ |

#### 3.2.5 随机种子的安全分析

D 的种子由 server 下发（`round_seed`），所有 client 用相同种子。server 知道 D，但：
- server 看到的是 `H*D*x_k + mask`，即使知道 D 和 H，也无法分离 x_k（因为 mask 保护）
- 逆变换需要 sum_q（移除 mask 后的聚合值），server 只能对聚合结果做逆变换，不能对单 client 做
- **安全性不变**：server 仍只能看到聚合后的 delta，看不到单 client 的更新

### 3.3 Raw Bytes 上传

#### 3.3.1 当前问题

```python
z_hex = pack_zq(z_k, modulus_bits).hex()  # 135 MiB raw → 270 MiB hex string
body = {"z_hex": z_hex, ...}              # JSON body，传输量翻倍
```

#### 3.3.2 已落地

```python
z_bytes = pack_zq(z_k, modulus_bits)
key = upload_secagg_window_key(round, client, window)
minio.put_bytes(key, z_bytes)             # 135 MiB raw，不走 hex
POST {"z_key": key, "vector_len": n, ...} # 控制面只传对象键
# server: minio.get_bytes(z_key) → unpack_zq
```

旧客户端仍可 POST `z_hex`，server 两边都认。

#### 3.3.3 预期效果

| 指标 | hex JSON | raw bytes |
|---|---|---|
| 传输量 | 270 MiB（hex string） | 135 MiB（raw bytes） |
| encode 耗时 | ~55s（hex 编码 CPU 密集） | pack + MinIO put |
| upload 耗时 | ~48s（传输翻倍量经 aggregation HTTP） | 对象存储直传 |

### 3.4 实现计划与落地

```
Phase 1: Hadamard 旋转     ✅ 向量化 FWHT + **全局 scale（只报 global_amax）** + pad-to-p2
Phase 1b: Issue #1         ✅ FP32 delta / 独立 residual decay / rel_L2·cosine·SQNR
Phase 2: Raw Bytes 上传    ✅ MinIO z_key；hex 兼容保留
Phase 2b: 生产修复         ✅ session 同步；self-master 后台 finalize
Phase 3a: 5 轮验证         ✅ 202609171717，R5 eval=1.364
Phase 3b: 20 轮对照        ⏳ 进行中
Phase 4（后备）            未做：EF21 / Kashin / int24
```

关键文件：

| 文件 | 改动 |
|---|---|
| `experiments/shared/fixed_point.py` | 向量化 `fwht`、量化误差指标 |
| `experiments/shared/state_dict_utils.py` | `sub_state/add_state(keep_fp32=)`；SCALE-3 去掉 FP16 往返 |
| `experiments/shared/block_selection.py` | `quant_decay=1.0`；`update_block_memory_from_states` + `merge_quant_residual_memory` |
| `experiments/shared/secagg_client.py` | `extract_window_to_send_fp32`；Hadamard amax |
| `experiments/run_s3r12v3_fsdp.py` | SecAgg 主路径走 FP32 + raw bytes；peer-keys 同步 session；self-master timeout=180s |
| `experiments/server/aggregation_server.py` | `z_key` 读 MinIO raw bytes；后台 finalize |
| `experiments/server/secagg_coordinator.py` | pad-to-p2 逆变换；session 写入 key-announce / peer-keys |
| `configs/s3r12v3-fsdp-secagg-verify.yaml` | `quant_residual_decay: 1.0`，`secagg_hadamard: true` |

开关：

```yaml
federated:
  memory_decay: 0.9            # 仅 block-mask residual
  quant_residual_decay: 1.0    # INT16 量化残差不衰减
security:
  secagg_hadamard: true
```

单元测试：`PYTHONPATH=experiments python experiments/tests/test_secagg.py`

### 3.5 建议 Ablation（与 Issue #1 对齐）

保持 Block Mask / 种子 / 本地步数 / 学习率 / 数据 / INT16 / window scale 不变：

| ID | 设置 |
|---|---|
| A | 旧实现：中间 FP16 + quant residual decay=0.9 |
| B | FP32 delta/to_send，quant decay=0.9 |
| C | FP32 + quant decay=1.0（当前默认，无 Hadamard） |
| D | C + block/quant residual 分离（当前默认） |
| E | D + Hadamard + raw bytes（当前 yaml） |
| fp16 | 非 SecAgg 基线 |

对比：train/eval loss、rel L2、cosine、SQNR、window amax/RMS、encode/upload 耗时。

### 3.6 5 轮验证（2026-09-17, `results/202609171717`）

Hadamard + 全局 scale + Issue #1 + raw bytes。端口 8081。

| 轮次 | train | eval |
|---|---|---|
| R1 | 2.041 | 1.498 |
| R2 | 1.592 | 1.427 |
| R3 | 1.474 | 1.398 |
| R4 | 1.462 | 1.394 |
| **R5** | **1.420** | **1.364** |

对照：旧 SecAgg（per-window amax，无 Hadamard）R5 eval=1.432；fp16 基线 R5 eval≈1.377。全程 `clip_windows=0`，`self_master` 0.01s（后台 finalize），session 两端一致。R2 未再发散。

曾踩过的坑（已修，勿回退）：

1. **空 session**：client 本地 `SecAggPlan` 的 `secagg_session_id` 为空，pairwise mask 仍能互消，但 server 用 hashed session 重生 self-mask，噪声经 Hadamard 全局 scale 放大后 LayerNorm 等非 p2 window 被 clip。必须从 peer-keys 同步 session，空 session 拒绝 mask。
2. **非 2 的幂 window**：LayerNorm 896 等原先跳过旋转，却和 p2 window 共用全局 scale。必须 pad 到下一个 2 的幂再 FWHT。
3. **self-master 超时**：最后一名 client 的 HTTP 里同步 unmask+写 MinIO ≈31s，超过 30s timeout。改为先回 200，后台 finalize。

## 4. 参考文献

| # | 论文 | arXiv | 核心贡献 |
|---|---|---|---|
| 1 | Bonawitz et al., Autotuned SecAgg with Random Rotation | 1912.00131 | SecAgg + Hadamard 的奠基工作 |
| 2 | Alistarh et al., QSGD | 1610.02132 | 量化 SGD 奠基 |
| 3 | Karimireddy et al., EF Fixes SignSGD | 1901.09847 | Error feedback 理论 |
| 4 | Richtárik et al., EF21 | 2106.05203 | 现代 EF，compress deviation |
| 5 | Mayekar & Tyagi, RATQ | 1908.08200 | 旋转 + 均匀量化理论最优 |
| 6 | Ben-Itzhak et al., ScionFL | 2210.07376 | Kashin 表示，SOTA |
| 7 | Karimireddy et al., SCAFFOLD | 1910.06378 | Client drift 修正 |
| 8 | Stollenwerk & Jacques, FO-SGD | 2405.11095 | Hadamard + 1-bit 训练 |
| 9 | Kim & Park, HLQ | 2406.15102 | Hadamard 4-bit 训练 |
| 10 | Micikevicius et al., FP8 | 2209.05433 | Loss scaling + 双格式 |
| 11 | Stich & Karimireddy, EF Framework | 1909.05350 | EF 统一框架 |
| 12 | MXFP4 Pretraining | 2605.09825 | Wgrad 需确定性 Hadamard |
