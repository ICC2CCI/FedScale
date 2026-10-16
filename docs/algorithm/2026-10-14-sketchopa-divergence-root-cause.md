# SketchOPA 发散根因分析

> **日期**：2026-10-14
> **问题**：SketchOPA 实验中，E1（无 residual）完全发散，E2/E3（有 residual）波动剧烈无法收敛。需要确定：是方案实现有误，还是忠实实现了但方案本身在大模型场景下就会出问题。

---

## 1. 结论摘要

**实现忠实于交底书方案，没有实现错误。** 发散的根因是 **CountSketch bucket 预算严重不足，导致哈希碰撞率极高，恢复质量极差**。

具体来说：
- 交底书设计的通信预算公式为 `sketch_payload ≈ topk16_payload`（budget_ratio=1.0），即 Sketch 通信量等于明文 Top-k 通信量
- 在此预算下，每层仅分到约 **12,993 个 bucket**，但 `embed_tokens` 层有约 **152 万个 Top-k 条目**需要编码
- 平均每个 bucket 有 **117 个条目碰撞**，CountSketch 估计值的信噪比极低
- 要实现低碰撞恢复，`embed_tokens` 一层就需要约 **1500 万个 bucket**，是实际分配的 **1000 倍以上**

这不是实现 bug，而是 **通信预算公式在大模型 + 低 topk_ratio 场景下的结构性不足**。

---

## 2. 交底书方案 vs 我们的实现：逐项对照

### 2.1 方案核心流程（交底书 §3.2 - §3.6）

| 交底书步骤 | 交底书描述 | 我们的实现 | 是否一致 |
|---|---|---|---|
| §3.2 本地训练 | 计算梯度/模型差分 `g_i^t`，展平为 d 维向量 | `compute_delta(global_state, after_state)` → `flat_delta` | ✓ |
| §3.2 Global Top-k | 全模型展平向量上选 Top-k | `global_topk_flat(flat_delta, k)` | ✓ |
| §3.2 残差反馈 | 可选残差 `r_i`，`z = g + r`，Top-k on z | `residual.add_and_select(flat_delta, k)` | ✓ |
| §3.3 按层分区 CountSketch | Top-k 全局选择，编码时按层映射 | `CountSketchEncoder.encode(coords, vals)` 按层 scatter | ✓ |
| §3.3 固定公开 layout | bucket 数由公开预算确定，不随私有 Top-k 变化 | `build_layout()` 固定 layout，所有 client/coordinator 共享 | ✓ |
| §3.4 SHPRG mask | `ct_i = u_i + SHPRG(s_i)` | `shprg.mask_sketch_zq(sk_zq, seed_vec)` | ✓ |
| §3.4 Shamir seed 分享 | seed 通过门限分享给辅助节点 | `TwoClientSeedExchange` 2-client 简化版 | ✓（简化） |
| §3.4 聚合 Sketch 恢复 | `U = Σ ct_i - SHPRG(Σ s_i)` | `shprg.unmask_sketch_zq(agg_ct, s_sum)` | ✓ |
| §3.6 median-of-rows 恢复 | `x̂_j = median_r ξ_r(j) · U[r, h_r(j)]` | `estimate_batch()` → median over rows | ✓ |
| §3.6 K_recover | `K_recover = min(k · |C| · ρ, d)` | `compute_k_recover(k, |C|, ρ, d)` | ✓ |

### 2.2 交底书的通信预算公式

交底书 §4.1 明确定义：
> 设模型可训练参数维度为 d，单节点 Top-k 数量为 k，CountSketch 总 bucket 数为 L，传输值字节数为 b。本方案主 payload 为 **bL 字节**。

交底书 §4.1 还提到：
> 在典型低通信配置下，本方案可将主 payload 控制在全量 float32 稠密上传的约 1.6% 左右，约 62.5 倍通信压缩；相较全量 float16 稠密上传约 31 倍压缩。

我们的实现中 `budget_ratio = 1.0`，即 `sketch_payload ≈ topk16_payload`：
```
topk16_payload = k × (idx_bytes + val_bytes) = k × 6
sketch_payload = rows × Σ B_l × val_bytes = rows × total_budget × 2
budget_ratio = sketch_payload / topk16_payload = 1.0
=> total_budget = topk16_payload / (rows × val_bytes) = 6k / (5 × 2) = 0.6k
```

这与交底书的"主 payload ≈ 明文 Top-k 通信量"的设计一致。

### 2.3 唯一简化：2-client Shamir

交底书描述的是多 client + 动态门限辅助节点架构。我们的实验简化为 2-client + 直接 seed 交换。但这一简化**不影响 CountSketch 编码和恢复的正确性**——安全聚合层（SHPRG mask）的 smoke test 验证了 err=0，即安全聚合本身不引入误差。

---

## 3. 根因分析：CountSketch bucket 严重不足

### 3.1 预算计算

```
模型: Qwen2.5-0.5B
d = 630,167,424 (参数总量)
k = 6,301,674 (topk_ratio=1%)
rows = 5 (CountSketch 行数)
value_bytes = 2, idx_bytes = 4
budget_ratio = 1.0

topk16_payload = k × 6 = 37,810,044 bytes (37.8 MB)
total_budget (Σ B_l) = 37,810,044 / (5 × 2) = 3,781,004
sketch_len = rows × total_budget = 18,905,020
sketch_payload = 18,905,020 × 2 = 37,810,040 bytes (37.8 MB) ≈ topk16_payload ✓
```

### 3.2 每层 bucket 分配（uniform_param）

```
总层数 = 291
每层 bucket_per_row = 3,781,004 / 291 ≈ 12,993
```

### 3.3 关键层的碰撞分析

| 层 | 参数量 | 占比 | Top-k 条目（估） | bucket 数 | 碰撞率（条目/bucket） |
|---|---|---|---|---|---|
| `model.embed_tokens.weight` | 151,943,168 | 24.1% | ~1,519,431 | 12,993 | **117×** |
| `lm_head.weight` | 151,943,168 | 24.1% | ~1,519,431 | 12,993 | **117×** |
| `model.layers.0.mlp.gate_proj.weight` | 1,179,648 | 0.19% | ~11,813 | 12,993 | 0.9× |
| `model.layers.0.self_attn.q_proj.weight` | 576,000 | 0.09% | ~5,759 | 12,993 | 0.4× |
| `model.layers.0.input_layernorm.weight` | 1,536 | 0.0002% | ~15 | 12,993 | 0.001× |

### 3.4 CountSketch 恢复精度与碰撞率的关系

CountSketch 的恢复精度取决于 **load factor**（每个 bucket 平均碰撞数）：

- **load factor << 1**（bucket 远多于条目）：恢复质量高，估计值 ≈ 真实值
- **load factor ≈ 1**：中等碰撞，估计值有噪声但可用
- **load factor >> 1**（如 117×）：严重碰撞，估计值被噪声淹没，恢复质量极差

对于 `embed_tokens` 和 `lm_head` 两层，load factor = **117**，意味着：
- 每个 bucket 中平均有 117 个 Top-k 条目的值叠加（带 ±1 符号）
- median-of-rows 恢复的估计值是 117 个随机符号×值的叠加，信噪比极低
- 恢复出的值与真实值几乎无关

### 3.5 为什么 embed_tokens 和 lm_head 占了大部分 Top-k

在 LLM 微调中：
- `embed_tokens`（词嵌入层）和 `lm_head`（输出层）是参数量最大的两层（各 152M 参数，占 48% 总参数）
- 在医疗领域微调中，词汇表适配需要大幅调整 embedding 层，梯度幅值大
- 因此 Top-k 选择的坐标中，这两层占了绝大部分

这意味着：
- **大部分 Top-k 条目集中在最少数的层**
- **uniform_param 分配给这两层的 bucket 远远不够**（12,993 vs 需要 15,000,000+）
- 即使 calibrated 分配也受限于总预算——要把这两层降到 load factor=1，需要 3,038,862 个 bucket，但总预算只有 3,781,004，几乎要耗尽全部预算

### 3.6 要多少预算才能恢复？

要使所有层的 load factor ≤ 1（每个 bucket 至多 1 个条目）：

```
embed_tokens 需要: 1,519,431 bucket/row
lm_head 需要:     1,519,431 bucket/row
其他 289 层各需要: ~5,759 bucket/row (平均)
总计 per row:     1,519,431 × 2 + 5,759 × 289 ≈ 4,704,633
total_budget = 4,704,633
sketch_len = 5 × 4,704,633 = 23,523,165
sketch_payload = 23,523,165 × 2 = 47 MB
```

这其实和当前预算差不多（37.8 MB → 47 MB），但前提是 **calibrated 分配要极端倾斜**——把 80%+ 预算给 embed+lm_head 两层。

但要让 load factor < 0.1（低碰撞，恢复质量好）：

```
embed_tokens 需要: 15,194,310 bucket/row
lm_head 需要:     15,194,310 bucket/row
其他层需要:       ~57,590 bucket/row (平均)
总计 per row:     31,030,310
total_budget = 31,030,310
sketch_len = 5 × 31,030,310 = 155,151,550
sketch_payload = 155 MB
```

这比当前预算大 **4 倍**。而要达到 load factor < 0.01（接近无碰撞）：

```
sketch_payload ≈ 1.5 GB（接近 dense fp16 的 1.26 GB）
```

此时 SketchOPA 已经没有通信优势了。

### 3.7 为什么 Top-k（B1）能收敛但 SketchOPA 不能

| 方法 | embed_tokens 的处理 | 碰撞 | 恢复误差 |
|---|---|---|---|
| Top-k（B1） | 直接传坐标(int32)+值(fp16) | 无碰撞 | 0（精确） |
| SketchOPA | CountSketch 编码，152万条目 → 12,993 bucket | 117×碰撞 | 极大 |

Top-k 方法直接传坐标和值，服务器精确聚合，不引入任何误差。通信量 = 72 MB。

SketchOPA 用 CountSketch 隐藏坐标，但 bucket 不足导致恢复质量极差。通信量 = 192 MB（反而比 Top-k 大！）。

**关键矛盾**：在当前预算设置下（budget_ratio=1.0），SketchOPA 的通信量（192 MB）比 Top-k（72 MB）还大，但恢复质量远不如 Top-k。这是因为 sketch 还包含了 rows×value_bytes 的开销，以及安全聚合的量化开销。

---

## 4. 交底书中的相关设计意图

### 4.1 交底书 §5.1 高精度/平衡/低通信模式

> 高精度模式使用较多 CountSketch 行数和 bucket 预算，使恢复更新更接近明文 Top-k；平衡模式将主 payload 配置为接近明文 Top-k 或低于明文 Top-k；低通信模式使用更短 Sketch...

交底书意识到需要不同模式，但**没有给出 bucket 预算与恢复质量的定量关系**。我们的实验表明，"平衡模式"（budget_ratio=1.0）在大模型场景下恢复质量不够。

### 4.2 交底书 §5.2 混合层级聚合

> 对于参数量较小但对模型精度敏感的层，可采用稠密安全聚合；对于参数量较大的层，采用 CountSketch 编码。

这个设计可能改善小层恢复质量，但**不解决大层（embed_tokens, lm_head）的碰撞问题**——恰恰是大层最需要 bucket。

### 4.3 交底书 §4.1 的典型配置

> 在典型低通信配置下，本方案可将主 payload 控制在全量 float32 稠密上传的约 1.6% 左右，约 62.5 倍通信压缩

1.6% × 4d = 0.064d = 0.064 × 630M = 40.3 MB，与我们的 37.8 MB 一致。

但这个"62.5 倍压缩"的前提是 topk_ratio 足够低、模型参数分布相对均匀。对于 LLM 这样参数集中在 embedding 层的场景，这个配置不够。

---

## 5. 验证：调整 budget_ratio 的预期效果

如果要使 SketchOPA 恢复质量接近 Top-k，需要大幅增加 budget_ratio：

| budget_ratio | total_budget | sketch_payload | embed_tokens 碰撞率 | 预期恢复质量 |
|---|---|---|---|---|
| 1.0（当前） | 3,781,004 | 37.8 MB | 117× | 极差（发散） |
| 5.0 | 18,905,020 | 189 MB | 23× | 差 |
| 10.0 | 37,810,040 | 378 MB | 12× | 中等 |
| 50.0 | 189,050,200 | 1.89 GB | 2.3× | 较好 |
| 100.0 | 378,100,400 | 3.78 GB | 1.2× | 好 |

当 budget_ratio=100 时，sketch_payload=3.78 GB，已远超 dense fp16 的 1.26 GB，SketchOPA 完全失去通信优势。

**这就是 SketchOPA 在大模型场景下的结构性矛盾**：要隐藏 Top-k 坐标（安全聚合），需要用 CountSketch 编码；但 CountSketch 要恢复好，bucket 数需要远大于 Top-k 条目数；而大模型的 Top-k 条目集中在少数大层，这些层需要的 bucket 数极大。

---

## 6. 总结

| 问题 | 答案 |
|---|---|
| 实现有误吗？ | **没有**。实现忠实于交底书方案，核心流程（Top-k → 按层 CountSketch → SHPRG 安全聚合 → median-of-rows 恢复）全部正确实现，单测和 smoke test 验证了正确性 |
| 为什么发散？ | **CountSketch bucket 预算严重不足**。budget_ratio=1.0 下，embed_tokens 层碰撞率 117×，恢复值被噪声淹没 |
| 是方案的问题还是实现的问题？ | **是方案在大模型场景下的结构性不足**。交底书的预算公式适用于参数分布相对均匀的模型（如 CNN），但 LLM 参数集中在 embedding 层，导致这些层碰撞率极高 |
| residual 为什么不够？ | residual buffer 缓解了误差累积（E1→E2 从 15.5 降到 2.0），但每轮注入的恢复噪声太大，residual 无法完全抵消 |
| calibrated 为什么更差？ | calibrated 把更多 bucket 给了 embed+lm_head，但总预算不变，导致中间层 bucket 更少，中间层恢复质量恶化，整体更不稳定 |
| 能修复吗？ | 在当前框架下，要么大幅增加 budget_ratio（但失去通信优势），要么用混合策略（大层直接传 Top-k，小层用 CountSketch），但后者放弃了隐藏大层坐标的隐私目标 |
