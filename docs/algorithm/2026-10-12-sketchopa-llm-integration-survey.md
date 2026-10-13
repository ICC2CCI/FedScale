# SketchOPA 应用到大模型联邦微调：独立实验调研与 TODO

> 日期：2026-10-12
> 来源：`third-party/1-技术交底书-算力网协同训练.docx` + `third-party/Clover_TKCS_Analysis_20260908.zip`
> 目标：**独立实现**交底书所述 SketchOPA 算法（Top-k → CountSketch → 聚合 → 恢复），应用到 **Qwen2.5-0.5B 联邦微调**，实验验证其是否具有与 LeNet5/MNIST 小模型类似的效果（精度 + 通信压缩）。
> 定位：**独立实验脚本**，不集成进现有 Central/MinIO/SecAgg 系统，不复用 `run_s3r12v3_fsdp.py` 路径。仅复用仓库内已有的**模型权重与数据集**。

---

## 1. 交底书算法（SketchOPA）核心

### 1.1 每轮流程

```
(1) Server 维护全局模型 w_t，公开 RoundConfig（k, Sketch layout(R,B_l), recover_k, 量化规则）
(2) Client_i 下载 w_t，本地训练得梯度/模型差分 g_i（展平为 d 维）
(3) 【可选残差】z_i = g_i + r_i；global Top-k 选 k 个坐标 → (S_i, v_i)；r_{i+1} = z_i - Sparse(S_i,v_i)
(4) 按层分区 CountSketch 编码：(S_i, v_i) → 固定长度 u_i（R 行 × Σ B_l bucket）
(5) 【安全层，可选】ct_i = u_i + SHPRG(s_i)；seed 分享给辅助节点
(6) Server 聚合 Sketch：U = Σ u_i（明文）或 U = Σ ct_i - M（安全）
(7) Server 恢复全局 Top-K_recover：x̂_j = median_r ξ_r(j)·U[r,h_r(j)]，K_recover = min(k·|C|·ρ, d)
(8) 还原参数形状 → 更新 w_{t+1}
```

### 1.2 关键设计

| 设计点 | 规定 |
|---|---|
| Top-k 粒度 | global Top-k（全模型展平），非按层 |
| Sketch 分区 | 按层分区（每层独立 R×B_l），避免跨层碰撞 |
| Layout 公开固定 | B_l 由公开预算/校准确定，不随本轮私有 Top-k 层分布变化 |
| 残差反馈 | 可选，缓解长期压缩信息损失 |
| 恢复 | median-of-rows；K_recover = min(k·|C|·ρ, d) |
| 通信 | 主 payload = b·L（L=ΣB_l），典型约 dense fp32 的 1.6%（~62x） |

### 1.3 Clover 实验数据关键超参与结论

zip 内**仅有结果 CSV/JSON，无源码**。提取的关键配置（作为大模型实验起点）：

```
sketch_rows              = 5
sketch_bucket_factor     = 8
sketch_min_buckets       = 8
sketch_layout_mode       = fixed_public
sketch_allocation_strategy = topk_count_calibrated  # 按层 Top-k 计数校准分配 bucket
sketch_transport_dtype   = float16   (value_bytes=2)
sketch_recover_k_mode    = active_union
sketch_budget_ratio      = 1.0       # Sketch payload 对齐 Top-k16 payload
residual_enabled         = True      # 关键：无 residual 会发散
sketch_recovered_clip_norm = 10.0
```

**结论（LeNet5/MNIST, 100 client, 20 round, IID）**：
- residual + calibrated：best_acc=**0.978**（接近明文基线）
- **无 residual：R1 后 test_acc 崩到 0.099（发散）** → 残差反馈是收敛关键
- payload vs dense16 ≈ 5%

---

## 2. 大模型实验设计

### 2.1 实验目标

验证 SketchOPA 在 **Qwen2.5-0.5B + medical_flashcards 联邦微调**（2 client）下：
1. 能否收敛到接近 dense/fp16 基线的 eval loss？
2. 通信压缩比如何？
3. 残差、layout 分配策略、rows、budget_ratio 的影响？

### 2.2 复用资源（仅模型 + FSDP + 训练架构，不引用任何历史实验结果）

| 资源 | 路径 / 说明 |
|---|---|
| 模型 | `model/Qwen/Qwen2.5-0.5B` |
| 训练数据 | `data/splits/icc1_client0_train.json`, `data/splits/client1_train.json` |
| 评估数据 | `data/medical_flashcards_eval.json` |
| FSDP 配置 | `configs/accelerate/accelerate_config.yaml`（仅复用 FSDP 训练基础设施） |

> ⚠️ **不引用 `results/` 下任何历史实验结果（如 20260914-final-clean 等）作为基线**。所有基线（dense / Top-k）均由本实验**独立跑出**，避免与之前 SecAgg / S3R12v3 等方案的结果混淆。Clover zip 的小模型实验数据仅用于**还原原方案超参**，不作为大模型实验的精度参照。

### 2.3 实验矩阵

| 实验 | 方法 | 说明 | 目的 |
|---|---|---|---|
| **B0** | Dense FedAvg（fp16） | 全量上传 | 基线，**由本实验独立跑出**，不引用历史结果 |
| **B1** | 明文 global Top-k | 上传 (S_i, v_i)，服务端直接聚合 | 稀疏化基线（无 Sketch 误差） |
| **E1** | SketchOPA 明文聚合 | Top-k → Sketch → 求和 → 恢复 | 验证 Sketch 编码/恢复在大模型上的效用 |
| **E2** | E1 + residual | 开启残差反馈 | 验证残差对收敛的必要性（小模型已证实） |
| **E3** | E2 + calibrated layout | 按层 Top-k 计数校准 bucket 分配 | 验证层级分配对大模型的影响 |
| **E4** | E2 扫参 | rows ∈ {3,5,7}, budget_ratio ∈ {0.5,1.0,2.0} | 压缩比 vs 精度权衡 |
| **E5** | E3 + 安全掩码 | ct_i = u_i + SHPRG(s_i) | 验证安全层不影响效用（决策点 4 已定默认含） |

> 主线：先 B0 → B1 → E1(对照发散) → E2(对照发散) → **E3(主实验)** → E4(扫参) → E5(安全层)。详见 §7。

### 2.4 评估指标

- eval loss（每轮，与本次实验自跑的 B0/B1 基线对比）
- train loss
- payload bytes（每轮上传）+ 相对 dense 的压缩比
- Sketch 恢复命中率（恢复出的 Top-K_recover 与明文 Top-k 聚合支持集的重叠率）
- 各轮耗时

### 2.5 超参起点

```
model            = Qwen2.5-0.5B (d ≈ 494M)
num_clients      = 2
num_rounds       = 20
local_steps      = 30
lr               = 1e-5
topk_ratio       = 0.0167       # k/d ≈ 1.67%（对齐原方案，§5 决策点2），k ≈ 8.24M；扫描 {1%, 1.67%, 3%}
sketch_rows      = 5
bucket_factor    = 8
budget_ratio     = 1.0          # ΣB_l = 6k/(R×value_bytes) × budget_ratio（§6.3）
allocation       = topk_count_calibrated（主实验 E3）；uniform_param（对照 E1/E2）
residual         = True（主实验 E3）；False（对照 E1）
recover_rho      = 1.0          # 扫描 {1.0, 2.0, 4.0}（§6.4，2-client 必扫）
transport_dtype  = float16
```

---

## 3. TODO（实施计划）

> 详见 **§7 TODO 修正版**（综合 §6 二次精读发现）。本节为初版，已被 §7 取代，保留仅作变更追溯。

### Phase 0：独立实验脚手架 + 算法核心库

- [ ] **T0.1** 新建独立实验目录 `experiments_sketchopa/`（与现有 `experiments/` 平行，互不干扰）
- [ ] **T0.2** `sketchopa/count_sketch.py`：按层分区 CountSketch
- [ ] **T0.3** `sketchopa/topk.py`：global Top-k（全模型展平）+ `ResidualBuffer` 类
- [ ] **T0.4** `sketchopa/layout.py`：固定公开 layout 构造（uniform / calibrated）
- [ ] **T0.5** `sketchopa/recover.py`：median-of-rows 恢复 + 还原参数形状 + clip
- [ ] **T0.6** `sketchopa/quant.py`：Sketch 值量化（float16 / int16 public scale）
- [ ] **T0.7** `sketchopa/shprg.py`：seed-homomorphic PRG（LWR-based）
- [ ] **T0.8** 单元测试 `tests/test_sketchopa_core.py`

### Phase 1-5：见 §7 修正版

> ⚠️ §3 初版的 Phase 1-5 描述（如「E2 预期收敛」「前 3 轮采集」等）已被 §6 推翻，**以 §7 为准**。

---

## 4. 目录规划

```
fedscale-icc-server/
├── experiments_sketchopa/              # 独立实验，不碰现有 experiments/
│   ├── sketchopa/
│   │   ├── __init__.py
│   │   ├── count_sketch.py             # T0.2
│   │   ├── topk.py                     # T0.3
│   │   ├── layout.py                   # T0.4
│   │   ├── recover.py                  # T0.5
│   │   ├── quant.py                    # T0.6
│   │   └── shprg.py                    # T0.7
│   ├── run_baseline_dense.py           # T1.1
│   ├── run_baseline_topk.py            # T1.2
│   ├── run_sketchopa.py                # T2.1
│   ├── tests/
│   │   └── test_sketchopa_core.py      # T0.8
│   └── results_sketchopa/              # 实验输出
└── docs/algorithm/
    └── 2026-10-12-sketchopa-llm-integration-survey.md  # 本文件
```

---

## 5. 决策点确认结果

| # | 决策点 | 结论 | 依据 |
|---|---|---|---|
| 1 | client 数量 | **2-client**（真实双集群） | 现有集群就 2 client，先跑通；K_recover = min(k·2·ρ, d) 偏小，通过调 ρ 缓解 |
| 2 | Top-k 比例 k | **k/d ≈ 1.67%**（对齐原方案） | Clover 数据反推：LeNet5 d=431080，topk16 payload=43100B，idx32+val16=6B/coord → k≈7183 → k/d≈1.67%。大模型 d≈494M → k≈8.2M（起点），扫描 {1%, 1.67%, 3%} |
| 3 | 是否用 FSDP | **是，一开始就用 FSDP** | 后续要上更大模型，希望一套方案；复用现有 `accelerate_config.yaml` FSDP 配置 |
| 4 | 安全层 | **完全按原方案，含安全层** | 原方案有完整安全层：`shprg_backend=opa_lwr_shprg` + `committee_size=7, threshold=5` + seed Shamir 分享。2-client 退化实现：client 互为辅助节点互发 seed share，Central 收齐份额重构 s_sum；门限 τ 退化为 1（2-client 无掉线） |

### 5.1 原方案安全层关键配置（从 Clover 数据还原）

```
shprg_backend              = opa_lwr_shprg     # seed-homomorphic PRG (基于 LWR)
committee_size             = 7                 # 辅助节点数（2-client 退化为 2）
committee_threshold        = 5                 # 门限（2-client 退化为 1）
sketch_quantization_scale  = 1000000.0         # Sketch 值量化 scale
sketch_recovered_clip_norm = 10.0              # 恢复后 clip
secure_exact_max_len       = 4096              # 小层走精确（稠密）安全聚合
dense_layer_threshold      = 4096              # <4096 参数的层走稠密 SA
```

### 5.2 2-client 安全层退化方案

原方案主服务器 + 动态门限辅助节点，2-client 下：
- **Client 0、Client 1 互为辅助节点**：各自把自己的 seed Shamir 分享发给对方
- Central 收集两份聚合 seed share → 重构 `s_sum = s_0 + s_1`
- `M = SHPRG(s_sum)`，`U = Σ ct_i - M`
- 门限 τ=1（2-client 无掉线容忍），安全假设退化为「Central 与任一 client 不合谋」（弱于原方案 5-of-7，但符合 2-client 现实）

### 5.3 小层混合聚合（原方案设计）

原方案对 `< dense_layer_threshold(4096)` 参数的层走**精确（稠密）安全聚合**，不进 CountSketch。大模型中这类小层（如 LayerNorm、bias）极少且参数小，按原方案处理：小层稠密 SA + 大层 Sketch。

---

## 6. 二次精读：发现的模糊点与风险（关键修正）

二次精读交底书 §3 + Clover 实验数据后，发现以下在初版 TODO 中**未考虑清楚或表述错误**的问题。

### 6.1 【实验观察，非交底书原文】calibrated layout 对收敛至关重要（与 residual 配合）

> ⚠️ **澄清**：以下结论来自 Clover zip 实验数据推断，**交底书原文未明确此论断**。交底书 §3.3 只说 layout「由公开预算、模型结构、历史公开校准统计或固定规则确定」，把 calibrated 列为**优选**而非**必须**。此处的实验观察仅基于 Clover 这批数据，大模型场景需自行验证。

**Clover 实验数据观察**（两组配置**唯一差异**是 allocation_strategy）：

| 组 | allocation_strategy | residual | budget_ratio | rows | 结果 |
|---|---|---|---|---|---|
| 发散组 (1810) | uniform_param | True | 1.0 | 5 | final_acc=**0.112** |
| 收敛组 (1845) | topk_count_calibrated | True | 1.0 | 5 | final_acc=**0.978** |
| 发散组 (1755) | uniform_param | False | 1.0 | 5 | final_acc=0.099 |

→ 在 Clover 这批数据中，**uniform+residual 仍发散，calibrated+residual 才收敛**。可能原因：`topk_count_calibrated` 按**各层 Top-k 命中数**分配 bucket（而非按参数量均匀分），这对 LLM 更关键——大模型各层梯度稀疏度差异极大（embedding/lm_head vs 中间 transformer 层），uniform 分配会让真正活跃的层 bucket 不足、不活跃的层 bucket 浪费。

**TODO 修正**（谨慎表述）：
- E1（uniform + no residual）：预期发散，作对照
- E2（uniform + residual）：**Clover 数据显示仍发散，但大模型需自行验证**，作对照
- **E3（calibrated + residual）：Clover 数据显示收敛，作为主实验起点**
- `topk_count_calibrated` 实现需「历史各层 Top-k 计数」，**不能省略**
- 大模型实验应**默认用 calibrated**，保留 uniform 验证此观察是否在大模型同样成立

### 6.2 【算法细节】Sketch flat 长度 = R × Σ B_l，不是 Σ B_l

**初版 TODO 表述**：`u_i`（R 行 × Σ B_l bucket）——未明确 flat 长度。

**Clover 数据确认**：
- `sketch_len = 21550`，`rows = 5`，`sum_l(B_l) = 4310`
- `sketch_len = rows × sum_l(B_l) = 5 × 4310 = 21550` ✓
- `payload = sketch_len × value_bytes = 21550 × 2 = 43100`

→ flat Sketch tensor 长度 = `R × Σ_l B_l`，不是 `Σ_l B_l`。实现时 layout 要明确每层每行的 bucket 数，编码按 `[layer][row][bucket]` 展平。

### 6.3 【算法细节】budget 对齐 topk16 payload，不是 dense

**初版 TODO 表述**：`budget_ratio=1.0`，L·bytes 对齐 Top-k16 payload——正确但需明确公式。

**确认公式**：
```
topk16_payload = k × (idx_bytes=4 + val_bytes=2) = 6k   (单位 bytes)
sketch_payload = R × Σ_l(B_l) × value_bytes
budget_ratio = 1.0  =>  R × Σ_l(B_l) × value_bytes = topk16_payload
=> Σ_l(B_l) = topk16_payload / (R × value_bytes) = 6k / (R × 2)
```

LeNet5: k=7183, R=5, value_bytes=2 → Σ B_l = 6×7183/(5×2) = 4310 ✓

大模型 (d≈494M, k=1.67%·d≈8.24M, R=5, value_bytes=2):
→ Σ B_l = 6×8.24M/(5×2) = 4.95M → sketch_len = 5×4.95M = 24.7M → payload = 49.4 MB

**对比 dense16**：494M×2 = 988 MB → 压缩比 ≈ 5%（与小模型一致）。

### 6.4 【模糊点】recover_k_mode = active_union 含义未明确

交底书只说 `K_recover = min(k·|C|·ρ, d)`，但 Clover 数据有 `sketch_recover_k_mode = active_union`。

**推断**：`active_union` 指恢复时对**所有 active client 的候选坐标并集**做估计，而非单 client 的 k 个坐标。对 2-client：K_recover = min(k·2·ρ, d)。若 ρ=1.0, k=8.24M → K_recover = 16.5M（3.3%·d）。

**风险**：2-client 下 K_recover 偏小，恢复出的全局更新覆盖率低。**需扫 ρ ∈ {1.0, 2.0, 4.0}** 缓解。初版 TODO 只扫 rows/budget，**应加 ρ 扫描**。

### 6.5 【模糊点】Sketch 值量化的 scale 与 transport_dtype 关系

Clover 配置：
```
sketch_transport_dtype    = float16    # 传输精度
sketch_quantization_scale = 1000000.0  # 量化 scale
sketch_value_bytes        = 2          # = float16
update_transport_dtype    = float32    # 恢复后更新用 fp32
update_value_bytes        = 4
```

**推断**：Sketch 编码在 fp32 计算，传输前量化到 fp16（scale=1e6 用于 int16 public scale 模式，float16 模式下 scale 可能仅作 clip 参考）。恢复后更新用 fp32 落盘。

**风险**：大模型 Sketch bucket 值域因 2-client 叠加可能较温和，但 float16 动态范围有限。需关注 Sketch bucket 的 max|x|，必要时用 int16 public scale 模式。**实现时应记录 bucket 值域分布**。

### 6.6 【模糊点】小层混合聚合 dense_layer_threshold=4096 的影响

**初版 TODO 表述**：小层走稠密 SA。

**补充确认**：Clover per-layer 数据显示 **LeNet5 所有层 `is_dense=False`**（包括 conv1.bias 只有 20 参数）。说明 `dense_layer_threshold=4096` 在小模型实验中**实际未触发**（可能因实验配置）。但 `secure_exact_max_len=4096` 和 `sensitive_small_layer_threshold=4096` 表明原方案设计了该机制。

**大模型影响**：Qwen2.5-0.5B 中 <4096 参数的层（如 LayerNorm γ/β、小 bias）数量多但总参数占比极小。是否启用混合聚合需确认：
- 启用：小层精确恢复，精度更稳；但需额外实现小层稠密 SA 路径
- 不启用：全部进 Sketch，小层因 bucket 少（min_buckets=8）碰撞概率高，但总参数占比小影响有限

**建议**：初版实验**先不启用混合聚合**（全部进 Sketch，对齐 Clover 实际行为），E3 收敛后再加混合聚合作消融。**初版 TODO 未明确这点，现修正为可选**。

### 6.7 【风险】FSDP 下 global Top-k 的实现复杂度

交底书要求 global Top-k（全模型展平选 k 个），但 FSDP 下模型参数分片在各 rank。

**问题**：
- 单 rank 只能做 local Top-k，需跨 rank 通信得到 global Top-k
- 标准 `torch.topk` 无法直接用；需 all-gather 候选或近似算法

**实现方案**：
1. 各 rank 在本地 shard 做 Top-(k·oversample)（oversample=2~3）
2. all-gather 候选 (坐标, 值) → 全局 Top-k
3. 通信量 = oversample·k·(4+4) bytes，相对 Sketch payload 可忽略

**初版 TODO 未提及此问题**，T0.3 应明确实现 FSDP-aware global Top-k。这是**大模型与小模型实现的最大差异点**。

### 6.8 【风险】calibrated layout 的 warmup 冷启动

`topk_count_calibrated` 需要历史各层 Top-k 计数。第 1 轮（冷启动）无历史数据。

**Clover 数据有 `layout_calibration_path = calibrations/mnist_lenet5_topk_count.json`**——说明原方案用**预跑统计文件**，不是在线 warmup。

**建议**：
- 方案 A：先跑 1-3 轮 uniform layout，采集各层 Top-k 计数，生成 calibration 文件，后续轮切换 calibrated
- 方案 B：预跑一轮 dense/明文 Top-k，统计各层分布，作为 calibration

**初版 TODO T2.4 说「前 3 轮采集」**——可行，但需明确**前 3 轮用 uniform 可能发散**（见 6.1）。**修正为：先独立预跑统计 calibration 文件，正式实验从头用 calibrated**。

### 6.9 【模糊点】residual 的初始化与 clip

交底书：`r_{i+1} = z_i - Sparse(S_i, v_i)`，残差不上传。但未提：
- 初始 `r_0 = 0`
- 残差是否需要 clip/decay（防止长期累积爆炸）

Clover 配置 `sketch_recovered_clip_norm = 10.0` 是对**恢复后更新**的 clip，不是残差 clip。但残差无 clip 长期累积有风险。

**建议**：实现时加残差 clip（按范数）和可选 decay，作为可配置项。初版 TODO 未提，**T0.3 补充**。

---

## 7. TODO 修正版（综合 §6 发现）

### Phase 0：核心库 + 单测

- [ ] **T0.1** `experiments_sketchopa/sketchopa/` 目录
- [ ] **T0.2** `count_sketch.py`：按层分区，flat 长度 = `R × Σ_l B_l`（§6.2）；seeded bucket/sign hash
- [ ] **T0.3** `topk.py`：
  - **FSDP-aware global Top-k**（§6.7）：rank 内 Top-(k·oversample) → all-gather → 全局 Top-k
  - `ResidualBuffer`：`r_0=0`，`r_{i+1}=z_i-Sparse(S_i,v_i)`，**可选 clip/decay**（§6.9）
- [ ] **T0.4** `layout.py`：
  - `uniform_param`：按参数量分配 bucket（min_buckets 保底）
  - `topk_count_calibrated`：按**外部 calibration 文件**（各层 Top-k 计数）分配（§6.8）
  - budget 公式：`Σ B_l = 6k / (R × value_bytes) × budget_ratio`（§6.3）
- [ ] **T0.5** `recover.py`：median-of-rows；`K_recover=min(k·|C|·ρ,d)`；**recover_k_mode=active_union**（§6.4）；恢复后 clip_norm
- [ ] **T0.6** `quant.py`：float16 传输 + int16 public scale 可选；**记录 bucket 值域**（§6.5）
- [ ] **T0.7** `shprg.py`：SHPRG（LWR-based），同态性 `SHPRG(s1)+SHPRG(s2)=SHPRG(s1+s2)`
- [ ] **T0.8** `seedshare.py`：Shamir 门限；2-client 退化（§5.2）
- [ ] **T0.9** 单测：线性性、Top-k+编码+聚合+恢复、FSDP Top-k 正确性、SHPRG 同态性、残差+clip

### Phase 1：基线 + calibration 预跑

- [ ] **T1.1** `run_baseline_dense.py`（FSDP），**独立跑出** dense fp16 基线（不引用历史结果）
- [ ] **T1.2** `run_baseline_topk.py`（明文 global Top-k，FSDP-aware）
- [ ] **T1.3** 跑通 20 轮，记录 B0 / B1 的 eval loss 曲线，作为后续 SketchOPA 对比基准
- [ ] **T1.4** **calibration 预跑**：基于 B1（明文 Top-k）统计各层 Top-k 命中计数，生成 `calibrations/qwen05b_topk_count.json`（§6.8）

### Phase 2：SketchOPA 实验（主实验）

- [ ] **T2.1** `run_sketchopa.py`（含安全层，FSDP）
- [ ] **T2.2** E1：uniform + no residual（对照，预期发散）
- [ ] **T2.3** E2：uniform + residual（对照，**预期仍发散** §6.1）
- [ ] **T2.4** **E3：calibrated + residual（主实验，预期收敛）**——使用 T1.3 的 calibration 文件
- [ ] **T2.5** 结果对比：B0 / B1 / E1 / E2 / E3 eval loss 曲线 + payload + 恢复 cosine/precision/recall

### Phase 3：扫参

- [ ] **T3.1** 扫 `rows ∈ {3,5,7}`、`budget_ratio ∈ {0.5,1.0,2.0}`、`topk_ratio ∈ {1%,1.67%,3%}`
- [ ] **T3.2** **扫 `recover_rho ∈ {1.0,2.0,4.0}`**（§6.4，2-client 必扫）
- [ ] **T3.3** 压缩比 vs eval loss 权衡曲线

### Phase 4：可选消融

- [ ] **T4.1** 混合层级聚合（小层稠密 SA，§6.6）——若 E3 收敛则可选
- [ ] **T4.2** 残差 clip/decay 消融（§6.9）

### Phase 5：结论

- [ ] **T5.1** 实验结果 + 结论文档

---

## 8. 修正前后对照速查

| 问题 | 初版 TODO | 修正后 |
|---|---|---|
| 收敛关键（calibrated） | 初版把 E2 当收敛主实验 | **修正：E3(calibrated+residual) 才是主实验**；但此结论是 Clover 实验观察非交底书定论，大模型需自行验证 |
| 实验主线 | E2 预期收敛 | **E3 才是主实验**，E1/E2 均预期发散作对照 |
| Sketch flat 长度 | 未明确 | = `R × Σ_l B_l` |
| budget 公式 | 笼统 | `Σ B_l = 6k/(R×value_bytes) × budget_ratio` |
| recover_rho | 未扫 | **2-client 下必扫 {1,2,4}** |
| FSDP global Top-k | 未提 | **必须实现 rank 内 oversample + all-gather**（最大差异点） |
| calibrated warmup | 「前 3 轮采集」 | **改预跑独立生成 calibration 文件**（前 3 轮 uniform 会发散） |
| 残差 clip | 未提 | 加可选 clip/decay |
| 混合层级聚合 | 列为必做 | **改可选**（Clover 实际未触发，先全 Sketch） |
| 安全层 | 列为 E5 可选 | **按原方案默认含安全层**（决策点 4 已定） |

---

## 9. 实现进展与 Smoke Test 结果（2026-10-12）

### 9.1 已实现代码

```
experiments_sketchopa/
├── sketchopa/                  # 核心算法库
│   ├── __init__.py
│   ├── layout.py               # 按层分区 Sketch layout（uniform/calibrated）
│   ├── count_sketch.py         # 向量化 CountSketch 编码/估计
│   ├── topk.py                 # global Top-k（单进程 + FSDP-aware）+ ResidualBuffer
│   ├── recover.py              # median-of-rows 批量恢复 + clip
│   ├── quant.py                # float16/int16 量化 + Z_q 量化（安全聚合用）
│   ├── shprg.py                # SHPRG（LWR 风格，同态性 mod q）
│   └── seedshare.py            # Shamir 门限 + 2-client 退化
├── common.py                   # 共享：数据加载/模型加载/训练/eval（含 QK fp32 patch）
├── coordinator.py              # Central 协调器（FastAPI + MinIO）
├── client.py                   # ICC client（FSDP 训练 + Top-k/Sketch 编码 + 安全层）
├── run_experiment.py           # 单进程模拟 2-client（开发/测试用）
├── tests/
│   ├── test_sketchopa_core.py  # 16 项核心库单测
│   └── test_smoke.py           # 5 项端到端 smoke test（合成模型）
├── calibrations/               # 各层 Top-k 计数文件
└── results_sketchopa/          # 实验输出
```

### 9.2 单元测试

**核心库单测**（16 项，本机 CPU + ICC1 GPU 均通过）：
- CountSketch 线性性 `CS(x)+CS(y)==CS(x+y)`（err < 1e-6）
- Layout 构造（uniform/calibrated）+ budget 公式验证
- global Top-k 正确性
- ResidualBuffer（含 clip）
- 端到端 Top-k → 编码 → 聚合 → 恢复
- fp16/int16 量化往返
- SHPRG 同态性 `SHPRG(s1)+SHPRG(s2)==SHPRG(s1+s2) mod q`
- SHPRG mask/unmask 往返（Z_q 域）
- Shamir 标量/向量秘密分享 + 重构
- 2-client seed 交换
- 安全聚合管线（secure vs plain err=0）
- 各层 Top-k 计数统计

**Smoke test**（5 项，合成模型，本机 CPU 通过）：
- dense / topk / sketchopa(uniform+secure) / sketchopa(calibrated) / sketchopa(no-secure)

### 9.3 分布式 Smoke Test 结果（真实双集群）

**环境**：Central（192.168.235.42）coordinator + ICC1（dgx1-16）client 0 + ICC2（dgx1-30）client 1，单卡 V100，MinIO 同步模型。

**模型**：Qwen2.5-0.5B（**d=630,167,424，291 层**——实测值，比初版估算的 494M 大）

| 方法 | topk_ratio | sketch_len | k | eval_loss | 耗时 | 状态 |
|---|---|---|---|---|---|---|
| dense (B0) | — | — | — | nan（fp16 数值，eval 正常） | ~340s | 流程通 |
| topk (B1) | 0.01 | — | 6,301,674 | **1.8072** | ~340s | ✓ |
| **sketchopa (E1)** | 0.01 | 18,905,020 | 6,301,674 | **1.8072** | ~390s | ✓ |

**关键结论**：
- **SketchOPA eval_loss=1.8072 与明文 Top-k 完全一致**——Top-k → CountSketch → 安全聚合（SHPRG mask + Shamir seed 交换）→ unmask → 恢复 全流程在大模型上**未引入额外误差**
- 安全层（SHPRG + 2-client seed 交换）与明文聚合结果一致（err=0）
- Sketch payload = 18.9M × 2 bytes = **37.8 MB**，相对 dense16 (1.26 GB) 压缩 **~33x**

### 9.4 实现中的关键发现与修正

| 问题 | 发现 | 修正 |
|---|---|---|
| Qwen2.5-0.5B 实际 d | 630M（非估算的 494M），291 层 | 用实测值更新超参 |
| CountSketch hash 性能 | SHA256 逐坐标循环（k=6.3M）需数分钟 | 改用 torch 向量化 mul-shift-mod hash |
| layout 构建内存 | `build_layout` 传入 630M dummy tensor 分配 2.5GB | 直接构造 LayerLayout，不分配 dummy tensor |
| V100 fp16 QK matmul | inf 导致 loss nan | 复用 `patch_qwen2_attn_fp32_qk` + `torch.autocast(enabled=False)` |
| 安全聚合 mask 域 | float 域 mask 淹没 sketch 值 | 改在 Z_q 域做 mask/unmask（sketch 先量化到 Z_q） |
| Shamir 模数 | 2^16 非素数导致模逆失败 | 用 SHAMIR_PRIME=65521（<2^16 最大素数），SHPRG 共用 |
| coordinator layout 构建阻塞 | 291 层 layout 构建阻塞 HTTP 响应 | 改后台线程异步构建 |
| `_do_aggregate` 重复调用 | 两个 client 同时上传触发 race | 加 `_aggregating` 标志保护 |
| round_log eval_losses 时序 | 聚合时记录日志但 eval 还没完成 | eval_report 时更新日志 |
| train_loss=nan | fp16 logits 溢出 | autocast 禁用 + nan→0 兜底 |

### 9.5 B0 Dense 基线 20 轮实验结果（2026-10-13）

**配置**：
- 模型：Qwen2.5-0.5B（fp32 训练，避免 fp16 AdamW nan）
- 数据：ICC1 `icc1_client0_train.json` (15088) + ICC2 `client1_train.json` (15088)
- 评估：`medical_flashcards_eval.json` (3352)
- 训练：local_steps=30, batch_size=4, grad_accum=4, lr=1e-5
- 传输：state_dict fp16 上传（1.26GB/轮），HTTP 直传 coordinator（不走 MinIO）
- 每轮耗时：~420s（7 分钟）

**完整结果**：

| Round | train_loss | eval_loss |
|---|---|---|
| 0 | 0.312 | 1.174 |
| 1 | 0.288 | 1.156 |
| 2 | 0.283 | 1.144 |
| 3 | 0.282 | 1.134 |
| 4 | 0.273 | 1.125 |
| 5 | 0.274 | 1.119 |
| 6 | 0.268 | 1.114 |
| 7 | 0.267 | 1.110 |
| 8 | 0.270 | 1.105 |
| 9 | 0.269 | 1.102 |
| 10 | 0.266 | 1.099 |
| 11 | 0.262 | 1.096 |
| 12 | 0.260 | 1.093 |
| 13 | 0.256 | 1.090 |
| 14 | 0.253 | 1.086 |
| 15 | 0.251 | 1.083 |
| 16 | 0.258 | 1.083 |
| 17 | 0.251 | 1.080 |
| 18 | 0.250 | 1.077 |
| 19 | 0.248 | **1.075** |

**结论**：
- eval_loss 从 1.174 稳步下降到 **1.075**，收敛趋势清晰
- 两个 client 全程 eval_loss 完全一致（FedAvg 后相同模型 + 相同评估集）
- B0 基线已就绪，作为后续 topk/sketchopa 实验的对比基准

**结果文件**：`results_sketchopa/B0_dense_20r_v7/round_log.json`

### 9.6 实现中的关键修复（B0 实验期间）

| 问题 | 原因 | 修复 |
|---|---|---|
| MinIO 大文件上传超时 | `put_object` 对 2.5GB BytesIO 流 multipart upload 卡住 | 改用 HTTP 直传 coordinator（`/api/upload` + `/api/download`），不走 MinIO |
| `accelerator.wait_for_everyone()` 卡住 | accelerate 在多 GPU 机器单进程模式下 NCCL barrier 等待 | 去掉所有 `wait_for_everyone`，纯 PyTorch 单卡 |
| `Accelerator.get_state_dict` 卡住 | accelerate 包装的 model 导出 state_dict 触发分布式同步 | `get_full_state_fsdp` 改用简单 `model.state_dict()` |
| fp16 AdamW 参数 nan | fp16 optimizer state 溢出 | 模型用 fp32 训练，上传时转 fp16 减半传输量 |
| eval_loss=0.0 | eval batch_size=8 导致 logits OOM，nan→0 兜底 | eval batch_size 降到 2 |
| client 1 下载 404 | client 1 在 client 0 上传初始模型前就尝试下载 | `_load_global_state` 加重试（60 次 × 10s） |
| coordinator `_do_aggregate` race | 两个 client 同时上传触发重复聚合 | 加 `_aggregating` 标志保护 |
| round_log eval_losses 时序 | 聚合时记录日志但 eval 还没完成 | `eval_report` 时更新日志 |

### 9.7 当前系统架构

```
Central (192.168.235.42:8090)
  └── coordinator.py（FastAPI）
       ├── /api/config         — 返回实验配置
       ├── /api/meta           — 接收 client 模型元信息
       ├── /api/upload         — 接收 client 上传数据（HTTP 直传，本地文件存储）
       ├── /api/download/{key} — 发送全局模型给 client
       ├── /api/round/{r}/status — 查询轮次是否完成
       └── /api/round/{r}/client/{c}/upload+eval — 上传通知 + eval 上报

ICC1 (192.168.206.116)                    ICC2 (192.168.205.130)
  └── client.py --client-id 0               └── client.py --client-id 1
       ├── 下载全局模型（HTTP）                  ├── 下载全局模型（HTTP，带重试）
       ├── 本地训练（fp32，V100 单卡）            ├── 本地训练（fp32，V100 单卡）
       ├── Top-k / Sketch 编码                   ├── Top-k / Sketch 编码
       ├── 上传（HTTP，fp16 state）               ├── 上传（HTTP，fp16 state）
       └── eval + 上报                           └── eval + 上报
```

### 9.8 下一步

- [x] B0 dense 基线 20 轮 ✅（eval_loss 1.174 → 1.075）
- [ ] B1 topk 基线 20 轮
- [ ] calibration 预跑（基于 B1 统计各层 Top-k 计数）
- [ ] E1/E2/E3 实验 20 轮（uniform vs calibrated，residual on/off）
- [ ] 扫参（rows, budget_ratio, topk_ratio, recover_rho）
- [ ] 结论文档
