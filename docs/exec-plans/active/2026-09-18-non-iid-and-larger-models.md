# 后续：Dolly Non-IID + 更大模型

- **状态**：active
- **创建**：2026-09-18
- **更新**：2026-09-18（主实验改为 Dolly 同集 IID vs Dirichlet；跨机构后置）
- **不改**：Hadamard + INT16 + Issue #1 量化语义；S3R12v3 `public_random` mask；SecAgg 协议不按模型名特化
- **关联**：
  - [当前联调流程](../../algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
  - [生产切片 DATA/SCALE](2026-09-11-dual-cluster-to-production.md)（DATA-1 切分脚本已有；DATA-2 共享 eval 仍是现状）
  - [SecAgg 时间效率](2026-09-18-secagg-time-efficiency.md)（0.5B 墙钟标尺）
  - 当前 IID 基线：无 SecAgg `20260914-final-clean`；SecAgg `202609180941` / `202609181406`
  - [Non-IID 相关工作调研](../../algorithm/2026-09-18-non-iid-related-work.md)

## 0. 现状（先写死，避免和「真实机构」混谈）

当前双集群 **不是两份雷同文件**，但是 **独立同分布（IID）**：

| 项 | 现在 |
|---|---|
| 来源 | 同一份 `medalpaca/medical_meadow_medical_flashcards` |
| 训练 | `data/splits/icc1_client0_train.json` 与 `client1_train.json`，各 15088 条 |
| 切法 | `seed=20260831` 对训练集 **随机 50/50**（见 `data/splits/README.md`） |
| 领域 | 两端都是医学闪卡 QA |
| 评估 | 两端共用 `data/medical_flashcards_eval.json`（3352 条） |

所以：样本不重复，**分布相同、领域相同**。FedAvg 很「顺」，eval 从 ~1.50 降到 ~0.98，不能代表跨机构。

仓库里已有 `scripts/split_federated_data.py` 的 Dirichlet，但是按闪卡文本前缀当伪标签，**不是** FedIT 那种按 Dolly `category` 的切法。

**主实验改成审稿人预期的 B 档**（见 [相关工作 §7](../../algorithm/2026-09-18-non-iid-related-work.md)）：两端都用 **Dolly-15k**，IID 50/50 vs 按 category 的 Dirichlet。闪卡 vs Dolly 的跨机构（C 档）后置。变量一次只动一类：先 0.5B 换切法，再 3B/7B。

---

## 1. 看板

| ID | 优先级 | 状态 | 任务 | 依赖 |
|---|---|---|---|---|
| DATA-N0 | P0 | **done**（本文） | 写清当前闪卡是同域 IID；B 档 ≠ C 档跨域 | — |
| DATA-D1 | P0 | **done** | 下载 Dolly-15k，转成 `instruction/input/output`，保留 `category` | — |
| DATA-D2 | P0 | todo | 两套切分：IID 50/50 vs 按 `category` Dirichlet \(\alpha=0.5\)（2 client）+ 共享 Dolly eval | DATA-D1 |
| DATA-D3 | P1 | todo | 0.5B、无 SecAgg：Dolly IID vs Dirichlet，各至少 5 轮 | DATA-D2 |
| DATA-D4 | P1 | todo | 同上 + SecAgg；对照 Dolly IID，不对照闪卡 0.985 | DATA-D3 |
| DATA-C1 | P2 | todo | **后置**：ICC1 闪卡 + ICC2 Dolly（跨机构）；每端 hold-out 分开记 | DATA-D3 |
| MODEL-3B | P1 | todo | Qwen2.5-3B，仍用现有 medical IID，先无 SecAgg 再 SecAgg | — |
| MODEL-7B | P2 | todo | Qwen2.5-7B；先过显存/Central RAM/上传体积，再 5 轮 | MODEL-3B |
| SCALE-MEM | P1 | todo | 上 3B/7B 前复核 Central 峰值内存（SCALE-1 并未真正流式） | MODEL-3B |
| QUANT-8 | P2 | todo | 大模型带宽不够时：SecAgg 传输 INT16 → INT8 对照（保 Hadamard） | MODEL-3B |
| QUANT-4 | P3 | todo | INT4 / 更低 bit；需单独评估（Kashin 等），不能当 INT16 开关 | QUANT-8 |

建议落地顺序：**DATA-D1 → D2 → D3 → D4**；**DATA-C1 跨机构后置**。MODEL-3B 可并行下权重，实验不要同一周交叉。QUANT-8 只在大模型 INT16 能跑且体积成瓶颈时才做。

---

## 2. 主实验：Dolly 同集 Non-IID（先做）

两端都用 `databricks/databricks-dolly-15k`（约 15k，8 个 category）。这是 FedIT / Shepherd / FS-LLM 的标准设定。

训练入口只认 `instruction/input/output`；转换时 **必须留下 `category`**，Dirichlet 才能按类切。现有 `split_federated_data.py` 用文本前缀当伪标签，对 Dolly **不够**，要按真正的 category 列切。

| 切法 | 做法 | 评估 |
|---|---|---|
| IID 50/50 | 随机对半，seed 固定 | **可以共用一份** Dolly hold-out（同域，这是通用报法） |
| Dirichlet | 按 8 类 `Dir(α=0.5)` 分给 2 client（FedIT 同款 α） | 仍报同一份 Dolly eval；可加每端类别直方图证明切开了 |

成功：Dirichlet 的 eval 差于或明显不稳于 IID；SecAgg 与无 SecAgg 的相对关系仍说得清。数字 **不要** 和闪卡 0.985 比。

切分脚本参数化：比例、按 label/来源的 Dirichlet 非 IID、每 client 样本上限——闪卡那套伪标签 Dirichlet 不能拿来冒充本实验。

### 2.1 后置：跨机构（DATA-C1）

ICC1 继续医学闪卡，ICC2 用已转换的 Dolly。领域不同，eval **必须拆开**，不能平均。等 Dolly 的 D3 跑通再做。更远的候选：Code-Alpaca、FinGPT；Alpaca-GPT4 等 3B/7B。

---

## 3. 更大模型（3B → 7B）

家族继续 **Qwen2.5 base**（不要第一枪换 Instruct / 换 Llama，tokenizer 和 `Qwen2DecoderLayer` wrap 能少改）。

两端各 8×V100 32GB；`run_s3r12v3_fsdp.py` 已开 `gradient_checkpointing`。部署文档估算 7B + batch 4 + seq 512 ≈ 19GB/卡，**纸面够，但没实跑**。

### 3.1 规模会放大什么（0.5B 标尺 × 倍数）

当前 SecAgg 约 coverage 10%：上传 ~140 MiB，~269 window，全量 `global_state` ~0.9 GB。

| | 0.5B（已跑） | 3B（约 ×6） | 7B（约 ×14） |
|---|---|---|---|
| 每端上行 INT16 ~10% | ~0.14 GB | ~0.8 GB | ~2 GB |
| window 数（同 `block_size`） | ~269 | ~1.6k | ~3.7k |
| Central 全量 fp32 量级 | ~1 GB | ~6 GB | ~14 GB |
| 20 轮 MinIO（含偶发全量） | 可接受 | 需盯盘 | 必须 `write_full_global_every_n_rounds` |

P0/P1 单 blob / 并行 unmask **不跟模型名走**，window 变多时请求次数仍应是 O(1) PUT；墙钟会跟 **计算 + 带宽** 涨。Central **没有 GPU**，unmask 仍在 CPU；7B 的 server `wait_s` 会明显长于 0.5B 的 ~6s。

SCALE-1 的「流式、不整模常驻」**并未做到**（结项里写的是峰值可接受、真正流水线留到 7B 前）。上 3B 先量一次 Central RSS；若 7B 峰值顶满再改 `apply_block_delta`。

### 3.2 MODEL-3B

1. 两端 + 评估机下载 `Qwen/Qwen2.5-3B` 到与 0.5B 对称的 `model/Qwen/Qwen2.5-3B`
2. `accelerate` wrap 仍 `Qwen2DecoderLayer`；`batch_size`/`seq_len` 必要时降到 4/512
3. **数据先保持 medical IID**（只改模型）
4. 无 SecAgg 5 轮冒烟 → 无 SecAgg 20 轮（新 fp16 基线，**不能**沿用 0.985）→ SecAgg 5 轮对齐 eval → 再 20 轮
5. 超时：`client_upload_timeout_s` 按上传体积加大

### 3.3 MODEL-7B

在 3B 的 SecAgg 5 轮稳定之后：

1. 同样只换 `model_path`，数据仍 IID 医学
2. 先 1～2 轮看：OOM、MinIO 超时、Central RSS、`upload_blocks_MiB`
3. coverage 可先保持 10%；若 Central/带宽吃紧再降 H（那是另一实验，不要和「7B 能不能跑」绑死）
4. 5 轮通过后再谈 20 轮和跨域

### 3.4 传输量化（QUANT-8 / QUANT-4）— 大模型带宽杠杆，不是现在的加速

这是 **通信 bit 宽**，和 S3R12v3 的「传 10% block」是正交的。0.5B 每端约 140 MiB，墙钟不在字节上，**不要在 0.5B 上为了「再压一点」去改 INT16**。

| 路径 | 现状 | 以后 |
|---|---|---|
| 非 SecAgg | `transfer_dtype: int8` 已有（SCALE-2），体积约 fp16 的一半 | 大模型非 SecAgg 对照可直接开 |
| SecAgg | 默认 **Hadamard + INT16**（`202609181406` R20≈0.985） | **QUANT-8**：同一套旋转，只把模数/载荷改 8-bit |
| 更低 bit | 未做 | **QUANT-4**：不能当 yaml 开关；文献上往往要更强旋转（Kashin） |

顺序：**3B/7B 先用现有 INT16 跑通 → 体积/MinIO 真吃紧 → 先考虑再降 `coverage_h` → 再 QUANT-8 对照 eval**。INT8 目标是 7B 每端从 ~2 GB 量级减半，验收仍是「该模型自己的 INT16 基线同量级」，不是沿用 0.5B 的 0.985。

不要和跨域同一周叠：eval 变差时分不清是 Non-IID 还是量化。

---

## 4. 验收总则

- 跨域：N3 无 SecAgg 能跑完；N4 的 SecAgg eval **按本域分别** 对照，不要求医学 0.985
- 3B/7B：该模型自己的 fp16 基线；SecAgg 5 轮与该基线同量级；`upload_blocks_MiB` 与「选中元素 × 2B」相符
- 协议：禁止按 3B/7B 写死 window 数或层名

---

## 5. 明确以后再开（不进本看板）

- 13B / 多 client（>2）
- 中英混合跨域
- 为跨域再换量化或关掉 Hadamard（跨域和 QUANT-8 分开做）
- PERF-7 AES-CTR（仍按时间计划 won't do；7B 若 SHAKE 真成 encode 大头再单开）
- 0.5B 上为压体积而做 SecAgg INT8/INT4

---

## 6. 对照：为何不是 2019 的 Top-k / 三值量化

传统通信压缩（Konečný 2016  sketched updates、DGC、**STC** Sattler 2019 的 top-k + ternary `{-μ,0,+μ}` 等）也是「少传更新」。**少传本身不是本方案的发明**；FedAvg 本来传的就是模型 delta，不是单步反传梯度。

不能直接换成 Top-k / ±5 档，原因不是「他们传梯度、我们传参数」这种假对立，而是 **选哪些坐标、服务器能看见什么**：

| | 2019 Top-k / STC | 本方案：公开 block mask + SecAgg INT16 |
|---|---|---|
| 选谁 | 看本轮私有 \|delta\| 大小 | 公开随机、H 轮无放回，**不看**本轮更新 |
| 两端坐标 | 各不相同 | **同一套** selected blocks / window |
| 服务器看见 | 下标 + 明文（或三值）更新 | SecAgg 下只见整数域 **和** |
| 带宽 | 更狠（top-k × 1～2 bit） | 约 10% × INT16，可预期、均匀 |
| LLM | 非结构化 top-k 不友好 | 1MB block，对齐 FSDP |

规范原话：mask **不得**使用任一 ICC 的私有更新作输入（见 `docs/ai-design/Public_Block_Mask_v1.txt`）。Top-k 正好违反这一条，也 **无法** 做 Windowed SecAgg（支撑集不一致就不能对位相加、mask 对消）。

优势首先是 **能安全聚合的结构化稀疏**，其次才是省带宽。若服务器可信、只要体积，STC 类可以更小；那是另一条实验，不替代当前默认路径。

