# AI_DESIGN_REVIEW — 外部研究评审报告（FedScale `deploy/central-server-minio`）

- **生成**：2026-09-18
- **评审后端**：Codex MCP，**`gpt-5.6-sol` @ `xhigh`**（按 `pure-github-repo/CLAUDE.md` 的 `## Reviewer Configuration` 覆盖 skill 默认的 `gpt-6-astra` / `ultra`）
- **threadId**：`01a0b3e2-54e9-7b03-9024-821824c2a7e4`　**轮次**：1　**耗时**：6m34s
- **请求 brief**：`idea-stage/RESEARCH_REVIEW_REQUEST.md`
- **输入**：`CURRENT_METHOD.md` · `AI_DESIGN_RELATED_WORK.md` · `CURRENT_VS_LITERATURE.md`
- **Trace**：`.aris/traces/research-review/2026-09-18_run01/`（manifest + 逐字原始回复）
- **状态**：**未出现 `REVIEW_UNAVAILABLE`**；无传输错误；未触发能力回退
- **约束遵守**：本轮**未执行任何新实验、未修改任何代码**。§6 的补强项全部是建议清单。

> **证据等级标注**：全文对每条实质判断标注 `SUPPORTED` / `PARTIAL` / `SPEC_ONLY` / `UNKNOWN`。
> `SPEC_ONLY` 不因规范写得好而升级为 `SUPPORTED`。

---

## 0. 一句话结论

> **[SUPPORTED] 当前方案最准确的研究地位是「工程集成」（engineering integration）** —— 它把已有的稀疏/部分上传机制、标准 Bonawitz 式安全聚合、成熟的 Hadamard 旋转量化、以及常规分布式系统工程**拼装进一条可运行的双集群 0.5B 流水线**，并给出了一个窄但真实的结果。
> **[UNKNOWN] 但它目前**尚未**达到系统类研究贡献的证据门槛**：关键的扩展性轴、协议一致性、可重放性、dropout 行为全部未验证。

---

## 1. 相对已有工作的主要优点

| # | 优点 | 等级 |
|---|---|---|
| S1 | **block 调度有一个简单、可审计的组合性质**：按 `position mod H` 分配排列位置，使每个可轮转 block 在一个完整 Epoch 内恰好占一个 slot。**该性质独立于学习结果成立**，是整套方案里最干净的部分 | `SUPPORTED` |
| S2 | **block 粒度直接消除了 key 大小不均造成的上传波动**：H=5 时 selected/uploaded 体积 18.5%–21.5%，对照旧 key 级路径 7.3%–37.4%，且保持 Epoch 精确覆盖。**这是全部材料中最扎实的系统性正向结果** | `SUPPORTED` |
| S3 | **"全模型本地训练 + 部分上传"是与 FedRolex / FjORD / HeteroFL 式部分训练真实不同的设计取舍**（算力换带宽的方向相反），尽管它本身不构成新算法 | `SUPPORTED` |
| S4 | **数值集成在文档所述的那一个配置下可信**：20 轮终点 0.985 vs 0.987；5 轮优化路径逐轮复现先前的 eval 值。**支持"可行性"，不支持"统计等价"或"泛化"** | `PARTIAL` |
| S5 | **单 blob、O(1) 通知、并行 unmask、异步状态写盘是有用的工程**：5 轮把整轮时间压到 110–134s 且上报载荷不变 | `PARTIAL` |
| S6 | **书面 SecAgg 状态机比多数原型描述规范得多**：明确分离 public 选择状态与密码学 attempt 状态、把 mask 绑定到 cohort/layout/window 标识、冻结 `U*`、写明禁止双重重构不变量。这些是**继承自已确立 SecAgg 工作的、正确的协议设计选择** | `SPEC_ONLY` |
| S7 | **前期文献分析正确识别了两个最关键的父本**：Bonawitz CCS 2017（密码学）与 Bonawitz et al. arXiv:1912.00131（数值） | `SUPPORTED` |

---

## 2. 主要缺点与证据缺口（按被审稿人攻击的概率排序）

### D1. 最亲近的方法论文献家族被漏识别 —— 本方案更像「带 memory 的通信稀疏化」而非「部分训练」

- `[SUPPORTED]` 客户端**训练完整模型**、只传输选中坐标、把未传坐标的信息留在 `block_memory` —— 从机制上看，它**至少同样自然地属于 communication sparsification with memory / error feedback** 家族，而不只是 partial training 家族。
- `[SUPPORTED]` 漏掉的近邻是 Random-k / 坐标稀疏化文献，特别是 **Stich, Cordonnier, Jaggi, *Sparsified SGD with Memory*（arXiv:1809.07599，2018）**（已独立核验）。
- `[SUPPORTED]` 前期分析**只为 `quant_residual` 引了 error feedback**，却没有充分比较未选中坐标的 `block_memory`；而它的 **decay = 0.9 是一种非标准、启发式的残差处理**，与 EF 的标准定义不同。
- `[PARTIAL]` 这一漏项**进一步压缩第 1 层的新颖性**：剩下的区别只是"确定性的、cohort 一致的、无放回轮转的、有界每轮体积的 block 调度"，而**不是**"全训 + 稀疏通信 + memory"这个一般想法。

### D2. 材料把「规范」与「实际被验证的子集」混在一起

- `[SPEC_ONLY]` Shamir 分享、两条 dropout 恢复分支、签名 survivor transcript、禁止双重重构的执行、fresh-attempt 恢复、overflow 拒绝、全部 dropout 故障用例 —— **只存在于 SecAgg 规范中**。
- `[PARTIAL]` 执行文档建立的只有：pairwise 对消、self mask 移除、2 client 无 dropout 的 session 同步。**它们没有在实验上建立机密性** —— 一次成功的 2-client 运行**无法证明**服务器缺少重构所需的信息。
- `[UNKNOWN]` 文档**没有在"当前已实现方法"与"未来协议"之间划出版本边界**。两者都叫 "FedScale Windowed SecAgg v2"，这会**诱导审稿人把 spec-only 的性质读成已交付的性质**。这是文档措辞层面的重大风险。

### D3. 算术包络只支持已被演示的 2-client 情形（**本轮最有分量的新发现**）

- `[SUPPORTED]` SecAgg 规范 §9.2 要求 `N_max · Q_max < q/2`。当前数值路径固定 `q = 2^16 = 65536`、`Q_max = 16383`。于是：

  | N | `N · Q_max` | `< 32768` ? | 余量 |
  |---|---|---|---|
  | **2** | 32766 | ✅ | **仅 2** |
  | 3 | 49149 | ❌ | 超界 16381 |
  | 4 | 65532 | ❌ | 超界 32764 |
  | 8 | 131064 | ❌ | 超界 98300 |

- `[SUPPORTED]` 即：**当前 16-bit 配置在规范自己的保守解码条件下，只对 N=2 成立**。而规范附录 A **推荐的是 modulus 2^32**（`2^31 / 16383 ≈ 131072` 个等权客户端），实现却选了 2^16。
  > **本报告已独立复算并核对原文**：`_secagg.txt:156`（§9.2 边界）、`_secagg.txt:277`（附录 A "优先 2^32"）、`2026-09-15-...md:187`（q=2^16、Q_max=16383）。**结论成立。**
- `[UNKNOWN]` **加权 FedAvg 也未定义清楚**：文档交替出现 "FedAvg"、除以 `N`、暴露 `num_examples`、以及提到加权 overflow 条件，但**没有冻结不等样本权重在整数域中如何编码**。
- `[PARTIAL]` 这正是"四层分解掩盖了安全–数值–聚合耦合"的一个具体例证 —— 也是**任何 N>2 或跨机构可扩展性主张的硬性阻断**。

### D4. 跨实现可重放性目前是「定义不清」，而不只是「没测」

- `[SUPPORTED]` block-mask 规范要求 canonical encoding + manifest anchoring + HKDF-SHA256 + ChaCha20 + 无偏 Fisher–Yates，并**明文禁止 Python `random`**；算法文档却用 **Python `random` + 两级 SHA256 + 不同分组 + 无 manifest anchor 记录**。
- `[SUPPORTED]` **第 2 层还有第二处独立分歧**：SecAgg 规范写 ChaCha20，而 active 执行计划记载 pairwise/self mask **当前实际用 SHAKE-256**，把换成 AES-CTR/ChaCha 列为未来工作（`2026-09-18-secagg-time-efficiency.md:52`，PERF-7）。**本报告已核对原文，确认成立。**
- `[UNKNOWN]` 因此**目前根本不存在一个"规范化的排列"可供跨语言测试**。可重放性主张必须收窄到"在某一指定实现变体下可重放"。

### D5. 证据面太窄，支撑不了一般化的收敛主张

- `[PARTIAL]` 全部实质性 SecAgg 精度证据 = **1 个模型、2 个 client、IID 50/50、1 个种子、20 轮、1 个上传比例**。
- `[UNKNOWN]` 无更大 cohort、无不等权重、无非 IID、无第二个模型、无更长训练、无 run-to-run 方差、无统计等价性证据。
- `[PARTIAL]` "0.985 vs 0.987" 是**观测到的终点差**，不是"INT16 等价于 fp16"或"两条学习轨迹可互换"的证明。

### D6. 安全叙述低估了实际可观测的 transcript

- `[SUPPORTED]` 服务器拿到的**远不止** masked 向量与 `global_amax`：文档自己列出了 per-client train/eval loss、`num_examples`、timings、以及 **per-block `block_energies`**（`2026-09-15-...md:211`，§3.4.1）。
- `[PARTIAL]` 因此 **"server 只看到聚合结果"只能作为"受保护更新向量"在给定假设下的简写**；作为对整个系统 transcript 的描述，**它是错的**。
- `[UNKNOWN]` **没有任何语义泄露分析**覆盖 `global_amax` + losses + 参与信息 + timings + block energies 的**联合披露**。

### D7. 系统贡献缺少它最关键的扩展性结果

- `[PARTIAL]` 5 轮只支持"一个 workload 上的局部延迟改进"。
- `[UNKNOWN]` 计划的核心主张 —— **"window 数增长时墙钟跟字节/计算走、而非 `RTT × window_count`"** —— **没有被测量**。
- `[UNKNOWN]` cohort 规模的扩展还额外受 **all-to-all masking** 与 **D3 的 modulus 边界**双重约束。

### D8. "全训优于 FedRolex" 的证据超出本轮授权证据集

- `[SUPPORTED]` 方案确实"训全模型、传部分 block"。
- `[UNKNOWN]` 但该比较依赖 `2026-09-04-fedrolex-partial-training.md`，它属于任务明示的**「对照、不进主路径」**材料，**不在本方案的十一份证据源之内**。**因此本评审中不能把它当作受支持的比较结果使用。**（此点同时修正了 `CURRENT_VS_LITERATURE.md` §1 与 §5 的用法。）

---

## 3. 对审前自评三项的裁决（含独立核验）

| 审前自评 | 裁决 | 说明 |
|---|---|---|
| `global_amax` 相对 1912.00131 的 autotune 是隐私退步 | **`PARTIAL` — 方向正确，强度被夸大** | 确实把每个 client 的旋转域 L∞ 额外暴露给服务器（父本从**安全聚合结果**推断调参信息），**在 transcript 意义上确属退步**。但**这并非被量化的隐私退化**：文档自评"低–中"**没有任何 DP / 互信息 / 重构 / 攻击分析支撑**。且系统还暴露更丰富的元数据，**把 `global_amax` 当成唯一隐私问题是片面的** |
| SESA (ISIT 2024) 是未被回应的威胁 | **`PARTIAL` — 是"缺失的对照/证明义务"，而非对本设计的现实攻击** | SESA 的索引信道来自**客户端各自**的子模型/更新坐标选择。本方案要求 **cohort 全体共用同一公开、数据无关的坐标 mask**，并要求 **全窗统一 survivor 集**；规范甚至写明了这正是为避免 private sparse-index 泄露。**在这些假设下该特定索引信道看起来已被结构性消除**。真正缺的是**形式化的 transcript 论证、对 SESA 的引用、以及对 per-block 元数据的处理** —— 而不一定需要新的索引隐藏机制 |
| PRG 分歧使跨实现可重放主张失效 | **`SUPPORTED` — 自评正确** | ChaCha20/HKDF 构造与 Python `random`/SHA256 构造**无法共同支撑同一个跨实现可重放主张**。在统一口径并跑通测试向量之前，超出"已评测实现"的可重放性是 `UNKNOWN` |

---

## 4. 文献关系类型的修正（评审对 `AI_DESIGN_RELATED_WORK.md` 的裁定）

| 条目 | 原类型 | **修正后** | 等级 |
|---|---|---|---|
| A1 FedRolex | ancestor | **closest neighbor / baseline**（本方案未记载派生自它，且解决的是不同问题） | `PARTIAL` |
| A2 FjORD / A3 HeteroFL / A4 EMBRACE / A7 Federated Dropout | ancestor | **baselines / related predecessors** | `SUPPORTED` |
| **A9 Stich, Cordonnier, Jaggi, *Sparsified SGD with Memory* (arXiv:1809.07599, 2018)** | **缺失** | **应补为最亲近的机制祖先之一（`block_memory` 属此家族）** | `SUPPORTED` |
| A5 Barbieri / A6 FedNILO / A8 FedMask | neighbor | **不能承担强定位权重**（原文自标 title-level / 未核验） | `UNKNOWN` |
| B1 Bonawitz CCS 2017 | ancestor | **ancestor（维持）** | `SUPPORTED` |
| B2 SecAgg+ / B3 LightSecAgg / B4 Turbo-Aggregate | 已有机制 | **更强替代基线 / 既有设计**（非本方案所用机制） | `SUPPORTED` |
| B5 去中心化 dropout 鲁棒聚合 | neighbor | **边缘**（信任与拓扑假设不同） | `PARTIAL` |
| B6 / B7 SESA / B8 X-Secure T-Private FSL | neighbor | **正确（维持）** | `SUPPORTED` |
| C1 Suresh / C2 1912.00131 / C3 QSGD / C5+C7 EF | ancestor | **正确（维持）** | `SUPPORTED` |
| C4 RATQ | 支持性结论 | **理论基线 / 祖先** | `PARTIAL` |
| C6 EF21 | 上位替代 | **替代性压缩设计**，非"严格更优的替代" | `PARTIAL` |
| C8 ScionFL | 本方案全局更弱 | **竞争基线**；说本方案"全局更弱"**不成立** —— 它在自己的设置下量化比特率更强，但**表示、MPC 机制、任务、威胁模型、总坐标传输量都不同** | `PARTIAL` |
| C11 FP8 / SCAFFOLD | neighbor | **语境引用，非紧密方法近邻** | `PARTIAL` |
| D1–D4 DiLoCo 家族 | 正交 | **正确（维持）** | `SUPPORTED` |
| D6 SHE-LoRA | 应用路线 | **竞争性应用路线，非祖先** | `PARTIAL` |
| D5 SecLoRA / D7–D8 FedShield-LLM, FLAGuard | frontier | **不足以支撑 2025–2026 frontier 主张**（原文自标标识符不完整/未核验） | `UNKNOWN` |

---

## 5. Claim 分级

### 5.1 可以写进论文（附最强可用措辞）

| # | Claim | 等级 | 措辞 |
|---|---|---|---|
| P1 | block 调度与覆盖 | `SUPPORTED` | *"In a documented two-client Qwen2.5-0.5B experiment with H=5, canonical approximately 1 MB block partitioning and position-mod-H scheduling covered every rotating block exactly once per five-round epoch while keeping per-round selected bf16 volume between 18.5% and 21.5%."* |
| P2 | 端到端数值 | `PARTIAL` | *"For one Qwen2.5-0.5B, two-client, IID 50/50, single-seed configuration with `coverage_h=10`, the documented 20-round Hadamard–INT16 SecAgg run reached endpoint eval loss 0.985, compared with 0.987 for the corresponding fp16 non-SecAgg baseline."* |
| P3 | 传输/执行优化 | `PARTIAL` | *"In a documented five-round comparison at the same configuration, single-blob upload, deferred object fetch, parallel unmasking, and asynchronous full-state persistence preserved the recorded per-round eval values and reduced round time to 110–134 seconds."* |
| P4 | 协议设计 | `SPEC_ONLY` | *"The written protocol specifies a Bonawitz-style pairwise-plus-self-mask secure aggregation state machine with attempt-domain separation, a frozen all-window survivor set, threshold recovery, and a no-double-reconstruction invariant; the threshold recovery portions have not yet been validated in the documented execution record."* |
| P5 | 索引信道 | `SPEC_ONLY` | *"The protocol requires a public, cohort-uniform coordinate plan chosen independently of the clients' current private updates, which is intended to eliminate client-specific sparse-index disclosure; this property has not received a formal transcript-level analysis."* |

> **措辞纪律**：P2 必须写成**观测**（"0.985 vs 0.987 in one documented run"），**不得**写成 "equivalent" / "lossless" / "general"。
> **指标纪律**：selected-coordinate ratio、wire bytes、communication frequency、total training communication **必须作为四个分离指标**分别报告。

### 5.2 只能降级为「工程实现」或「已有方法组合」

| 原表述 | **降级为** | 等级 |
|---|---|---|
| 新颖的部分训练 / 均匀覆盖算法 | **确定性、block 粒度的通信调度**，叠加已有稀疏化/memory 思想 | `SUPPORTED` |
| 新颖的 SecAgg | **把 Bonawitz 式 SecAgg 适配到一个确定性 window plan** | `SUPPORTED` |
| Hadamard+INT16 贡献 | **对已有旋转量化的实现，外加一个更简单但更暴露的 scale 选择规则** | `SUPPORTED` |
| 选块/SecAgg 解耦 | **一个有用的协议工程不变量**，非算法新颖性 | `SUPPORTED` |
| 对象存储 / HTTP 控制面 / SAW1 单 blob / 并行 unmask / 异步 finalize | **系统工程** | `SUPPORTED` |
| 接近 fp16 的收敛 | **单一配置下的单跑次终点对齐** | `PARTIAL` |
| 近似均匀带宽 | 限于**实测的 selected bf16 体积**；**不得**暗示在 INT16 padding、容器头、控制流量之后仍存在精确的端到端 wire-byte 比例 | `PARTIAL` |
| 可重放 mask | **在指定实现变体下可重放** | `UNKNOWN` |

### 5.3 目前完全不能声称

| Claim | 等级 |
|---|---|
| 跨语言 / 跨实现 mask 可重放 | `UNKNOWN` |
| dropout 鲁棒性、阈值安全、禁止双重重构的强制执行、失败后的安全重试 | `SPEC_ONLY` |
| 已部署路径的形式化 aggregate-only 机密性保证 | `SPEC_ONLY` |
| 抗服务器–客户端合谋 / 恶意客户端 / Byzantine / 流量分析 / 侧信道 | `UNKNOWN` |
| 对 SESA 式攻击的无条件索引隐私保证（尤其当 `block_energies` 等元数据仍在 transcript 中） | `UNKNOWN` |
| 以所述 `q`、`Q_max` 下的通用 **N>2** INT16 聚合正确性 | `UNKNOWN` |
| 不等样本权重的整数域加权 FedAvg 正确性 | `UNKNOWN` |
| 与 fp16 的统计等价、模型无关收敛、非 IID 鲁棒性、多种子可复现 | `UNKNOWN` |
| window 数 / cohort 规模 / 大模型扩展性 | `UNKNOWN` |
| 总通信量低于 DiLoCo 家族或 LoRA+HE/FE 系统 | `UNKNOWN` |
| 基于本轮证据集的对 FedRolex 的实证优越性 | `UNKNOWN` |
| 新密码学 / 新 Hadamard 量化 / 新通用 error-feedback 方法 | `SUPPORTED`（即：**不可声称**） |

---

## 6. 最小补强项（**仅建议，本轮不执行**）

### 6.1 Protocol

| # | 建议 | 等级 |
|---|---|---|
| R1 | **选定唯一的权威 block-mask 构造**（分组、种子输入、anchor 语义、编码、PRG、拒绝采样），**废止冲突的书面变体**，并发布固定的跨语言测试向量 | `SUPPORTED` |
| R2 | **把已评测的 SecAgg 子集与完整 dropout 恢复规范分别标版本**。要么后续实现并验证完整状态机，要么明确把 Shamir / `U*` / 禁止双重重构**标注为未来协议设计** | `SUPPORTED` |
| R3 | **冻结算术契约**：最大 cohort 规模、聚合权重、权重编码、`q`、`Q_max`、有符号解码、强制 overflow 拒绝。**当前 16-bit 参数在改变该契约之前不得声称支持 2 个以上客户端** | `SUPPORTED` |
| R4 | **定义完整的 server-visible transcript**（含 `global_amax`、losses、timings、参与信息、`block_energies`），并给出**明确的 SESA 对照**。把"公开公共支撑集"表述为**索引信道**的缓解，**不要暗示它消除了其他元数据泄露** | `PARTIAL` |
| R5 | 要么**把 per-client `global_amax` 换成聚合导出或安全聚合的调参**，要么**显式声明系统为简单与精度接受这一额外披露** | `PARTIAL` |

### 6.2 未来实验（**本轮不执行**）

| # | 建议 | 等级 |
|---|---|---|
| R6 | **最小的 SecAgg 一致性验证集**：一个足以形成非平凡阈值的 cohort，覆盖两种指定的 dropout 位置、阈值失败、survivor 冻结、禁止的双重重构、以及 fresh-attempt 重试 | `SPEC_ONLY` |
| R7 | **最小的系统扩展性验证集**：变化 window 数与 cohort 规模，足以检验"非 `RTT×N`"主张与 all-to-all setup 开销 | `UNKNOWN` |
| R8 | **最小的学习证据扩充**：增加重复种子 + 一个不等分/非 IID 配置。**在此之前，措辞必须严格限定在特定配置** | `PARTIAL` |

### 6.3 Wording

| # | 建议 | 等级 |
|---|---|---|
| R9 | 标题与摘要使用 **"engineering integration" / "system integration"**，**不得**使用 "new algorithm" / "new secure aggregation protocol" / "novel Hadamard quantization" | `SUPPORTED` |
| R10 | 把 selected-coordinate ratio、wire bytes、communication frequency、total training communication **作为四个分离指标**保留 | `SUPPORTED` |
| R11 | 端点观测写为观测，而非 "equivalent" / "lossless" / "general" | `PARTIAL` |

---

## 7. 残余不确定性

- `[UNKNOWN]` Run ID 只是文档级引用。按本轮边界，评审接受其书面形式，未要求原始目录。
- `[UNKNOWN]` 优化路径的 **20 轮验证仍在进行中**；文档只记载了它的 5 轮数值一致性。
- `[UNKNOWN]` 代码不在本轮范围内，因此**规范文档与执行文档之间的冲突无法裁定谁为准**。
- `[UNKNOWN]` SecLoRA、FedShield-LLM、FLAGuard、FedNILO、Barbieri et al.、FedMask 在本轮材料中**标识符不完整或未核验**，基于它们的优先级/frontier 主张应予保留。
- `[UNKNOWN]` **`block_energies`、per-client losses 等辅助字段究竟是有意为之的最终研究协议的一部分，还是附带的 telemetry，尚不清楚**。这一区分**实质影响任何端到端隐私陈述**。
- `[UNKNOWN]` 生产计划存在**状态措辞不一致**（部分安全项在看板里标 done，而它的详细日志说是 design-only 且默认关闭）。**未来论文不能把计划的状态标签当作证据**，除非先确立"权威版本"规则。

---

## 8. 与 §3/§4 产物的一致性说明

本报告**接受**评审对 `AI_DESIGN_RELATED_WORK.md` 与 `CURRENT_VS_LITERATURE.md` 的以下修正，并已在两份文件中追加 **post-review Errata** 段落（保留审前原文以维持审计链）：

1. FedRolex 由 ancestor 改判为 **closest neighbor / baseline**（§4 表）。
2. 补入缺失的近邻家族 **Sparsified SGD with Memory（arXiv:1809.07599）**，并指出 `block_memory` 属该家族、其 decay=0.9 是非标准启发式（§4 表 + D1）。
3. **D8**：`CURRENT_VS_LITERATURE.md` §1 与 §5 中"全训优于 FedRolex（1.1190 vs 1.1829）"的用法**超出授权证据集**，不得作为受支持的比较结果使用。
4. **D3** 是审前产物未发现的新结论，为纯增量。
5. ScionFL 的"本方案全局更弱"表述改为"竞争基线"（§4 表）。

---

## 9. 停止条件达成情况

本轮 §6 要求的五个文件已全部生成：

| 文件 | 状态 |
|---|---|
| `idea-stage/CURRENT_METHOD.md` | ✅ |
| `idea-stage/AI_DESIGN_RELATED_WORK.md` | ✅（含 post-review Errata） |
| `idea-stage/CURRENT_VS_LITERATURE.md` | ✅（含 post-review Errata） |
| `idea-stage/RESEARCH_REVIEW_REQUEST.md` | ✅ |
| `idea-stage/AI_DESIGN_REVIEW.md` | ✅（本文件） |

**未进入** `/aris:idea-discovery`、experiment-plan、experiment-bridge、training 或任何代码实现流程。**未执行任何新实验或代码修改。**
