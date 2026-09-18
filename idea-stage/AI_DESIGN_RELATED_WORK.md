# AI_DESIGN_RELATED_WORK — 文献定位（FedScale `deploy/central-server-minio`）

- **生成**：2026-09-18
- **上游输入**：`idea-stage/CURRENT_METHOD.md`（待比较的书面方案）
- **方法**：`/aris:research-lit "Public hierarchical permutation-and-rotation block mask for federated LLM updates; windowed secure aggregation with pairwise+self masks, dropout recovery, and Hadamard+INT16 quantization for cross-silo FL" — sources: web, semantic-scholar — arxiv download: false`
- **定位纪律**：本文不是普通综述。每篇给出与当前方案的**关系类型**（ancestor / closest neighbor / baseline / 已有机制 / 工程组合 / 差异点），并明确指出**当前方案还剩什么**。

---

## 0. 检索执行情况（含降级说明）

| 请求的 source | 状态 | 说明 |
|---|---|---|
| `web` (WebSearch) | ✅ **contributed** | 12 次检索，覆盖全部 5 个方向 |
| `arxiv` (helper `arxiv_fetch.py`) | ⚠️ **helper 失败** | `HTTP Error 406 Not Acceptable`（helper 走 http→301 后被拒）。**已改用 `https://export.arxiv.org/api/query` 直连 curl 完成校验**，23/23 ID 全部命中 |
| `semantic-scholar` (helper) | ❌ **failed** | 持续 `HTTP 429 Too Many Requests`（无 API key）。**未贡献** |
| CrossRef（helper 的第 2 层，直连补充） | ✅ contributed | 补验 4 篇无 arXiv ID 的会议/期刊论文 |
| zotero / obsidian / local | 跳过 | 未配置 / 未请求 / 无本地 PDF |

> **WARN (local)**: local contributed nothing — no PDFs found in `papers/`, `literature/`, or a configured paper library. To include yours, add a "## Paper Library" heading to CLAUDE.md followed by the directory path.
>
> **WARN (semantic-scholar)**: 请求了但不可用（HTTP 429）。venue 元数据改由 CrossRef 补齐。venue-only 论文（IEEE/ACM 非 arXiv）覆盖因此**可能不完整**。

**贡献 source 列表**：`web`, `crossref`（+ 通过直连 curl 完成的 arXiv 校验）。

**反幻觉校验（Step 1.5）**：25 个候选 → arXiv 直连校验 23 命中；CrossRef 校验 4 命中；**1 个错误 ID 被拦截**：
- `2011.00241` 我原以为是 HeteroFL，实测为 *"Methods for Pruning Deep Neural Networks"*。经 arXiv 标题检索更正为 **HeteroFL = arXiv:2010.01264**。

---

## 1. 总览：当前方案四层 vs 文献的最邻关系

| 当前方案的层 | 最近的文献 | 关系类型 | 剩余差异化空间 |
|---|---|---|---|
| **① 选块**：公开 block permutation + 无放回轮转 | FedRolex (2212.01548)、FedNILO、Barbieri et al. layer-selection | **closest neighbor + ancestor** | 窄：block 粒度均匀带宽 + 可重放 mask_hash + 与 SecAgg 坐标冻结耦合 |
| **② 安全聚合**：pairwise + self + dropout recovery | Bonawitz CCS 2017 (1611.04482 / ePrint 2017/281) | **ancestor（直接父本）** | **几乎为零**（密码学层面无新增） |
| **②′ 子模型 × SecAgg 的索引隐私** | **SESA (ISIT 2024)**、X-Secure T-Private FSL (ICC 2021)、arXiv:2111.01432 | **closest neighbor（尚未在方案中被讨论）** | 需要正面回应，否则主张不成立 |
| **③ 数值路径**：Hadamard + INT16 + global_amax | **Bonawitz et al. 1912.00131**、Suresh 1611.00429 | **ancestor（直接父本）** | 窄；且当前设计在隐私上**弱于**父本 |
| **③′ error feedback / memory 拆分** | EF 1901.09847、EF21 2106.05203、EF Framework 1909.05350 | **已有机制（教科书级）** | 无 |
| **④ 传输/执行**：对象存储 + 单 blob + 并行 unmask | DiLoCo 系列、SecLoRA / SHE-LoRA（LLM FL 系统） | **工程组合 / 竞争应用路线** | 系统贡献可写，但非算法 |

**一句话**：当前方案在**算法原理层面没有新的祖先**可言 —— 选块对应 partial-training 家族，安全聚合对应 Bonawitz，数值路径对应 Bonawitz-2019 + error-feedback。**真正还没被文献占掉的位置，是「公开、可重放、跨实现一致、与 SecAgg 坐标计划冻结耦合的 block 级带宽预算机制」这一工程/系统定位**，以及它在真实 0.5B LLM 上 10% 上传率端到端可收敛的**实证**。

---

## 2. 方向 A：partial / submodel / block-wise training 与部分上传

### A.1 文献表

| # | 论文 | Venue | 核心 | 与当前方案的关系 |
|---|---|---|---|---|
| A1 | **FedRolex**: Model-Heterogeneous FL with Rolling Sub-Model Extraction — Alam, Liu, Yan, Zhang | NeurIPS 2022 · [arXiv:2212.01548](https://arxiv.org/abs/2212.01548) ✅ | rolling sub-model extraction，让全局模型各部分**被均匀训练**，缓解 client drift；理论优于 Federated Dropout | **closest neighbor + ancestor**。同样"轮流覆盖全模型各部分"的思想；但 FedRolex 是**层/宽度粒度**、服务于 **model-heterogeneous**、客户端**只训子模型**（省算力） |
| A2 | **FjORD / Ordered Dropout** | NeurIPS 2021 · [arXiv:2102.13451](https://arxiv.org/abs/2102.13451) ✅ | nested 结构化丢参，per-layer dropout rate 采样；通信节省 ≈ 1/p²；自蒸馏 | ancestor（nested submodel 家族）。**不要求 nested**，也不改训练范围 |
| A3 | **HeteroFL** | ICLR 2021 · [arXiv:2010.01264](https://arxiv.org/abs/2010.01264) ✅ | width/depth reduction，异构客户端训练不同规模子模型 | baseline / ancestor |
| A4 | **EMBRACE / EmbracingFL**: Enabling Weak Client Participation via Partial Model Training | 2024 · [arXiv:2406.15125](https://arxiv.org/abs/2406.15125) ✅ | 弱客户端部分模型训练，报告优于 HeteroFL / FjORD 的 width-reduction | recent neighbor |
| A5 | **Layer Selection Optimizer for Communication-Efficient Decentralized FL** — Barbieri et al. | Politecnico di Milano | **Policy #2：coordinator 约束所有设备选同一批层** | **最贴近"cohort-wide 公共选层"的邻居** |
| A6 | **FedNILO**（freeze index 调度） | 学位论文 | `I_o(r,K,F)=⌈(r−K)/F⌉`，逐步冻结前段层，只同步未冻结层；每端每轮仅多传 16-bit 层索引 | **公共、确定性、周期性调度**的最近先例（但为单调冻结，非无放回轮转） |
| A7 | Federated Dropout | — | 每轮随机抽子模型训练 | baseline（FedRolex 的比较对象） |
| A8 | FedMask / FedLayerPrune | — | **可学习**的 mask（SGD 优化后阈值化）/ 层敏感度 + mask 共识投票 | 差异点：当前方案的 mask **不可学习、公开、与私有更新无关** |

### A.2 判断：Public Block Mask 相比 FedRolex / federated dropout 真正新增了什么？

**已被覆盖的部分（不能声称新颖）**：
1. "让全局模型各部分在若干轮内被均匀覆盖" —— FedRolex 的核心主张就是这个，且是 NeurIPS 2022 的正式贡献。
2. "选择规则公开且确定性" —— FedRolex 的 rolling schedule、FedNILO 的 freeze index、FjORD 的 ordered dropout **同样都是公开确定的**，不含私有随机性。因此"公开/确定性"本身**不构成差异化**。
3. "省上传带宽" —— partial-training 家族的共同目标。

**仍有差异的部分（可作为工程/系统主张）**：
1. **粒度与均匀性**：当前方案在**扁平化后的 canonical block**（≈1MB）上做选择，直接优化"每轮上传字节数近恒定"这一**传输预算**目标；FedRolex/FjORD 的层/宽度粒度会造成逐轮字节数随层的参数规模波动（本仓库自己的 S3R12v2 → S3R12v3 记录正是这个问题：key 级 7.3%~37.4% → block 级 18.5%~21.5%）。**这是一个明确、可测量的工程差异化**。
2. **训练范围与传输范围解耦**：方案明确"ICC 仍可对完整本地模型执行训练，然后仅导出 mask 选中的 block 更新"（S-BM §1）。FedRolex 是"只训部分层"，本仓库的 FedRolex 对照（R20=1.1829）也验证了二者收敛差距（S3R11 全训只传部分 = 1.1190 更优）。**"全训 + 部分传 + memory 补偿"是当前方案相对 FedRolex 的实质取舍，且已有本仓库内对照数据**。
3. **可重放性与审计性**：seed + layout → 所有 ICC 独立重算同一 `mask_hash`，并生成跨语言 test vector。文献中的 rolling schedule 多为"按轮次编号索引"（隐式可重放），但**显式冻结 seed/layout/manifest anchor 的审计协议**在 FL 文献里不是常见关注点。这一条只有在**规范真正落地并跨实现验证**后才成立（当前 UNKNOWN，见 §5）。
4. **与 SecAgg 坐标计划的耦合**：mask 决定**进入 SecAgg 的坐标集合与顺序**，因此必须在密码学层之前冻结且 cohort 内完全一致。SESA（§3.3）指出这恰恰是 submodel + SecAgg 的**危险交界**。

**结论**：Public Block Mask 相对 FedRolex/federated dropout 的**真正新增是"block 粒度均匀带宽预算 + 与训练范围解耦 + 显式可重放审计协议"**，属于**机制组合与系统工程**，不是新的部分训练原理。若写成"我们提出了均匀覆盖的部分训练方法"会被 FedRolex 直接驳回。

---

## 3. 方向 B：安全聚合与 dropout recovery

### B.1 文献表

| # | 论文 | Venue | 核心 | 与当前方案的关系 |
|---|---|---|---|---|
| B1 | **Practical Secure Aggregation** — Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth | ACM CCS 2017 · [arXiv:1611.04482](https://arxiv.org/abs/1611.04482) ✅ · ePrint 2017/281 | DH pairwise masks + self mask + **Shamir t-of-n 分享 `sk_i` 与 `b_i`**；服务器恢复 survivor 的 `b_i`、dropout 的 pairwise secret；**"either/or, not both"** 约束；O(N²) | **ancestor（直接父本）** |
| B2 | **SecAgg+** — Bell, Bonawitz, Gascón, Lepoint, Raykova | ACM CCS 2020 · [doi:10.1145/3372297.3417885](https://doi.org/10.1145/3372297.3417885) ✅ | k-regular 图替代 all-to-all，O(N log N) | **已有机制（更优的可扩展设计）**；当前方案 baseline 仍是 all-to-all |
| B3 | **LightSecAgg** — So et al. | MLSys 2022 · [arXiv:2109.14236](https://arxiv.org/abs/2109.14236) ✅ | 对 survivor 做**一次性 aggregate-mask 重构**；O(d)/user；支持异步 FL | 已有机制；比当前方案的 per-dropout 重构更优 |
| B4 | **Turbo-Aggregate** — So, Guler, Avestimehr | IEEE JSAIT 2021 · [doi:10.1109/jsait.2021.3054610](https://doi.org/10.1109/jsait.2021.3054610) ✅ | O(N log N) + Lagrange 编码，容忍 50% dropout | 已有机制 |
| B5 | **Privacy-Preserving, Dropout-Resilient Aggregation in Decentralized Learning** | 2024 · [arXiv:2404.17984](https://arxiv.org/abs/2404.17984) ✅ | 去中心化场景的 dropout 鲁棒聚合 | recent neighbor |
| B6 | **Practical and Light-weight Secure Aggregation for Federated Submodel Learning** | 2021 · [arXiv:2111.01432](https://arxiv.org/abs/2111.01432) ✅ | cuckoo hashing + DPF，让客户端** oblivious** 指示服务器聚合其更新 | **layer ①×② 交界的邻居** |
| B7 | **SESA — Secure Submodel Aggregation for Resource-Aware FL** | IEEE ISIT 2024 · [doi:10.1109/isit57864.2024.10619575](https://doi.org/10.1109/isit57864.2024.10619575) ✅ | 证明**朴素地对异构 submodel 做参数级 SA，服务器仍可重构诚实用户的训练样本**；因此同时隐藏**被训练的子模型**与**被更新参数的索引**；honest-but-curious，容忍 T 个与服务器合谋的用户 | **最重要的 closest neighbor（方案目前完全未讨论）** |
| B8 | **X-Secure T-Private Federated Submodel Learning** | IEEE ICC 2021 · doi:10.1109/icc42927.2021.9500855 | 子模型学习的 T-私密保护 | 同一交界 |

### B.2 判断：2-client、无真实 dropout 的验证能支撑多强的 SecAgg 安全主张？

**先说结论：非常弱，且不足以支撑任何"dropout 鲁棒"或"阈值安全"的主张。**

支撑依据：

1. **协议层面对齐的是 Bonawitz 2017 的完整机制**。当前规范 §7 的两类恢复秘密（pairwise-mask recovery secret + `self_master`）、t_rec-of-n 分享、survivor 冻结、以及 §7.3.3「禁止双重重构」，**逐条对应 Bonawitz 的 `sk_i` / `b_i` 双分享与 "either/or, not both" 约束**。因此：
   - **密码学层面没有新增**（规范自己承认：S-SA「实现原则」明确"本文不把这些密码学原语本身作为论文创新"）；
   - 任何"我们提出 dropout-resilient SecAgg"的写法都会被 Bonawitz 2017 直接覆盖。
2. **已实验验证的部分只有协议最平凡的一段**：
   - pairwise mask 无 dropout 时精确对消（2 client，单元测试 + 端到端）；
   - self mask 由 server 移除；
   - 这恰恰是 Bonawitz 协议里**不需要秘密分享**就能工作的情形（N−D > T 且 D=0）。
3. **Shamir/阈值/dropout recovery 全部停留在规范**：`CURRENT_METHOD.md` §3.3 已核验 —— A-PREC / A-SURV / A-FLOW 三份执行文档**均未记录**阈值分享实现、U\* 冻结、attempt_id fresh-mask 重试、或 S-SA §16.2 的 5 项 dropout 故障注入中的任何一项。
4. **2-client 的安全性下限本身很低**：附录 A 记 `q_min ≥ 2`。在 N=2 时，"隐私阈值 T" 与 "掉线容忍 D" 满足 `N − D > T` 的空间几乎为零 —— T=0 时 D=1 已触界。**2-client 实验无法演示任何有意义的 (T, D) 权衡**。
5. **必须区分的两条线**（任务显式要求）：SYNC 层（P-PROD SYNC-1/SYNC-2 的 `min_clients_to_aggregate`、迟到 409、缺席 inactive）是 **FL 调度层**的部分参与；它**不是** Shamir/threshold dropout recovery。执行文档未建立二者等价关系，因此不能把"部分参与已 done"读成"dropout recovery 已验证"。
6. **额外的负面证据（SESA, B7）**：在 submodel FL 中，**参数级 SecAgg 不足以保证索引隐私** —— 服务器可能借被更新坐标的位置反推客户端数据。当前方案的 Public Block Mask 是**公开且 cohort 一致**的，所以单看"索引"确实不泄露客户端私有信息（这是当前设计的一个真实优点）。**但方案文档从未论证这一点，也从未与 SESA 一类的攻击模型对照**。这是一个必须补上的论证缺口，而不是可以默认成立的结论。

**可以支撑的最强主张**（措辞上限）：
> "在一个 honest-but-curious 服务器、2 个已认证且不串谋的参与方、无掉线的设置下，服务器仅观察到 pairwise+self mask 后的整数向量与一个全局 amax 标量，无法分离出任一方的未掩码更新。"

**不能支撑的主张**：dropout 鲁棒、阈值安全、抗合谋、N>2 的可扩展性、对恶意/投毒客户端的鲁棒性。

---

## 4. 方向 C：Hadamard + 量化 + SecAgg，以及 error feedback

### C.1 文献表

| # | 论文 | Venue | 核心 | 与当前方案的关系 |
|---|---|---|---|---|
| C1 | **Distributed Mean Estimation with Limited Communication** — Suresh, Yu, Kumar, McMahan | ICML 2017 · [arXiv:1611.00429](https://arxiv.org/abs/1611.00429) ✅ | **structured random rotation 把 MSE 从 Θ(d/n) 降到 O((log d)/n)**，即 O(d/log d) 改善；固定长度编码可与加密/SecAgg 组合 | **ancestor（旋转思想的源头）** |
| C2 | **Federated Learning with Autotuned Communication-Efficient Secure Aggregation** — Bonawitz, Salehi, Konečný, McMahan, Gruteser | Asilomar 2019 · [arXiv:1912.00131](https://arxiv.org/abs/1912.00131) ✅ | **R = HD（Walsh–Hadamard × 随机符号对角阵）→ 量化 → mod k**；利用旋转后近似正态 + SecAgg 模回绕得到 **wrapped normal**，由**聚合结果**拟合 σ 并反解 bin size 实现 autotuning；k=2⁸ | **closest neighbor + ancestor（与当前数值路径几乎逐项对应）** |
| C3 | **QSGD** — Alistarh et al. | NeurIPS 2017 · [arXiv:1610.02132](https://arxiv.org/abs/1610.02132) ✅ | per-coordinate 随机均匀量化 + 归一化；传 scale | ancestor |
| C4 | **RATQ** — Mayekar & Tyagi | 2019 · [arXiv:1908.08200](https://arxiv.org/abs/1908.08200) ✅ | 旋转 + 自适应均匀量化近信息论最优 | **支持性结论**（为"旋转 + 均匀量化"背书） |
| C5 | **Error Feedback Fixes SignSGD…** — Karimireddy et al. | ICML 2019 · [arXiv:1901.09847](https://arxiv.org/abs/1901.09847) ✅ | `e_t = g_t − Q(g_t + e_{t−1})`；EF 让 contractive 压缩器达到 O(1/√T) | **已有机制（当前 `quant_residual` 就是它）** |
| C6 | **EF21** — Richtárik et al. | NeurIPS 2021 · [arXiv:2106.05203](https://arxiv.org/abs/2106.05203) ✅ | 压缩"与 memory 的差"，动态范围天然小，无需旋转 | 已有机制；**当前方案的上位替代** |
| C7 | **The Error-Feedback Framework** — Stich & Karimireddy | 2019 · [arXiv:1909.05350](https://arxiv.org/abs/1909.05350) ✅ | EF 统一框架 | 已有机制 |
| C8 | **ScionFL** — Ben-Itzhak et al. | SaTML 2024 · [arXiv:2210.07376](https://arxiv.org/abs/2210.07376) ✅ | MPC-SecAgg + 1-bit 线性量化 + **Kashin 表示**（ℓ∞ bound O(1/√d) 优于 Hadamard 的 O(√(log d)/√d)）+ 抗投毒 ScionFL-Aura | **SOTA 竞争点**；当前 INT16 Hadamard 在精度/比特轴上弱于它 |
| C9 | **FO-SGD**（flattened one-bit SGD） | 2024 · [arXiv:2405.11095](https://arxiv.org/abs/2405.11095) ✅ | Hadamard + 1-bit 压缩分布式优化 | neighbor |
| C10 | **HLQ** | 2024 · [arXiv:2406.15102](https://arxiv.org/abs/2406.15102) ✅ | Hadamard 低秩 4-bit 训练 | neighbor |
| C11 | **FP8 Formats for Deep Learning** | 2022 · [arXiv:2209.05433](https://arxiv.org/abs/2209.05433) ✅ | E4M3/E5M2 双格式 + loss scaling | neighbor（量化工况参考） |

### C.2 判断：Hadamard + INT16 相比已有 Hadamard-SecAgg 还剩什么贡献？

**几乎不剩方法层面的贡献。** 逐项对照 C2（1912.00131）：

| 维度 | 文献 C2 | 当前方案 | 差异性质 |
|---|---|---|---|
| 旋转矩阵 | R = HD，server 下发，O(d log d) 应用 | pad-p2 → 符号翻转 D → FWHT | **相同** |
| 随机性来源 | 随机符号对角阵 D | 每 window 符号翻转 D | **相同** |
| 量化 | 均匀量化 + mod k | INT16 均匀 + stochastic rounding | **相同**（位数 2⁸ vs 2¹⁶） |
| **scale 选择** | **利用旋转后 wrapped normal，由聚合结果拟合 σ 反解 bin size（autotune）** | **每 client 上报 1 个 `global_amax`（旋转域 L∞），server 取 max × 1.05** | **当前方案是简化替代，且隐私更差**（见下） |
| SecAgg 兼容 | 有 | 有 | 相同 |
| error feedback | 未强调 | `block_memory`(0.9) + `quant_residual`(1.0) | **来自 C5–C7，非新增** |

**两条关键判断：**

1. **`global_amax` 在隐私上严格劣于父本 C2。** C2 的 autotuning 从**聚合结果**推断尺度分布，**不需要任何单客户端标量**；当前方案要求每个 client 每轮上报自己的旋转域 L∞，即**把单客户端的更新幅度直接交给服务器**（A-PREC §3.4.1 泄露项 #1，文档自评"低–中"风险）。当前方案相对旧 per-window 路径（269 个标量）确有改善，但**相对 2019 年的父本是一条退步**。审稿人若熟悉 1912.00131，这是最容易被直接质疑的一点。
   - 方案文档里其实已有更强的退路（A-PREC §3.4.2：`update_public_block_scales`，公开聚合 max），但被标注"有一轮滞后且精度较差（曾导致 R2 eval 上升）"而搁置。**这正是应当被重估的取舍**。

2. **INT16 vs 1-bit/Kashin 的方向相反。** C8 (ScionFL) 用 1-bit + Kashin 表示在 MNIST/LeNet 上把通信从 16.14 GB 压到 0.94 GB（17.2×）且精度仅 99.04%→98.71%。当前方案是**升位宽换精度**（INT16），因此它的"通信效率"竞争力完全来自**选块层**（只传 10–20% 的参数），而非量化层。论文里不能把两者混着讲。

**结论**：Hadamard + INT16 + global_amax 的**方法内容已被 1912.00131 + 1611.00429 + C5–C7 完整覆盖**。可写的只有：(a) **在 0.5B LLM、10% 上传率、真实双集群上端到端收敛到 fp16 基线**这一实证；（b）`global_amax` 作为"实现简单 / 隐私退化"的显式权衡记录（须诚实标注为权衡而非贡献）。**"我们提出 Hadamard 旋转用于 SecAgg" 是不可写的。**

---

## 5. 方向 D：cross-silo / federated LLM 通信

| # | 论文/系统 | Venue | 核心 | 与当前方案的关系 |
|---|---|---|---|---|
| D1 | **DiLoCo** — Douillard et al. | ICML 2024 W · [arXiv:2311.08105](https://arxiv.org/abs/2311.08105) ✅ | 内层 AdamW 多步 + 外层 Nesterov 动量；通信频率降 ~500×；容忍非 IID 与节点掉线 | **正交维度**：DiLoCo 降**频率**，当前方案降**每轮载荷**；二者可组合 |
| D2 | **Async DiLoCo** (Asynchronous Local-SGD) | 2024 · [arXiv:2401.09135](https://arxiv.org/abs/2401.09135) ✅ | 无 barrier；Delayed Nesterov；容忍 4× 速度差 | 正交 |
| D3 | **HALoS** (Hierarchical Async Local SGD) | ICML 2025 · [arXiv:2506.04531](https://arxiv.org/abs/2506.04531) ✅ | 地理分布式层级参数服务器；比 DiLoCo 快 7.5× | 正交 |
| D4 | **Decoupled DiLoCo** | DeepMind | 异步数据流 + 自愈；跨四个美国区域训 12B；>20× 快于常规同步 | 正交 |
| D5 | **SecLoRA** | ePrint 2026/1212 | 首个去中心化**低秩矩阵积**精确安全聚合（PC-DMCFE）；O(m+n) 通信；面向 cross-silo | **竞争应用路线**（LoRA 而非全参） |
| D6 | **SHE-LoRA** | 2025 · [arXiv:2505.21051](https://arxiv.org/abs/2505.21051) ✅ | 选择性同态加密 + LoRA；通信降 94.9%，加密开销降 99.8%；semi-honest | 竞争应用路线 |
| D7 | **FedShield-LLM** | — | LoRA + L1 剪枝 + CKKS FHE，面向 cross-silo 多机构；抗梯度反演/MIA | 竞争应用路线 |
| D8 | **FLAGuard** | IEEE | 可验证的联邦 LoRA 聚合 | 竞争应用路线 |

**判断**：
- 与 DiLoCo 系列**不是竞争而是可叠加**：当前方案每轮传 ~10–20% 参数，DiLoCo 每 H 步才传一次。若论文声称"通信高效"，DiLoCo 的 ~500× 会显著盖过当前方案 5–10× 的载荷削减，因此**必须明确 scope 是"每轮载荷"而非"总通信量"**。
- 与 SecLoRA/SHE-LoRA/FedShield-LLM **是同一应用目标的竞争路线**：它们用 FE/HE 在 **LoRA** 上做安全聚合，规避了全参传输；当前方案用 **plain SecAgg + 全参 block 掩码**，**代价是带宽更大，收益是不需要 HE/FE**。这是一条可以在论文里讲清楚的取舍，但必须承认 2025–2026 的 frontier 已在这条竞争线上。

---

## 6. 方向 E：dropout recovery / survivor threshold

见 §3 的 B1/B3/B4/B5。要点：
- 该机制在文献中**已完全成熟**（Bonawitz 2017 起），当前规范 §7 是其**忠实复述**；
- 「禁止双重重构」= Bonawitz 的 "either/or, not both"，**非新不变量**；
- LightSecAgg / Turbo-Aggregate 提供了比当前 all-to-all 更好的可扩展点；
- 当前方案在这条线上**没有任何可主张的差异**，只有 SPEC_ONLY 的规范文本。

---

## 7. 还值得补检索的缺口（供后续）

- **venue-only**（非 arXiv）的 SecAgg 论文覆盖不完整（S2 不可用所致）。建议授权 S2 API key 后重跑 `— sources: semantic-scholar`。
- **"public verifiable randomness beacon + FL audit"** 方向：检索到的全是通用随机信标文献（NIST Beacon、drand、QuoRand、区块链 beacon），**未发现与 FL 结合的既有工作**。这提示当前方案的"可审计公共 mask"在**密码学审计文献**里可能是空白区，但在 FL 文献里缺乏直接对照 —— 这一条的定位偏弱，不宜作为主要卖点。
- **X-Secure T-Private Federated Submodel Learning (ICC 2021)** 与 **SESA** 的后续引用链（谁引用了 SESA）值得展开，用于定位 §3.3 那个"索引隐私"论证缺口。

---

## 8. 关键结论（供 §4 使用）

1. **最有分量的邻居是 1912.00131 与 Bonawitz CCS 2017** —— 当前方案的数值路径与安全聚合层分别是它们的直接后裔，方法层面几无可主张的新增。
2. **SESA (ISIT 2024) 是一个当前方案完全未回应的威胁**：submodel × SecAgg 的索引隐私问题。当前设计的公开公共 mask 可能是好答案，但**尚未被论证**。
3. **`global_amax` 相对 1912.00131 的 autotuning 是隐私上的退步**，这是最易被审稿人攻击的一点。
4. **真正未被占满的位置**：block 粒度均匀带宽预算 + 与训练范围解耦 + 显式可重放审计协议 + 与 SecAgg 坐标冻结的耦合 —— 这是**系统工程定位**，外加**0.5B LLM / 10% 上传 / 真实双集群的端到端实证**。
5. **dropout recovery 与 survivor threshold 在文献中已成熟**，且当前仅有规范、无实验，不构成贡献。

---

## 9. 引用清单（已验证）

| 引用 | 标识 | 校验 |
|---|---|---|
| FedRolex (NeurIPS 2022) | arXiv:2212.01548 | ✅ verified (via arxiv) |
| FjORD / Ordered Dropout (NeurIPS 2021) | arXiv:2102.13451 | ✅ verified (via arxiv) |
| HeteroFL (ICLR 2021) | arXiv:2010.01264 | ✅ verified (via arxiv; **ID 经更正**) |
| EMBRACE / EmbracingFL (2024) | arXiv:2406.15125 | ✅ verified (via arxiv) |
| Bonawitz et al., Practical Secure Aggregation (CCS 2017) | arXiv:1611.04482 / ePrint 2017/281 | ✅ verified (via arxiv) |
| SecAgg+ (CCS 2020) | doi:10.1145/3372297.3417885 | ✅ verified (via crossref) |
| LightSecAgg (MLSys 2022) | arXiv:2109.14236 | ✅ verified (via arxiv) |
| Turbo-Aggregate (IEEE JSAIT 2021) | doi:10.1109/jsait.2021.3054610 | ✅ verified (via crossref) |
| SESA, Secure Submodel Aggregation (IEEE ISIT 2024) | doi:10.1109/isit57864.2024.10619575 | ✅ verified (via crossref) |
| Light-weight SA for Federated Submodel Learning (2021) | arXiv:2111.01432 | ✅ verified (via arxiv) |
| Dropout-Resilient Aggregation in Decentralized Learning (2024) | arXiv:2404.17984 | ✅ verified (via arxiv) |
| Bonawitz et al., Autotuned Communication-Efficient SecAgg (Asilomar 2019) | arXiv:1912.00131 | ✅ verified (via arxiv) |
| Suresh et al., Distributed Mean Estimation (ICML 2017) | arXiv:1611.00429 | ✅ verified (via arxiv) |
| QSGD (NeurIPS 2017) | arXiv:1610.02132 | ✅ verified (via arxiv) |
| RATQ (2019) | arXiv:1908.08200 | ✅ verified (via arxiv) |
| Error Feedback Fixes SignSGD (ICML 2019) | arXiv:1901.09847 | ✅ verified (via arxiv) |
| EF21 (NeurIPS 2021) | arXiv:2106.05203 | ✅ verified (via arxiv) |
| Error-Feedback Framework (2019) | arXiv:1909.05350 | ✅ verified (via arxiv) |
| ScionFL (SaTML 2024) | arXiv:2210.07376 | ✅ verified (via arxiv) |
| SCAFFOLD (ICML 2020) | arXiv:1910.06378 | ✅ verified (via arxiv) |
| FO-SGD (2024) | arXiv:2405.11095 | ✅ verified (via arxiv) |
| HLQ (2024) | arXiv:2406.15102 | ✅ verified (via arxiv) |
| FP8 Formats (2022) | arXiv:2209.05433 | ✅ verified (via arxiv) |
| DiLoCo (ICML 2024 W) | arXiv:2311.08105 | ✅ verified (via arxiv) |
| Async DiLoCo / Asynchronous Local-SGD (2024) | arXiv:2401.09135 | ✅ verified (via arxiv) |
| HALoS (ICML 2025) | arXiv:2506.04531 | ✅ verified (via arxiv) |
| SHE-LoRA (2025) | arXiv:2505.21051 | ✅ verified (via arxiv) |
| SecLoRA (2026) | IACR ePrint 2026/1212 | ⚠️ **UNVERIFIED**（仅来自 WebSearch 结果，未做 DOI/ID 校验） |
| FedShield-LLM / FLAGuard / FedNILO / Barbieri et al. / FedMask | — | ⚠️ **UNVERIFIED**（仅标题级来自 WebSearch，无稳定标识符） |
| X-Secure T-Private Federated Submodel Learning (ICC 2021) | doi:10.1109/icc42927.2021.9500855 | ✅ verified (via crossref) |

---

## 10. Post-review Errata（2026-09-18，评审后追加）

> 本节由外部评审（`gpt-5.6-sol` @ `xhigh`，threadId `01a0b3e2-54e9-7b03-9024-821824c2a7e4`）的裁定触发。
> **§1–§9 的审前原文保留不动**，以维持审计链。完整评审见 `AI_DESIGN_REVIEW.md`。

**E1 — 关系类型修正（`PARTIAL`）**：A1 FedRolex 由 `ancestor` 改判为 **`closest neighbor / baseline`**。理由：本方案未被记载派生自 FedRolex，且解决的是不同问题（全训 + 稀疏传输，而非异构子模型训练）。同理 A2 FjORD / A3 HeteroFL / A4 EMBRACE / A7 Federated Dropout 应为 **baselines / related predecessors**，而非 direct ancestors。

**E2 — 缺失的关键文献家族（`SUPPORTED`）**：§2 漏掉了**带 memory 的通信稀疏化**这一家族。因为客户端训练完整模型、只传选中坐标、把未传坐标留在 `block_memory`，本方案**至少同样自然地属于该家族**。应补入：

| 条目 | 标识 | 校验 |
|---|---|---|
| **Stich, Cordonnier, Jaggi, *Sparsified SGD with Memory*** (2018) | arXiv:1809.07599 | ✅ verified (via arxiv，评审后独立核验) |

并应指出：`block_memory` 的 **decay = 0.9 是非标准、启发式的残差处理**，与 EF 的标准定义不同 —— 而 §4 只把 error feedback 引给了 `quant_residual`，未充分比较 `block_memory`。**该漏项进一步压缩了第 1 层的新颖性。**

**E3 — 第 2 层的第二处 PRG 分歧（`SUPPORTED`，审前遗漏）**：除 §2.3 记录的 block-mask PRG 分歧外，SecAgg 层另有独立分歧 —— SecAgg 规范写 ChaCha20，而 active 执行计划记载 pairwise/self mask **当前实际用 SHAKE-256**，把换 AES-CTR/ChaCha 列为未来工作（`2026-09-18-secagg-time-efficiency.md:52`，PERF-7）。

**E4 — 关系类型修正（`PARTIAL`）**：
- C4 RATQ 应为 **理论基线 / 祖先**，非"支持性结论"。
- C6 EF21 是**替代性压缩设计**，非"严格更优的替代"。
- **C8 ScionFL 是竞争基线**；称本方案"在精度/比特轴上弱于它"只在它自己的设置下成立，**"全局更弱"不成立**（表示、MPC 机制、任务、威胁模型、总坐标传输量均不同）。
- A5 Barhieri / A6 FedNILO / A8 FedMask、D5 SecLoRA、D7–D8 FedShield-LLM / FLAGuard 在本文件中自标为未核验或标识符不完整，**不能承担强定位权重**（`UNKNOWN`）。

**E5 — 未涉及但属本文件范围的新结论**：SecAgg 算术包络 `N_max · Q_max < q/2` 在当前 `q=2^16, Q_max=16383` 下**只对 N=2 成立**（32766 < 32768；N=3 为 49149 > 32768），而规范附录 A **推荐 2^32**。详见 `AI_DESIGN_REVIEW.md` §2 D3（已独立复算核验）。
