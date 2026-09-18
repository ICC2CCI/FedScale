# CURRENT_METHOD — 当前书面方案固化（FedScale `deploy/central-server-minio`）

- **生成**：2026-09-18
- **用途**：本文是 §4（`CURRENT_VS_LITERATURE.md`）与 §5（Research Review）的唯一方法输入。
- **证据边界**：只来自下列书面方案。**不**来自 git history、源码、YAML、测试文件、未入册实验目录或记忆。
- **工作名称**：本方案在规范中自命名为 **Hierarchical Public Permutation-and-Rotation Mask**（`_blockmask.txt` §14 末尾"v1 冻结建议"）；安全聚合层自命名为 **FedScale Windowed SecAgg v2**。

## 0. 证据出处清单

| 代号 | 文档 | 性质 |
|---|---|---|
| **S-BM** | `docs/ai-design/_blockmask.txt`（Public Block Mask 规范 v1） | 协议规范 |
| **S-SA** | `docs/ai-design/_secagg.txt`（Windowed SecAgg 规范 v2） | 协议规范 |
| **A-V3** | `docs/algorithm/2026-09-04-s3r12v3-block-uniform.md` | 算法 + 单机结果 |
| **A-FLOW** | `docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md` | 双集群执行流程 |
| **A-PREC** | `docs/algorithm/2026-09-15-secagg-quantization-precision-issue.md` | 量化精度问题与默认路径 |
| **A-SURV** | `docs/algorithm/2026-09-17-secagg-quantization-optimization-survey.md` | 量化调研与落地 |
| **P-INT** | `docs/exec-plans/completed/2026-09-10-dual-cluster-s3r12v3-fsdp.md` | 联调结项 |
| **P-PROD** | `docs/exec-plans/completed/2026-09-11-dual-cluster-to-production.md` | 生产切片结项 |
| **P-TIME** | `docs/exec-plans/active/2026-09-18-secagg-time-efficiency.md` | 时间效率（**active**） |

**对照、不进主路径**：`docs/exec-plans/completed/2026-09-04-fedrolex-partial-training.md`、`.../2026-09-04-s3r12v2-block-permutation.md`。二者只用于说明"这不是当前方法"。

**跑次可核验性**：本机 `results/` 下无任何时间戳跑次目录（仅 `README.md`、`figures/`、`round_logs/` 旧布局）。下文引用的跑次（`20260914-final-clean`、`202609162025`、`202609180941`、`202609181151` 等）**只以文档引用形式存在，无法独立核验**。按任务约定直接引用文档写明的数字，不视为 blocker。

## 1. 主路径四层总览

```
第 1 层 选块        Hierarchical Public Permutation-and-Rotation Block Mask
                    → 本轮上传哪些 canonical blocks（公开、可重放、无放回轮转）
        ↓
第 2 层 安全聚合    Windowed SecAgg v2：pairwise mask + self mask
                    + (规范中定义) dropout recovery / survivor threshold
                    → 隐藏已选 block 的单 ICC 数值；server 仅见聚合
        ↓
第 3 层 数值路径    pad-p2 → 符号翻转 D → FWHT → INT16 随机舍入（global_amax 单一 scale）
                    + error feedback（block_memory / quant_residual 拆分）
                    → 使整数域 secure sum 与低上传比例同时成立
        ↓
第 4 层 传输执行    MinIO 对象存储传更新；HTTP 控制面；single blob(SAW1)；
                    并行 unmask；异步 finalize / 异步全量写盘
```

分层原则（S-SA §2）：**Public Block Mask 决定"传哪些 blocks"；SecAgg Random Mask 决定"如何隐藏这些已选 block 的数值"。二者必须严格分层。**

---

## 2. 第 1 层：选块（Public Block Mask）

### 2.1 定义

| 要素 | 内容 | 出处 |
|---|---|---|
| 策略名 | Hierarchical Public Permutation-and-Rotation Mask，`mask_policy_id = hierarchical-permute-rotate-v1` | S-BM §9 / §14 |
| 粒度 | canonical **block**（不是 key/层）；flat-view 连续切片，默认 `BLOCK_SIZE = 524288` elements ≈ 1 MB bf16 | A-V3 §3.1 |
| 比例 ρ | 规范默认 **0.25**；实现按 `coverage_h` 决定 ρ ≈ slots_per_round / H | S-BM §2；P-PROD CFG-2 |
| 覆盖周期 H | 规范默认 **4**；S3R12v3 结项 **H=5**（≈20%）；SecAgg 验证配置常用 **H=10**（10%） | S-BM §2；A-V3 §4；A-PREC §5.2 |
| 分层 | 规范：按 Transformer 层（+ `always_on` 小参数）；实现：25 个组（24 层 + 1 non_layer），组内独立排列 | S-BM §5 / §7；A-V3 §3.2 |
| 排列 | Fisher-Yates，每 Mask Epoch 重新 shuffle，Epoch 内冻结 | S-BM §5；A-V3 §4 |
| 选择规则 | `M_t(B_j)=1 iff position_l(B_j) mod H = slot` | S-BM §6 |
| slot 推进 | `slot = successful_round_index mod H`，仅成功 commit 后 +1；失败轮不消耗 slot | S-BM §2.1 / §8 |
| 覆盖保证 | 一个 Mask Epoch 内每个可轮转 block **恰好被选一次**（无重复、无饥饿） | S-BM §3 / §6.1 |
| 公开性 | mask 在 ICC 本轮私有 update 产生**之前**即可确定；不使用任何 ICC 私有明文 update | S-BM §1 / §3 |

### 2.2 规范中定义的生成机制（S-BM §4–§6）

```
epoch_seed  = SHA256("FedScale-BlockMask-v1" ‖ job_id ‖ layout_version ‖ mask_epoch
                     ‖ epoch_anchor_manifest_hash)                 # §4.1
layer_seed  = HKDF-SHA256(IKM=epoch_seed, salt=layout_hash,
                          info="FedScale-LayerMask-v1" ‖ layer_id)  # §5
permutation = FisherYates(layer_blocks, PRG=ChaCha20(layer_seed))   # §5, rejection sampling
```

- **epoch anchor 冻结**（S-BM §4.2）：`epoch_anchor_manifest_hash` 是"当前 Mask Epoch 开始前的最后一个 committed manifest hash"，slot=0 派生一次，slot=1..H-1 复用；禁止每轮用最新 manifest 重新派生。
- **规范中定义**：确定性编码要求（length-prefixed job_id、固定宽度 u64 BE、raw digest 而非 hex）、跨语言 test vectors、ICC 独立重算 + `mask_hash` 校验后再进入 SecAgg（S-BM §4.3 / §9 / §10.3 / §13.2）。

### 2.3 实现口径与规范的差异（**重要定位风险**）

A-V3 §4.1 / §5 记录的实现口径与 S-BM 规范**不完全一致**：

| 项 | S-BM 规范 | A-V3 实现口径 |
|---|---|---|
| 种子派生 | SHA256(含 job_id/layout_version/anchor) + HKDF-SHA256(salt=layout_hash) | `epoch_seed = SHA256("FedScale-BlockMask-v1" \| SEED \| epoch)`；`group_seed = SHA256(epoch_seed \| "\x00" \| "FedScale-GroupMask-v1" \| group_id)` |
| PRG | ChaCha20（§5.1 **明令禁止**使用未固定跨语言输出语义的 RNG） | `PRG=Random(group_seed)`（Python `random`） |
| 分组 | 每 Transformer 层 + `always_on` | 25 组 = 24 layer + 1 non_layer（non_layer 合并 3 个 key 共 521 blocks） |
| epoch anchor | manifest hash 冻结 | 未在 A-V3 中记录 |

**判定**：这是**规范文档与算法文档之间的口径分歧**，不是"协议被代码改写"。二者同属"书面方案"，因此：
- "cross-language / cross-process replayable" 与 "不使用非确定性 RNG" 属于 **SPEC_ONLY（规范中定义）**；
- 已实验验证的只有单机/双集群在**同一实现**下的可复现性，**UNKNOWN** 是否满足规范要求的跨语言 test vectors；
- $\rho$ 与 $H$ 的取值差异（4 / 5 / 10）属于**执行旋钮**，不是协议分歧，不应写成"协议被改写"。

### 2.4 证据分级

| 命题 | 等级 | 依据 |
|---|---|---|
| 无放回轮转 + Epoch 级 100% 覆盖、无饥饿 | **SUPPORTED（单机）** | A-V3 §7/§8：20 轮 4 个 Epoch，每 Epoch 精确 630.2M elems = 100% |
| 上传量近恒定 | **SUPPORTED（单机）** | A-V3 §8：18.5%~21.5%，range 3.0% |
| 收敛质量（20% 带宽） | **SUPPORTED（单机，2 client）** | A-V3 §9：R20 eval 1.0514 vs 全量 S2 1.0091（gap +0.042，≈96% 全量质量） |
| 双集群链路可跑通 | **SUPPORTED（链路）** | P-INT §交付：2 client、20 轮、H=5 ≈20%、参考跑次 `results/202609101345/` |
| block 级 vs key 级上传波动改善 | **SUPPORTED（单机对照）** | A-V3 §9：range 30.1% → 3.0%，R20 1.0569 → 1.0514 |
| mask 公开、不依赖本轮私有 update | **SPEC_ONLY** | S-BM §3 不变量；A-V3 未做该性质的对照实验 |
| mask 确定性 + 跨实现一致 + 审计可重算 | **SPEC_ONLY**；跨语言部分 **UNKNOWN** | S-BM §13.2 要求 3 组 test vectors；文档未记录已通过 |
| 失败轮次不推进 slot | **SPEC_ONLY** | S-BM §8；实现侧未见故障注入记录 |
| `always_on` / 小参数策略 | **SPEC_ONLY** | S-BM §7；实现口径改用 non_layer 并组处理 |
| `coverage_h` 可当旋钮切 5/10/20/50% | **SUPPORTED（短测）** | P-PROD ALG-1：自测与 1/H 成比例；ALG-2 支持多 slot（H=5×2→≈40%） |

---

## 3. 第 2 层：安全聚合（Windowed SecAgg）

### 3.1 定义（S-SA §1 / §5–§10）

| 要素 | 内容 | 出处 |
|---|---|---|
| 安全目标 | aggregate-only：Server 最终仅得 cohort 聚合整数向量/反量化更新，得不到任一 ICC 未掩码 `q_{k,w}` | S-SA §1.1 |
| 威胁模型 | Server = **honest-but-curious**；ICC 已认证且按协议执行；网络由 TLS/mTLS + 签名控制面防护 | S-SA §1.2 |
| 明确非目标 | 不解决主动投毒、任意 Byzantine ICC、恶意量化值、侧信道、流量分析；**≠ DP** | S-SA §1.2 表 |
| 密码学原语 | X25519 + HKDF-SHA256 + ChaCha20 PRG + Shamir 阈值分享（成熟库/标准原语） | S-SA §5.2 / §6.2 / §7.3；附录 C |
| pairwise mask | all-to-all，符号按 canonical ICC id 排序（小加、大减），`R_{kl,w}` 在求和中对消 | S-SA §6.3 |
| self mask | `self_master_k`（每 attempt 新生成）→ HKDF 派生窗口 self seed → `B_{k,w}` | S-SA §7.2 |
| 域分离 | protocol/version、job_id+session_id、attempt_id、cohort_hash、mask_hash、window_id+window_layout_hash、peer pair、icc_id **均必须**入 HKDF info | S-SA §8 表 |
| session / attempt | `secagg_session_id` 描述"聚合什么"；`attempt_id` 描述"第几次密码学尝试"；attempt 变化则**全部密钥/mask 重生** | S-SA §5.1 / §5.3 |
| windowed secure sum | 一个 attempt 一次 setup；**每个 window 独立派生 mask**；setup 成本在 attempt 内摊销 | S-SA §10.1 |
| 量化边界 | 唯一一次量化，在 SecAgg 入口前完成；SecAgg 层只处理 `Z_q` 整数向量 | S-SA §9.3 |
| 回绕安全 | `N_max · Q_max < q/2`（等权）；加权需 `Σ|a_k|·Q_max < q/2`；计划阶段做 overflow budget check | S-SA §9.2 |
| 原子提交 | 所有 window 验证通过后一次性提交；all-windows atomic | S-SA §11.3；附录 A |

### 3.2 Dropout recovery / survivor threshold（S-SA §7，**全部为规范中定义**）

| 客户端状态 | 计入 U* | 需恢复什么 |
|---|---|---|
| 完成全部 masked windows + 正常 unmask | 是 | 正常协助移除 self mask |
| 完成全部 windows，unmask 前掉线 | 是（阈值允许时） | 重构 **self-mask secret** |
| key/share 阶段参加，未完成全部 windows | 否 | 恢复 **pairwise-mask 恢复秘密** |
| setup 前即未参加 | 否 | 无 |

- 每 ICC 对两类恢复秘密分别做 **t_rec-of-n 阈值分享**（① pairwise-mask recovery secret，② `self_master_k`）；share 绑定 `secret_type/owner/recipient/session/attempt`，经已认证加密信道或 Server 不可读 relay 转发（S-SA §7.3.1）。
- Server 在 deadline 后冻结 **U\*** 并发布签名 survivor transcript；ICC 仅针对该 transcript 释放允许的 shares（S-SA §7.3.2）。
- **禁止双重重构**（S-SA §7.3.3）：对任一 `k ∈ U*`，Server 不得同时获得足以恢复其 pairwise-mask 根秘密 **和** `self_master_k` 的 shares；违反即 abort。
- 阈值：`|U*| < q_min` 或低于重构阈值 → 整轮 abort，**不降级为明文聚合**，`successful_round_index` 不递增（S-SA §11.2 / §15.1）。

### 3.3 证据分级

| 命题 | 等级 | 依据 |
|---|---|---|
| pairwise mask 无 dropout 时精确对消（`Σ pairwise = 0 mod q`） | **SUPPORTED（单元级 + 2 client 跑通）** | A-PREC §1.2 明确"数学上精确抵消，单元测试验证，不是问题" |
| self mask 可在聚合后被 Server 移除 | **SUPPORTED（2 client 路径）** | A-FLOW §3.1 步骤 ⑩：去 self-mask |
| Server 只看到 masked `z_k` + 1 个 `global_amax`，看不到单 ICC `q_k` | **SUPPORTED（2 client，honest-but-curious）** | A-PREC §3.4.4 表；§3.4.5 总结 |
| X25519 临时密钥 / session 同步 / attempt 语义 | **部分落地**：session 同步与 self_master 提交流程已跑通；**attempt_id++ 与 fresh-mask 重试语义未见实验记录** | A-SURV §3.6 "曾踩过的坑"1（空 session）；A-PREC §3.3 |
| **Shamir 阈值分享 / dropout recovery（两类）** | **SPEC_ONLY（无实验证据）** | S-SA §7.3 / §16.2（5 项故障注入）；A-PREC / A-SURV / A-FLOW 均未记录实现或故障注入 |
| **survivor set U\* 冻结** | **SPEC_ONLY / UNKNOWN** | S-SA §4.3 / §7.3.2；执行文档只在 SYNC 层记录了部分参与 |
| 禁止双重重构不变量 | **SPEC_ONLY（未验证）** | S-SA §7.3.3 |
| overflow budget check | **SPEC_ONLY**；实现侧只记录 `clip_frac=0`、`zero_frac≈0.0003` | S-SA §9.2；A-PREC §4.3 |
| 掉线鲁棒（达到阈值可完成聚合，低于阈值整轮 abort） | **PARTIAL**：`|U*| < q_min → abort` 的执行侧语义在 P-PROD SYNC-1 记为 done；但这是**FL 同步层**的"部分参与"，**不等于** Shamir/threshold dropout recovery | P-PROD §4 SYNC-1/SYNC-2；S-SA §7 |

> **必须区分的两条线**（任务显式要求）：
> - **SYNC 层迟到/丢弃**（P-PROD SYNC-1/SYNC-2：`min_clients_to_aggregate`、迟到 409、连续缺席 inactive）是**执行层**的部分参与策略；
> - **Shamir / threshold dropout recovery**（S-SA §7）是**密码学层**的未配对 pairwise mask 与 self mask 恢复。
> 二者在文档中**没有**建立等价关系。当前 2-client、无真实 dropout 的验证**不覆盖**后者。

---

## 4. 第 3 层：数值路径（Hadamard + INT16 + global_amax）

### 4.1 定义

```
FP16 模型参数
  → FP32 delta（不 round 回 FP16）                      # Issue #1
  → FP32 (block_memory + quant_residual)
  → pad 到 2 的幂 → 符号翻转 D → FWHT（Hadamard）
  → INT16 + stochastic rounding（全局 scale）
  → pairwise + self mask (mod q = 2^16)
  → MinIO 单 blob 上传
  → Server unmask → 反量化 → 逆 FWHT → 去 pad → FedAvg
```
（A-PREC §3.1 / A-SURV §3.2.2）

| 参数 | 值 | 出处 |
|---|---|---|
| 模数 | `secagg_modulus_bits: 16`，q = 2^16 = 65536 | A-PREC §3.2 |
| Q_max | 16383（= q/4 − 1） | A-PREC §3.2 |
| scale 模式 | **全局 scale**：`secagg_hadamard: true`，只收 `global_amax`；`scale = max_k(global_amax)/Q_max × 1.05` | A-PREC §3.1/§3.2 |
| SCALE_COVERAGE | 1.05（amax 映射到 0.95·Q_max，避免 clip） | A-PREC §2.3 |
| 随机舍入 | 开启，`E[quantize(x)] = x` | A-PREC §3.2 |
| `memory_decay` | 0.9（**仅未选中 block** 的 block-mask residual） | A-PREC §3.2 |
| `quant_residual_decay` | **1.0**（量化残差不衰减，Issue #1） | A-PREC §3.2 |
| 非 Hadamard 退路 | per-window 当轮 amax，`secagg_hadamard: false` | A-PREC §3.1.1 |

**Issue #1（量化前多余精度损失）两条**（A-SURV §1.1）：① 过早 FP16 rounding（FP32 delta 后又 `.to(fp16).to(fp32)`）；② 量化残差被 `memory_decay=0.9` 衰减——标准 error-feedback 应保存 `e = x − D(Q(x))` 且不衰减。

**amax 隐私代价**（A-PREC §3.4.1）：当前每轮泄露 **1 个 `global_amax`**（旋转域 L∞）；旧 per-window 路径泄露 ~269 个 per-window L∞（已默认关闭）。同表另列 train/eval loss、`num_examples`、timing、`block_energies` 等**非 SecAgg 引入**的泄露项。

### 4.2 证据分级

| 命题 | 等级 | 依据 |
|---|---|---|
| Hadamard + INT16 + Issue #1 在 20 轮上对齐 fp16 基线 | **PARTIAL → 待审** | A-PREC §3.7 / A-SURV §3.7：`202609180941` R20 = **0.985** vs fp16 `20260914-final-clean` R20 = **0.987**（差 0.002） |
| 旧 per-window 路径 R20 = 1.328 | **PARTIAL** | `202609162025`（A-PREC §4.1）；同 2-client 设置 |
| 修复前发散（R5 = 2.362） | **PARTIAL** | A-PREC §1.1，无跑次号 |
| 无需 Hadamard 的工程退路仍可用 | **PARTIAL** | A-PREC §3.1.1（R20=1.328），标注为历史/隐私退路 |
| 随模型/客户端数/数据异构的泛化 | **UNKNOWN** | 文档只有 Qwen2.5-0.5B、2 client、IID 50/50；P-TIME §3 明确"通用流程 ≠ 数值从 0.5B 自动成立" |
| 量化残差与 block-memory 拆分是标准 error-feedback 的正确用法 | **SUPPORTED（文献）** | A-SURV §2.3（EF, arXiv:1901.09847）、§2.4（EF21, arXiv:2106.05203） |
| "旋转 + 均匀量化近最优" | **SUPPORTED（文献）** | A-SURV §2.5（RATQ, arXiv:1908.08200） |
| 单种子 / 单跑次 / 20 轮 → 统计结论 | **证据不足** | 无多种子、无重复跑次记录 |

**注意**：A-PREC §3.7 自身结论是"剩余差距不宜再主要归因于 INT16 位宽"，即**该 0.002 的差距被解释为可接受**，而非"INT16 无损"。这是一个"对齐"声明，不是"等价"证明。

---

## 5. 第 4 层：传输与执行

### 5.1 定义（A-FLOW §1/§3.1；P-TIME §1–§2）

| 项 | 内容 | 出处 |
|---|---|---|
| 大文件通道 | **MinIO 对象存储**（模型更新 / 全局增量 / 全量 checkpoint） | A-FLOW §1 |
| 小控制通道 | **HTTP REST**：问轮次、拿 plan、通知上传完、问聚合结果 | A-FLOW §6 |
| 上传对象 | 非 SecAgg：`blocks.pt`（fp16）；SecAgg：**一个** masked INT16 blob（`SAW1` 容器，内含全部 window） | A-FLOW §3 表 |
| 通知路径 | POST 只登记 `z_key` + `vector_len`，handler **O(1) 与对象大小无关**；GET 放到 finalize | P-TIME PERF-1 |
| FWHT 复用 | amax 阶段的 FWHT 结果复用，去掉第二次正向旋转 | P-TIME PERF-3 |
| Server finalize | 按窗**并行** unmask / iHadamard（`SECAGG_UNMASK_WORKERS`）；`global_delta` 写完即 `done`；全量 `global_state` **异步** PUT | P-TIME PERF-9 / PERF-11 |
| MinIO 客户端 | 进程级线程池；假死仍 rebuild | P-TIME PERF-5 |
| FWHT buffer | 双缓冲，每级不再 `empty_like` | P-TIME PERF-6 |

### 5.2 证据分级

| 命题 | 等级 | 依据 |
|---|---|---|
| 时间优化 5 轮不改变数值 | **SUPPORTED（5 轮）** | `202609181151` R1–R5 eval 与 `202609180941` **逐轮相同**（R5 = 1.363804）；`upload_blocks_MiB` 不变（R1 = 139.4） | 
| 优化后墙钟 ≈110–134s（对照 148s） | **PARTIAL（5 轮）** | P-TIME §5 表：R1 134s、R2 115s、R3 113s、R4 110s、R5 114s |
| `wait_agg` ≈10s | **PARTIAL（5 轮）** | P-TIME §5：R1 10.1s、R2 11.6s、R3 11.9s、R4 9.6s、R5 10.1s |
| "N 增大时墙钟主要跟带宽/计算走、不跟 RTT×N 走" | **UNKNOWN（未验证）** | P-TIME §1 把它列为**验收总则**，并说"同一套改动必须在 window 数变多时仍然成立"；文档未记录该压测结果 |
| 20 轮确认 R20≈0.985 | **UNKNOWN（进行中）** | P-TIME 表头："20 轮算法验证进行中"；不得补最新目录 |
| 25% block ≠ 25% bytes 的口径纪律 | **SPEC_ONLY（规范要求）** | S-BM §3.2 / §12：应记录 selected bytes，不得混写 |
| 端到端收敛（SecAgg 路径 20 轮） | **PARTIAL** | `202609180941` R20 = 0.985（coverage_h=10 ≈10%） |

---

## 6. 当前方法的四层"已验证 / 未验证"汇总

| 层 | 已验证 | 仅有规范 / 未验证 |
|---|---|---|
| **选块** | Epoch 内无放回 100% 覆盖、上传波动 range 3%、20% 带宽下 96% 全量质量（单机 2 client）；双集群链路跑通 | mask 公开性/确定性/跨语言可重放/test vectors；失败轮不推进 slot；`always_on`；规范与实现口径分歧的收敛 |
| **安全聚合** | pairwise 精确对消（单元 + 2 client）；self mask 移除；session 同步；server 仅见 masked `z_k` + 1 amax | **Shamir 阈值分享、两类 dropout recovery、U\* 冻结、禁止双重重构、attempt_id fresh-mask 重试语义、overflow budget check、dropout 故障注入 5 项** |
| **数值路径** | Hadamard + INT16 + Issue #1 在 2 client / 0.5B / IID / coverage_h=10 / 20 轮上 R20 = 0.985（vs fp16 0.987） | 跨模型、跨参与规模、非 IID、多种子；INT16 vs fp16 的"等价"（doc 只说"不宜再主要归因于 INT16"）；`global_amax` 泄露的可接受性量化 |
| **传输执行** | 单 blob + 并行 unmask + 异步写盘在 5 轮上数值逐轮一致，墙钟 148s → 110–134s | N 增大时的扩展性（window 数变多仍成立）；20 轮时间确认 |

## 7. 明确不属于当前主路径

- `block_vote`、`block_topk`：不在任何书面方案中，不构成方法。
- Flower / TKE：S-SA §8 实现原则与 §13 明确"**不依赖 Flower**"；Flower SecAgg+ 仅列为实验对照（附录 C）。
- 旧的逐窗 PUT / hex JSON 上传：P-TIME PERF-1/PERF-2 已替换；A-PREC §4x 标注"请勿当成当前瓶颈清单"。
- `docs/exec-plans/completed/2026-09-04-fedrolex-partial-training.md`、`.../2026-09-04-s3r12v2-block-permutation.md`：历史对照，不进主路径。
- 未写入正式文档的新实验目录（含 `202609181406`）：不得使用。

## 8. 一句话方法陈述（供 §4 与 Review 使用）

> FedScale 当前书面方案在**跨机构（cross-silo）联邦 LLM 微调**中，以 **公开、确定、可重放的 block permutation + 无放回轮转**决定每轮上传哪些 canonical blocks（H 轮内 100% 覆盖、上传量近恒定），在其上叠加 **Windowed SecAgg**（pairwise + self masks，规范含 Shamir dropout recovery 与 survivor threshold 冻结）以隐藏单 ICC 数值，并用 **确定性 Hadamard 旋转 + INT16 随机舍入 + global_amax 单一 scale + error feedback** 把整数域安全求和与低上传比例同时做到与 fp16 基线对齐；执行层以**对象存储传更新 + HTTP 控制面 + 单 blob + 并行 unmask + 异步 finalize** 把完成时间压到接近无 SecAgg 基线。

**已验证的内核是"选块 + 数值路径 + 执行"三位一体；安全聚合层目前只有 pairwise/self mask 的 2-client 无 dropout 验证，dropout recovery 与 survivor threshold 仍停留在规范。**
