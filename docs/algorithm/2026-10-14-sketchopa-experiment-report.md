# SketchOPA 大模型联邦微调实验报告

> **日期**：2026-10-13 ~ 2026-10-14
> **模型**：Qwen2.5-0.5B（d=630,167,424 参数，291 层）
> **目标**：在真实双集群环境下，验证 SketchOPA 算法（Top-k → CountSketch → 安全聚合 → 恢复）用于大模型联邦微调的可行性与收敛性

---

## 1. 实验背景

### 1.1 为什么要做这个实验

在联邦学习（Federated Learning）中，各参与方需要将本地训练后的模型更新上传到中心服务器进行聚合。对于大语言模型（如 Qwen2.5-0.5B，6.3 亿参数），每轮上传的全量梯度约 **2.5GB**（fp32）或 **1.26GB**（fp16），通信开销巨大。

SketchOPA 是一种压缩通信方案，核心思路是：
1. **Top-k 稀疏化**：每个 client 只保留梯度中绝对值最大的 k 个坐标（本实验 k/d = 1%，即 630 万个）
2. **CountSketch 编码**：将稀疏梯度编码到 CountSketch 数据结构中，进一步压缩
3. **安全聚合**：使用 SHPRG（Seed-Homomorphic PRG）mask + Shamir 秘密分享，让服务器只能看到聚合后的结果，看不到单个 client 的梯度
4. **恢复**：服务器聚合后，用 median-of-rows CountSketch 恢复算法近似还原 Top-k 坐标和值

理论上，SketchOPA 可以将每轮上传量从 1.26GB（dense fp16）压缩到约 **200MB**（sketch payload），压缩比约 **6x**，同时提供安全聚合保证。

### 1.2 实验目标

通过 5 组对照实验，回答以下问题：
- **B0（dense 基线）**：不压缩，全量 FedAvg，作为收敛性上限
- **B1（topk 基线）**：只做 Top-k 稀疏化（不经过 CountSketch），作为压缩方法的理想上限
- **E1（sketchopa, 无 residual）**：完整 SketchOPA 流程，不使用误差反馈，验证基本可行性
- **E2（sketchopa + residual, uniform allocation）**：加入误差反馈缓冲区，验证 residual 是否能稳住训练
- **E3（sketchopa + residual, calibrated allocation）**：用 calibration 数据指导 sketch 预算分配，验证是否进一步提升恢复质量

---

## 2. 实验设置

### 2.1 硬件环境

| 角色 | 节点 | IP | GPU | 显存 |
|---|---|---|---|---|
| Coordinator（聚合服务器） | Central | 192.168.235.42 | CPU 即可 | — |
| Client 0 | ICC1 | 192.168.206.116 | V100 32GB ×1 | 32GB |
| Client 1 | ICC2 | 192.168.205.130 | V100 32GB ×1 | 32GB |

- Coordinator 运行 FastAPI HTTP 服务（端口 8090），负责接收 client 上传、聚合、下发全局模型
- Client 通过 HTTP 直传文件给 coordinator（不走 MinIO，因 MinIO 对大文件 multipart upload 不稳定）
- 两台 ICC 节点位于不同子网，通过内网与 Central 通信

### 2.2 模型与数据

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-0.5B |
| 参数量 | 630,167,424（d） |
| 层数 | 291（含 embed_tokens, 28 层 transformer, lm_head 等） |
| 训练精度 | **fp32**（fp16 AdamW optimizer state 会溢出导致 nan） |
| 传输精度 | fp16（上传时转换，减半传输量） |
| Client 0 训练数据 | `icc1_client0_train.json`（15,088 条） |
| Client 1 训练数据 | `client1_train.json`（15,088 条） |
| 评估数据 | `medical_flashcards_eval.json`（3,352 条，两个 client 相同） |

### 2.3 训练超参

| 超参 | 值 | 说明 |
|---|---|---|
| num_rounds | 20 | 联邦训练轮数 |
| local_steps | 30 | 每轮本地训练步数 |
| batch_size | 4 | 每步 batch 大小 |
| grad_accum | 4 | 梯度累积步数（等效 batch_size=16） |
| lr | 1e-5 | 学习率（constant，无 scheduler） |
| optimizer | AdamW | fp32 optimizer state |

### 2.4 SketchOPA 算法超参

| 超参 | 值 | 说明 |
|---|---|---|
| topk_ratio | 0.01 | Top-k 比例，k = d × 0.01 = 6,301,674 |
| rows | 5 | CountSketch 行数（median-of-rows 恢复） |
| recover_rho | 2.0 | 恢复时的阈值参数 |
| sketch_len | 18,905,020 | CountSketch 列数（= k × 6 / (rows × 2)） |
| value_bytes | 2 | sketch 中 value 的字节数（int16） |
| idx_bytes | 4 | sketch 中 index 的字节数（int32） |
| secure | True | 启用安全聚合（SHPRG mask + Shamir seed 分享） |
| modulus_q | 65521 | 安全聚合的 Z_q 模数（< 2^16 最大素数） |
| SHPRG seed_dim | 4 | SHPRG 种子向量维度 |

### 2.5 Sketch 预算分配策略

CountSketch 的总 bucket 数（sketch_len = 18,905,020）需要在 291 层之间分配：

| 策略 | 说明 | 用于实验 |
|---|---|---|
| **uniform_param** | 按每层参数量比例分配 bucket | E1, E2 |
| **topk_count_calibrated** | 按各层 Top-k 坐标出现频率（calibration 数据）分配 bucket | E3 |

calibration 数据来自 E1 实验中统计的各层 Top-k 坐标计数（两个 client 合并，20 轮累计）。

### 2.6 Residual Buffer（误差反馈）

| 项 | 说明 |
|---|---|
| 作用 | 每轮未被 Top-k 选中的梯度残差累积到 buffer，下一轮加入梯度再选 Top-k，减少稀疏化信息损失 |
| clip_norm | 0.0（不裁剪） |
| decay | 1.0（不衰减） |
| 用于实验 | E2, E3 |
| 不用于 | E1（验证无 residual 时的表现） |

### 2.7 通信量对比

| 方法 | 每轮上传大小（per client） | 压缩比（vs dense fp16） |
|---|---|---|
| Dense fp16（B0） | 1,260,426,553 bytes（1.17 GB） | 1× |
| Top-k（B1） | 75,621,925 bytes（72 MB） | 16.7× |
| SketchOPA（E1/E2/E3） | 201,655,461 bytes（192 MB） | 6.3× |

> Top-k 上传更小是因为只传坐标(int32) + 值(fp16)，6.3M × 6 bytes = 37.8MB，加上元数据约 72MB。
> SketchOPA 上传更大是因为 CountSketch 结构本身有 18.9M × 5 rows × 2 bytes = 189MB，加上安全聚合 mask 开销。
> 但 SketchOPA 的优势在于安全聚合——服务器无法看到单个 client 的梯度，只能看到聚合结果。

---

## 3. 实验结果

### 3.1 总览

| 实验 | 方法 | Allocation | Residual | R0 eval_loss | R19 eval_loss | 状态 |
|---|---|---|---|---|---|---|
| **B0** | Dense FedAvg | — | — | 1.174 | **1.075** | 稳定收敛 ✓ |
| **B1** | Top-k FedAvg | — | — | 1.734 | **1.160** | 稳定收敛 ✓ |
| **E1** | SketchOPA | uniform | ✗ | 1.783 | **15.548** | 完全发散 ✗ |
| **E2** | SketchOPA | uniform | ✓ | 1.783 | **1.987** | 未崩溃，波动大 △ |
| **E3** | SketchOPA | calibrated | ✓ | 1.783 | **2.977** | 波动更大 △ |

### 3.2 B0 Dense 基线（完整 20 轮）

> **方法**：全量 FedAvg，无压缩，无安全聚合。这是收敛性的上限基准。

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
| 10 | 0.263 | 1.099 |
| 11 | 0.261 | 1.096 |
| 12 | 0.261 | 1.093 |
| 13 | 0.259 | 1.088 |
| 14 | 0.259 | 1.088 |
| 15 | 0.251 | 1.083 |
| 16 | 0.258 | 1.083 |
| 17 | 0.251 | 1.080 |
| 18 | 0.250 | 1.077 |
| 19 | 0.248 | **1.075** |

**结论**：eval_loss 从 1.174 稳步下降到 1.075，收敛趋势清晰。train_loss 从 0.312 降到 0.248。两个 client eval_loss 全程完全一致（FedAvg 聚合后模型相同 + 评估集相同）。

**结果文件**：`results_sketchopa/B0_dense_20r_v7/round_log.json`

### 3.3 B1 Top-k 基线（完整 20 轮）

> **方法**：Top-k 稀疏化（k/d=1%），client 上传坐标+值，服务器直接聚合。不做 CountSketch，不做安全聚合。这是压缩方法的理想上限。

| Round | train_loss | eval_loss |
|---|---|---|
| 0 | 0.312 | 1.734 |
| 1 | 0.332 | 1.481 |
| 2 | 0.297 | 1.380 |
| 3 | 0.300 | 1.289 |
| 4 | 0.294 | 1.260 |
| 5 | 0.292 | 1.241 |
| 6 | 0.293 | 1.230 |
| 7 | 0.288 | 1.217 |
| 8 | 0.298 | 1.201 |
| 9 | 0.290 | 1.194 |
| 10 | 0.289 | 1.188 |
| 11 | 0.287 | 1.185 |
| 12 | 0.289 | 1.181 |
| 13 | 0.286 | 1.179 |
| 14 | 0.287 | 1.175 |
| 15 | 0.284 | 1.173 |
| 16 | 0.287 | 1.169 |
| 17 | 0.284 | 1.166 |
| 18 | 0.280 | 1.163 |
| 19 | 0.283 | **1.160** |

**结论**：
- R0 eval_loss=1.734，比 B0 的 1.174 高很多——Top-k 1% 稀疏化在第一轮引入了显著的梯度近似误差
- 但随后快速收敛：R0=1.734 → R5=1.241 → R10=1.188 → R19=1.160
- 最终 eval_loss=1.160，比 B0 的 1.075 高 0.085（约 8%），这是 1% 稀疏化的代价
- train_loss 稳定在 0.28-0.30，未发散
- 通信量仅 72MB/轮/client，比 dense 的 1.17GB 压缩 **16.7x**

**结果文件**：`results_sketchopa/topk_coord_1791892598/round_log.json`

### 3.4 E1 SketchOPA Uniform 无 Residual（完整 20 轮）

> **方法**：完整 SketchOPA 流程（Top-k → CountSketch → SHPRG 安全聚合 → median-of-rows 恢复），uniform_param 分配，**不使用 residual buffer**。验证无误差反馈时 SketchOPA 是否可行。

| Round | train_loss | eval_loss |
|---|---|---|
| 0 | 0.312 | 1.783 |
| 1 | 0.336 | 1.737 |
| 2 | 0.305 | 1.759 |
| 3 | 0.319 | 2.191 |
| 4 | 0.344 | 2.044 |
| 5 | 0.333 | 2.633 |
| 6 | 0.396 | 2.120 |
| 7 | 0.327 | 2.391 |
| 8 | 0.444 | 4.273 |
| 9 | 0.421 | 12.355 |
| 10 | 1.138 | 13.979 |
| 11 | 0.703 | 15.672 |
| 12 | 1.928 | 15.929 |
| 13 | 3.021 | 15.163 |
| 14 | 2.391 | 18.296 |
| 15 | 3.450 | 15.570 |
| 16 | 2.957 | 15.963 |
| 17 | 3.686 | 16.034 |
| 18 | 3.022 | 16.033 |
| 19 | 3.496 | **15.548** |

**结论**：
- R0 eval_loss=1.783，与 B1 的 1.734 接近——说明单轮 CountSketch 编码+恢复的误差不大
- R1-R2 略有改善（1.737, 1.759），但 R3 开始恶化（2.191）
- **R8-R9 急剧恶化**（4.27 → 12.35），R10+ 完全崩溃（13.98 → 15.55）
- train_loss 也同步失控：R0=0.312 → R19=3.496（增大 11 倍）
- **根因**：无 residual buffer 时，每轮 CountSketch 恢复的误差直接注入模型，误差逐轮累积，最终导致模型完全崩溃
- 这证明 **residual buffer 是 SketchOPA 的必要组件**

**结果文件**：`results_sketchopa/E1_sketchopa_uniform_noresidual_20r/round_log.json`

### 3.5 E2 SketchOPA Uniform + Residual（完整 20 轮）

> **方法**：在 E1 基础上加入 residual buffer（误差反馈），uniform_param 分配。验证 residual buffer 能否稳住训练。

| Round | train_loss | eval_loss |
|---|---|---|
| 0 | 0.312 | 1.783 |
| 1 | 0.336 | 1.843 |
| 2 | 0.314 | 1.676 |
| 3 | 0.320 | 1.808 |
| 4 | 0.348 | 1.669 |
| 5 | 0.323 | 1.802 |
| 6 | 0.347 | 1.587 |
| 7 | 0.312 | **1.570** |
| 8 | 0.354 | 1.655 |
| 9 | 0.333 | 2.355 |
| 10 | 0.418 | 3.341 |
| 11 | 0.416 | 2.312 |
| 12 | 0.403 | 2.280 |
| 13 | 0.391 | 2.262 |
| 14 | 0.370 | 2.591 |
| 15 | 0.435 | 1.976 |
| 16 | 0.373 | 2.014 |
| 17 | 0.415 | 1.899 |
| 18 | 0.373 | 1.925 |
| 19 | 0.410 | **1.987** |

**结论**：
- R0=1.783（与 E1 相同——第一轮 residual buffer 为空，不影响）
- **residual buffer 成功防止了完全崩溃**：E1 R19=15.548，E2 R19=1.987
- 最佳点在 R7（eval_loss=1.570），但之后波动加大（R10=3.341，R15=1.976，R19=1.987）
- eval_loss 在 1.57-3.34 间大幅波动，**未能稳定收敛**
- train_loss 稳定在 0.31-0.44（远好于 E1 的 0.31-3.50），说明模型没有发散，但恢复质量不稳定
- **根因**：uniform_param 分配按参数量比例分配 bucket，但各层梯度的重要性差异巨大（embed_tokens 和 lm_head 占了 95%+ 的 Top-k 坐标），均匀分配导致重要层和次要层的恢复质量都不够好

**结果文件**：`results_sketchopa/E2_sketchopa_uniform_residual_20r/round_log.json`

### 3.6 E3 SketchOPA Calibrated + Residual（完整 20 轮）

> **方法**：在 E2 基础上，用 calibration 数据（E1 统计的各层 Top-k 计数）指导 sketch bucket 分配（topk_count_calibrated）。预期：重要层分到更多 bucket，恢复质量更好。

| Round | train_loss | eval_loss |
|---|---|---|
| 0 | 0.312 | 1.783 |
| 1 | 0.336 | 1.816 |
| 2 | 0.312 | 1.716 |
| 3 | 0.322 | 1.820 |
| 4 | 0.332 | 1.780 |
| 5 | 0.317 | 2.367 |
| 6 | 0.382 | 3.450 |
| 7 | 0.357 | 4.062 |
| 8 | 0.603 | 2.617 |
| 9 | 0.343 | 2.383 |
| 10 | 0.379 | 1.988 |
| 11 | 0.331 | 2.582 |
| 12 | 0.371 | 4.078 |
| 13 | 0.410 | 2.434 |
| 14 | 0.360 | 3.035 |
| 15 | 0.468 | 5.161 |
| 16 | 0.474 | 3.983 |
| 17 | 0.729 | 6.243 |
| 18 | 0.506 | 6.026 |
| 19 | 0.813 | **2.977** |

**结论**：
- **calibrated allocation 反而比 uniform 更不稳定**
- E2 的最佳 eval_loss=1.570（R7），E3 的最佳 eval_loss=1.716（R2）——calibrated 更差
- E3 的波动更大：eval_loss 在 1.72-6.24 间波动，train_loss 也更不稳定（R17=0.729，R19=0.813）
- **根因分析**：
  - calibration 数据显示 embed_tokens 和 lm_head 两层占了 95%+ 的 Top-k 坐标（各约 1.2 亿/2.5 亿总计数）
  - calibrated 分配将大部分 bucket 给了这两层，导致中间层（transformer layers）的 bucket 严重不足
  - 中间层虽然 Top-k 坐标少，但对模型质量影响大，bucket 不足导致恢复误差大
  - uniform 分配反而给了中间层更多 bucket，恢复质量更均衡

**结果文件**：`results_sketchopa/E3_sketchopa_calibrated_residual_20r/round_log.json`

---

## 4. 综合分析与结论

### 4.1 收敛性对比图（数值）

```
eval_loss 趋势（每 5 轮采样）

Round  B0(dense)  B1(topk)  E1(sk no-res)  E2(sk+res)  E3(sk+cal+res)
  0      1.174      1.734       1.783          1.783         1.783
  5      1.119      1.241       2.633          1.802         2.367
 10      1.099      1.188      13.979          3.341         1.988
 15      1.083      1.173      15.570          1.976         5.161
 19      1.075      1.160      15.548          1.987         2.977
```

### 4.2 核心结论

1. **Dense（B0）是收敛上限**：eval_loss 1.174 → 1.075，稳定收敛。所有压缩方法都应以此为基准。

2. **Top-k（B1）是压缩方法的上限**：eval_loss 1.734 → 1.160，稳定收敛。1% 稀疏化的代价约 +0.085 eval_loss。通信压缩 16.7x。这说明 Top-k 稀疏化本身是可行的。

3. **SketchOPA 无 residual（E1）完全不可行**：模型在 R9 左右崩溃，eval_loss 达到 15+。CountSketch 恢复误差在无误差反馈时逐轮累积，导致发散。

4. **Residual buffer 是必要但不充分的**：
   - E2（+residual）比 E1（no-residual）显著改善：R19 从 15.548 → 1.987
   - 但 E2 仍无法稳定收敛：eval_loss 在 1.57-3.34 间大幅波动
   - residual buffer 缓解了误差累积，但 CountSketch 恢复本身的质量波动仍会注入模型

5. **Calibrated allocation 未改善，反而恶化**：
   - E3（calibrated）比 E2（uniform）波动更大：R19 从 1.987 → 2.977
   - 根因：calibration 数据集中在 embed_tokens 和 lm_head 两层（95%+），导致中间层 bucket 不足
   - 这说明 **简单的 calibration-weighted 分配不适合层数多、重要性分布极端的模型**

6. **SketchOPA 与 Top-k 的差距**：
   - B1 Top-k 最终 1.160，E2 SketchOPA 最佳 1.570，差距 0.41
   - CountSketch 编码+恢复引入了显著的额外误差（相比直接传坐标+值）
   - 安全聚合的 SHPRG mask 本身不引入误差（smoke test 验证 err=0），误差来自 CountSketch 恢复

### 4.3 通信效率 vs 收敛性权衡

| 方法 | 通信量/轮/client | 压缩比 | 安全聚合 | 最终 eval_loss | 可用性 |
|---|---|---|---|---|---|
| Dense fp16 | 1.17 GB | 1× | ✗ | 1.075 | ✓ |
| Top-k 1% | 72 MB | 16.7× | ✗ | 1.160 | ✓ |
| SketchOPA | 192 MB | 6.3× | ✓ | 1.987（E2 最佳） | △ |

- SketchOPA 通信量是 Top-k 的 2.7 倍，但收敛性远不如 Top-k
- SketchOPA 的优势在于安全聚合（隐私保护），但当前恢复质量不足以支撑稳定训练

### 4.4 性能数据

| 方法 | 每轮耗时 | 说明 |
|---|---|---|
| B0 Dense | ~420s | 训练 + 上传 1.26GB + 下载 1.26GB |
| B1 Top-k | ~400s | 训练 + 上传 72MB + 下载 1.26GB |
| E1/E2/E3 SketchOPA | ~575s | 训练 + CountSketch 编码 + 上传 192MB + 下载 1.26GB + 恢复 |

SketchOPA 比 dense 多约 155s/轮，主要来自 CountSketch 编码和恢复计算。

---

## 5. 实现中的关键发现与修正

### 5.1 基础设施层面

| 问题 | 原因 | 修正 |
|---|---|---|
| MinIO 大文件上传超时 | `put_object` 对 2.5GB BytesIO 流 multipart upload 卡住 | 改用 HTTP 直传 coordinator（`/api/upload` + `/api/download`） |
| accelerate NCCL barrier 卡死 | 多 GPU 机器单进程模式下 NCCL barrier 等待 | 去掉 accelerate，用纯 PyTorch 单卡 |
| fp16 AdamW 参数 nan | fp16 optimizer state 溢出 | 模型用 fp32 训练，上传时转 fp16 |
| eval logits OOM | eval batch_size=8 导致 logits 矩阵过大 | eval batch_size 降到 2 |
| Client 1 下载 404 | Client 1 在 Client 0 上传初始模型前就尝试下载 | `_load_global_state` 加重试（60 次 × 10s） |
| Coordinator 聚合 race condition | 两个 client 同时上传触发重复聚合 | 加 `_aggregating` 标志保护 |

### 5.2 算法层面

| 问题 | 原因 | 修正 |
|---|---|---|
| SHPRG 矩阵 A 构建需 10+ 分钟 | Python 循环 + SHA256 逐元素派生 18.9M × 4 矩阵 | 改用 `torch.randint` + generator seed 一次性生成 |
| V100 fp16 QK matmul inf | V100 对 fp16 矩阵乘法精度不足 | `patch_qwen2_attn_fp32_qk` 强制 QK matmul 用 fp32 |
| 安全聚合 mask 域 | float 域 mask 淹没 sketch 值 | 改在 Z_q 域做 mask/unmask |
| Shamir 模逆失败 | 2^16 非素数 | 用 SHAMIR_PRIME=65521（< 2^16 最大素数） |
| Calibration 统计全为 0 | topk 方法 layout=None 导致 `count_topk_per_layer` 跳过 | 改用 `layer_meta` 直接统计 |

---

## 6. 下一步建议

### 6.1 可尝试的改进方向

1. **增大 topk_ratio**：从 1% 提高到 2-5%，增加 Top-k 覆盖率，减少残差，但通信量同比增加
2. **增加 CountSketch rows**：从 5 行增加到 8-10 行，median-of-rows 恢复更准确，但通信量同比增加
3. **混合分配策略**：对 embed_tokens/lm_head 用 Top-k 直接传输（不经 CountSketch），中间层用 CountSketch
4. **residual clip + decay**：当前 clip_norm=0, decay=1.0，可尝试 clip_norm>0 和 decay<1.0 控制残差累积
5. **分层 topk_ratio**：重要层用更高 topk_ratio，次要层用更低 topk_ratio

### 6.2 可能不值得继续的方向

- **calibrated allocation 的简单加权**：由于大模型的 Top-k 分布极端集中在 embedding 层，简单加权会导致中间层 bucket 不足
- **无 residual 的 SketchOPA**：已证明会发散

---

## 7. 附录

### 7.1 系统架构

```
Central (192.168.235.42:8090)
  └── coordinator.py（FastAPI）
       ├── /api/config                     — 返回实验配置
       ├── /api/meta                       — 接收 client 模型元信息
       ├── /api/upload                     — 接收 client 上传（HTTP 直传，本地存储）
       ├── /api/download/{key}             — 发送全局模型给 client
       ├── /api/round/{r}/status           — 查询轮次是否完成
       └── /api/round/{r}/client/{c}/eval  — 接收 eval 结果

ICC1 (192.168.206.116)                    ICC2 (192.168.205.130)
  └── client.py --client-id 0               └── client.py --client-id 1
       ├── 下载全局模型（HTTP，带重试）         ├── 下载全局模型（HTTP，带重试）
       ├── 本地训练（fp32，V100 单卡）           ├── 本地训练（fp32，V100 单卡）
       ├── Top-k → CountSketch 编码            ├── Top-k → CountSketch 编码
       ├── SHPRG mask + Shamir seed 分享       ├── SHPRG mask + Shamir seed 分享
       ├── 上传 sketch（HTTP，192MB）           ├── 上传 sketch（HTTP，192MB）
       └── eval + 上报                         └── eval + 上报
```

### 7.2 结果文件索引

| 实验 | 结果目录 |
|---|---|
| B0 Dense | `results_sketchopa/B0_dense_20r_v7/round_log.json` |
| B1 Top-k | `results_sketchopa/topk_coord_1791892598/round_log.json` |
| E1 SketchOPA uniform no-residual | `results_sketchopa/E1_sketchopa_uniform_noresidual_20r/round_log.json` |
| E2 SketchOPA uniform + residual | `results_sketchopa/E2_sketchopa_uniform_residual_20r/round_log.json` |
| E3 SketchOPA calibrated + residual | `results_sketchopa/E3_sketchopa_calibrated_residual_20r/round_log.json` |

### 7.3 代码索引

| 模块 | 文件 | 说明 |
|---|---|---|
| Coordinator | `experiments_sketchopa/coordinator.py` | FastAPI 聚合服务器 |
| Client | `experiments_sketchopa/client.py` | 训练 + 编码 + 上传 client |
| 共享工具 | `experiments_sketchopa/common.py` | 数据加载、模型加载、训练、eval |
| CountSketch | `experiments_sketchopa/sketchopa/count_sketch.py` | 编码 + 恢复 |
| Layout | `experiments_sketchopa/sketchopa/layout.py` | 层布局 + bucket 分配 |
| Top-k | `experiments_sketchopa/sketchopa/topk.py` | Top-k 选择 |
| 恢复 | `experiments_sketchopa/sketchopa/recover.py` | median-of-rows 恢复 |
| 量化 | `experiments_sketchopa/sketchopa/quant.py` | fp16/int16 量化 |
| SHPRG | `experiments_sketchopa/sketchopa/shprg.py` | 安全聚合 mask |
| Seed 分享 | `experiments_sketchopa/sketchopa/seedshare.py` | Shamir 秘密分享 |
| 单测 | `experiments_sketchopa/tests/test_sketchopa_core.py` | 16 项核心单测 |
| Smoke test | `experiments_sketchopa/tests/test_smoke.py` | 5 项端到端测试 |
