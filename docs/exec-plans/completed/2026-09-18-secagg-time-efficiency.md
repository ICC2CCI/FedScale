# SecAgg 时间效率（通用；不改量化语义、不增大传输量）

- **状态**：completed
- **创建**：2026-09-18
- **更新**：2026-09-18
- **结项**：2026-09-18
- **原则**：流程对任意 HF `state_dict` + FSDP 通用；**禁止**按某个模型（含 Qwen2.5-0.5B）写死层名、窗数、并发度。0.5B 只作回归载体。
- **约束**：
  1. 保住当前量化路径的效果：Hadamard + INT16 + Issue #1（FP32 delta、quant residual 不衰减）
  2. 每轮上传 **有效载荷字节不增大**（仍是「选中元素 × INT16」；pad 到 2 的幂的现有开销可保留，不再额外 pad）
  3. 配置化：`block_size` / `coverage_h` / `modulus_bits` 继续当旋钮，不写死
- **关联**：
  - [当前联调流程](../../algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md) §3.1
  - [SecAgg 精度 / 默认路径](../../algorithm/2026-09-15-secagg-quantization-precision-issue.md)
  - [量化调研](../../algorithm/2026-09-17-secagg-quantization-optimization-survey.md)
  - 基线跑次：无 SecAgg `results/20260914-final-clean`（稳态整轮≈88s，eval 每 5 轮）；SecAgg 精度对照 `results/202609180941`（R20 eval=0.985，整轮≈148s）；时间优化 5 轮 `results/202609181151`（R5 eval=1.364 与 941 逐轮一致，整轮≈110–134s）；时间优化 20 轮 `results/202609181406`（R20 eval=0.985，整轮≈113s）

## 结项说明

P0/P1（PERF-0/1/2/3/5/6/9/11）已落地并对照通过。0.5B 双集群上，N 窗同步链已拆掉：单 blob、通知不拉对象、server 并行 unmask、全量 `global_state` 异步写。验收：5 轮 eval 与 `202609180941` 逐轮相同；20 轮 `202609181406` R20=0.984994。

相对无 SecAgg，扣掉每轮 eval 后剩余税约 +15–20s，主要是 Hadamard+INT16+mask 计算，不是协议排队。PERF-7/8/10 只剩几秒级、且多数削不到整轮墙钟，**本计划不做**（见看板 `won't do`）。大模型 / 多 client 若 SHAKE 变成 encode 瓶颈，另开计划再做 PERF-7。

---

## 0. 怎么读

当前慢的不是「Hadamard 这个算法不能用在别的模型上」，而是实现把 **N 个 window** 串成同步链（N 随「选中参数量 / block_size」变，0.5B 验证里大约 269，7B 同比例会到数千）。

优化按 **N、字节数、RTT** 来做，不要按模型名来做。

### 不做

| 不做 | 原因 |
|---|---|
| 关掉 Hadamard 换时间 | 精度回退（旧 per-window 路径 R20=1.328） |
| 降 INT8 / 减 coverage 当「加速」 | 破效果或改实验定义；若将来要压带宽，另开计划 |
| 按 Qwen 层跳过旋转、写死 269 worker | 模型特化 |
| 把全部选中参数拼成一条超长向量再 FWHT | 总 FLOPs 变 `n log n` 更大，还可能多 pad |

### 验收总则（每条切片都要满足）

- 单元测试：`PYTHONPATH=experiments python experiments/tests/test_secagg.py`
- 回归：同一 yaml（`configs/s3r12v3-fsdp-secagg-verify.yaml`）至少 5 轮，R5 eval 与 `202609180941` 同量级（约 1.364）；改完一轮协议后建议 20 轮对照 R20≈0.985
- 传输：`upload_blocks_MiB` 不高于对照（允许计账误差）
- 计时：修正后 `encode_*` / `upload_*` **不要再互相包含**（见 PERF-0）

---

## 1. 看板

| ID | 优先级 | 状态 | 任务 | 依赖 |
|---|---|---|---|---|
| PERF-0 | P0 | **done** | 拆开 encode / upload / DH 计时，避免墙钟重复记账 | — |
| PERF-1 | P0 | **done** | `masked-window` / blob 通知不再同步 `get_bytes` | — |
| PERF-2 | P0 | **done** | N 个 window 打成 1 个 blob，字节不变（仅加极小目录头） | PERF-1 |
| PERF-3 | P0 | **done** | amax 那次 FWHT 结果复用，去掉第二次正向旋转 | — |
| PERF-4 | P1 | cancelled | 默认已是单 blob（先算完再一次 PUT）；多对象流水线不再作为默认 | PERF-1 |
| PERF-5 | P1 | **done** | MinIO 进程级线程池；假死仍 rebuild | — |
| PERF-6 | P1 | **done** | FWHT 双缓冲，每级不再 `empty_like` | PERF-3 |
| PERF-7 | P2 | **won't do** | Mask PRG：SHAKE-256 → 同 seed 的 AES-CTR/ChaCha | — |
| PERF-8 | P2 | **won't do** | GPU FWHT（同一矩阵，可选） | PERF-3 |
| PERF-9 | P2 | **done** | client 一次拉 `global_delta`；server 不再写 N 个 `agg_block` | PERF-1 |
| PERF-10 | P3 | **won't do** | DH 公钥与训练重叠；amax 仍在 delta 之后 | — |
| PERF-11 | P1 | **done** | server finalize：按窗并行 unmask/iHadamard；`global_delta` 写完即标记 `done`；全量 `global_state` 异步 PUT | PERF-9 |

**已合入（2026-09-18）**：单 blob 上传 + 通知不拉对象 + 复用 FWHT + 拆分计时 + MinIO 常驻线程池 + FWHT 双缓冲 + 一份 `global_delta` 回写 + server 并行 unmask + 全量 state 异步写盘。单元测试 `experiments/tests/test_secagg.py` 已过。**5 轮对照 `202609181151` 已过**（见 §5）；**20 轮 `202609181406` 已过**（R20=0.984994，见 §6）。

0.5B 上墙钟只是标尺；**同一套改动必须在 window 数变多时仍然成立**（用更大 `coverage` 或假 window 压测，不必真上 7B 才能合入）。

---

## 2. 任务说明

### PERF-0 计时拆分

**现状**：`encode_delta_s` 从抽 slice 计到全部 window 传完，已经包含 `upload_minio_s` 和 `secagg_dh_s`，对照时容易把 47s+41s 当成相加。

**要做**：独立字段，例如 `secagg_extract_s` / `secagg_amax_s` / `secagg_dh_s` / `secagg_compute_s`（FWHT+量化+mask） / `secagg_put_s` / `secagg_notify_s`。`round_total_s` 仍是墙钟。

**涉及**：`experiments/run_s3r12v3_fsdp.py`、`scripts/plot_s3r12v3_fsdp_run.py`

**验收**：一张图能看出计算 vs 网络 vs 同步等待；字段名不出现模型名。

---

### PERF-1 通知路径不拉对象（P0）

**现状**：client 每个 window：`put_bytes` → POST；server handler 里立刻 `minio.get_bytes(z_key)` 再回 200。同一份载荷在关键路径上走两遍，且 **串行 × N**。

**要做**：POST 只登记 `z_key`（+ `vector_len` / window_id）；GET 放到 finalize 或后台线程。handler 必须 O(1) 与对象大小无关。

**涉及**：`experiments/server/aggregation_server.py`（`secagg_block_uploaded`）、`experiments/server/secagg_coordinator.py`

**验收**：N 增大时 notify 总时间近似 `N × RTT_http`，不再含 `N × 对象下载`；eval 不变。

---

### PERF-2 合并 blob（P0，传输量不增）

**现状**：每 window 一个 MinIO 对象 + 一次 HTTP。对象数 = N，不是字节数。

**要做**：本轮 client 的全部 `z_k` 打成 **一个** blob（或按 S3R12v3 block 打成少数几个，**个数不随 N 线性涨到数千**）。格式自描述：`window_id → (offset, length)`，便于任意 `block_size`。有效 INT16 载荷与现在相同。

兼容：可保留单 window key 读路径一段时间，用协议字段切换。

**涉及**：`experiments/shared/protocol.py`、`run_s3r12v3_fsdp.py` 的 SecAgg 上传循环、server finalize 解包

**验收**：PUT 次数 ≪ N；`upload_blocks_MiB` ≤ 对照；5 轮 eval 对齐。

---

### PERF-3 复用 Hadamard 正向结果（P0）

**现状**：`global_amax_from_slices` 已 FWHT；`mask_window` 再 FWHT 一次。error-feedback 的逆变换仍需要（下一轮 seed 会变，不能把残差留在旧旋转域）。

**要做**：amax 阶段缓存 `(y, signs)`（或只缓存 `y`），量化直接用。内存按「当前选中 window」分配，与模型总参数无关。

**涉及**：`experiments/shared/secagg_client.py`、`fixed_point.py`

**验收**：正向 FWHT 每窗每轮 1 次；数值与现路径一致（单元测试比对 `z_k` / residual）。

---

### PERF-4 encode ∥ 上传

在仍有多个对象或 CPU 编码偏慢时：队列边算边传。PERF-2 单 blob 后收益下降，但大 N 时仍可「算下一块 / 传上一块」。

**验收**：墙钟接近 `max(compute, net)`；峰值内存有上限（配置 `inflight_windows`，默认与模型无关）。

---

### PERF-5 MinIO 客户端

**现状**：`put_bytes` 每次新建 `ThreadPoolExecutor(max_workers=1)`，假死恢复逻辑正确但 N 大时开销差。

**要做**：进程级连接池；硬超时仍保留。并发 PUT 仅当多对象时用可配置 `max_parallel_puts`（默认 4–8），**不要**按某次 0.5B 的 269 写死。

**涉及**：`experiments/shared/minio_client.py`

---

### PERF-6 FWHT buffer

`fwht` 每一级 `empty_like` 新分配。改为就地或双缓冲。行为与现实现逐元素一致。

---

### PERF-7 更快的 mask PRG（P2）— won't do

SHAKE-256 生成 pairwise + self mask，代价 ∝ 选中元素数。可换成 **确定性** AES-CTR / ChaCha20，双方同 seed 即可抵消。

**结项决定**：0.5B / 2 client 上 encode 税大约 8s，换 PRG 只剩几秒；不挡当前主路径。若以后上 7B 或多 client（pairwise 随 client 数涨），另开计划。

---

### PERF-8 GPU FWHT（P2，可选）— won't do

同一 Walsh–Hadamard，设备可切 CPU/GPU。自动：张量已在 CUDA 或 `secagg_fwht_device=auto`。没有 GPU 的 Central 仍走 CPU。

**结项决定**：Central 无 GPU，server unmask 仍走 CPU；client 上还要 CPU↔GPU 搬运再打包 INT16。0.5B 窗长收益不明，还要保证逐元素一致。不做。

---

### PERF-9 聚合侧流式 + delta 形状

finalize 按窗（或按 blob 内切片）unmask → iHadamard → 写出，不要等齐全部 tensor 再算。client 拉回用 **一份** `global_delta`（或少数 block 文件），避免 N 次 `exists`+GET。

与已归档 SCALE-1（server 不整模常驻）同方向；本计划只要求 SecAgg 路径不再按 N 同步放大。

**续（PERF-11）**：`pipeline_wait_agg` 里剩下的大头是串行 unmask/iHadamard + 全量 `global_state` PUT。client 只依赖 `global_delta` 与 `block-status.done`，因此：

- unmask 按 window 并行（`SECAGG_UNMASK_WORKERS`，默认 `min(8, CPU)`，不写死窗数）
- 写完 `global_delta` 立刻 `done`；全量 state 在后台 PUT（与下一轮 in-memory apply 互斥）

---

### PERF-10 DH 与训练重叠（P3）— won't do

公钥与 delta 无关，可在训练后半段 announce；**`global_amax` / `global_scale` 仍必须等本轮 delta**，量化不能提前。预期只削 `secagg_dh_s` 量级，不是大头。

**结项决定**：日志里一端 DH≈0、另一端 2–6s，那是先到的人在等对端训练结束。整轮时间由慢的一端决定，重叠几乎削不到墙钟。不做。

---

## 3. 换模型时怎么用这张表

换模型 = 换 `--model-path`、数据、超参，**不新开一套通信协议**。本计划合入后应自动受益：

- window 数随 \(P \times r / \texttt{block\_size}\) 变，PERF-1/2 保证请求次数不跟着线性爆炸
- Hadamard 仍按窗长 pad-p2，与层名无关
- 新模型上线仍需 **该模型自己的 fp16 基线** 对照 eval（通用流程 ≠ 数值从 0.5B 自动成立）

更大模型还依赖显存 / 少写全量 `global_state` / client 数与 INT16 overflow，那些不在本计划（见 [completed 生产切片](2026-09-11-dual-cluster-to-production.md) SCALE / SEC）。

---

## 4. 预期（0.5B 标尺，非目标函数）

P0/P1 之后，对照 `202609180941`：整轮从 ~148s 落到 ~110–120s（5 轮实测 R2–R5≈110–115s；20 轮 `202609181406` 平均≈113s）。无 SecAgg 稳态 ~88s（且往往跳过每轮 eval）**不是必须打平**：扣掉「每轮 eval ~10s」后，SecAgg 税大约 +15–20s，主要是 Hadamard+INT16+mask 计算。验收以 eval + 字节 + 「N 增大时墙钟主要跟带宽/计算走、不跟 RTT×N 走」为准。

## 5. 5 轮对照（`202609181151`）

同一 yaml `configs/s3r12v3-fsdp-secagg-verify.yaml`，端口 8081。Eval 与 `202609180941` **R1–R5 逐轮相同**（R5=1.363804）。`upload_blocks_MiB` 不变（R1=139.4）。

| 轮 | eval | wait_agg（1151 / 优化前 1137 / 正式 941） | 整轮 1151 | server wait_s |
|---|---|---|---|---|
| R1 | 1.498 | **10.1s** / 27.8s / 16.1s | 134s | 7.6s（mat 1.2 + unmask 4.3 + put_delta 2.0） |
| R2 | 1.427 | **11.6s** / 28.6s / 16.1s | 115s | 7.4s |
| R3 | 1.398 | **11.9s** / 31.9s / 18.1s | 113s | 7.5s |
| R4 | 1.394 | **9.6s** / — / 18.1s | 110s | 6.3s |
| R5 | 1.364 | **10.1s** / — / 14.2s | 114s | 6.6s |

Server 临界路径：`materialize ~1s + 8-worker unmask ~4s + put_delta ~2s`。全量 `global_state` ~942 MiB 仍约 15s，但在 `done` 之后异步写，不再进 `pipeline_wait_agg`。

相对无 SecAgg `20260914-final-clean`（R2 无 eval ≈88s）：现在 R2 有 eval ≈115s。扣掉 eval 后剩余税主要是 encode ~15s vs fp16 ~7s；**上传反而更快**（单 blob ~9s vs 多 block ~16s）。

## 6. 20 轮算法验证（`202609181406`）

同一 yaml、同一套 P0/P1 实现。R20 eval=**0.984994**（与精度对照 `202609180941` 的 0.985 对齐），平均整轮≈113s。
