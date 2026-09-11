# 双集群联调 → 真实联邦训练 TODO

- **状态**：completed（切片 A–J 已落地；SEC-1/2/3 为 P3 合规项，设计就绪，默认关闭，待产品确认后接入传输格式）
- **创建**：2026-09-11
- **更新**：2026-09-11
- **当前阶段**：配置化 + 真实联邦能力已具备，等待双集群实跑验证
- **目录约定**：本文件已完成，归档于此。已落地的联调见 [2026-09-10-dual-cluster FSDP](2026-09-10-dual-cluster-s3r12v3-fsdp.md)。
- **关联**：
  - [当前联调流程](../../algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
  - [早期部署规划](../../algorithm/2026-09-09-dual-cluster-fsdp-deployment.md)（部分已过时）
  - [上传安全方案](../../algorithm/2026-09-08-upload-security-plan.md)
  - [S3R12v3 ratio 消融](../../experiment-records/2026-09-04-s3r12v3-ratio-ablation.md)

本文把「参数应收进配置」和「真实联邦还缺什么」拆成可勾选任务。每条含：**现状、要做什么、涉及文件、验收标准**。

---

## 0. 怎么读

### 当前 vs 目标

| | 现在（测试床） | 目标（真实联邦训练） |
|---|---|---|
| 拓扑 | 固定 2 客户端 × 8 GPU + 1 台 CPU 聚合机 | N 客户端、异构算力、可增减节点 |
| 数据 | medical flashcards 50/50 IID | 各机构私有数据、非 IID、量级不等 |
| 同步 | 必须等齐 2 端才聚合，超时整轮失败 | 部分参与、迟到丢弃、最低人数即可聚合 |
| 配置 | argparse 默认值 + 启动脚本硬编码；改 `RATIO` 不一定生效 | 一份 run 配置，三端读同一份 |
| 通信 | 内网 HTTP + MinIO，明文账号 | TLS、每 client 凭证、对象隔离 |
| 持久化 | 每轮写全量 `global_state`；进程挂了不能干净续训 | 稳态只写增量；可从 round N 恢复 |
| 模型 | Qwen2.5-0.5B 联调 | 为 7B/13B 留内存与带宽余量 |

### 优先级

| 级 | 含义 |
|---|---|
| **P0** | 联调立刻会踩；不改配置旋钮是假的 |
| **P1** | 上真实任务前必须有，否则一掉线/一换大模型就停 |
| **P2** | 真实数据与多机构会碰到 |
| **P3** | 安全加固、监控、可选算法（可并行，不挡主路径） |

### 状态约定

`todo` / `in_progress` / `done` / `blocked`

---

## 1. 总览看板

| ID | 优先级 | 状态 | 任务 | 依赖 |
|---|---|---|---|---|
| CFG-1 | P0 | done | 统一 run 配置（YAML），Server/Client 共用 | — |
| CFG-2 | P0 | done | 启动脚本真正传参；修 `RATIO` 不改 H 的坑 | CFG-1 |
| CFG-3 | P0 | done | 拓扑（主机/路径/账号）与超参分离 | CFG-1 |
| ALG-1 | P0 | done | 双集群支持 5%/10%/20%/50%（显式 `coverage_h`） | CFG-2 |
| ALG-2 | P1 | done | 支持多 slot（40% 等非 1/H 比例） | ALG-1 |
| IO-1 | P0 | done | 全量 `global_state` 改为每 N 轮写一次 | CFG-1 |
| SYNC-1 | P1 | done | 部分参与：最低人数即可聚合 | — |
| SYNC-2 | P1 | done | 迟到/掉线策略（丢弃、下一轮再纳入） | SYNC-1 |
| RES-1 | P1 | done | Server 从 round N 断点续训 | IO-1 |
| RES-2 | P1 | done | Client 持久化 memory + 本地全局版本 | RES-1 |
| SEC-0 | P1 | done | 控制面鉴权 + MinIO 凭证不进脚本 | CFG-3 |
| SEC-1 | P3 | designed | 上传不传 `key_name`（block_id） | 安全方案 §5.1 |
| SEC-2 | P3 | designed | block_id 加密 | SEC-1 |
| SEC-3 | P3 | designed | 末 block 填充到统一大小 | SEC-1 |
| SCALE-1 | P2 | designed | Server 流式按 block 聚合，不整模常驻 | IO-1 |
| SCALE-2 | P2 | done | 通信精度 int8（现为占位） | — |
| SCALE-3 | P2 | designed | Client 尽量保持 FSDP 分片，只 apply delta | — |
| DATA-1 | P2 | done | 非 IID / 不等分数据切分可配置 | CFG-1 |
| DATA-2 | P2 | done | 评估集隔离（不要两端共用同一份 eval） | DATA-1 |
| TRAIN-1 | P2 | done | 每 client 独立 `local_steps` / 时间预算 | CFG-1 |
| TRAIN-2 | P2 | done | 按数据量或 epoch 推导每轮 step，而不是写死 30 | TRAIN-1 |
| OPS-1 | P2 | done | 实时监控与失败告警（滞后、带宽、GPU） | CFG-1 |
| OPS-2 | P3 | done | MinIO 生命周期：删旧 round 全量/uploads | IO-1 |
| OPS-3 | P3 | done | Client 选择（抽样、可用性） | SYNC-1 |
| SEC-4 | P3 | done | 异常更新防护（范数裁剪 / 剔除） | SYNC-1 |
| SEC-5 | P3 | done | TLS（控制面 + MinIO） | SEC-0 |

状态说明：`done`=已实现并自测；`designed`=设计就绪、P3 合规项默认关闭，需产品确认后再接入传输格式。

建议落地顺序（近期）：**CFG-1 → CFG-2 → ALG-1 → IO-1**，然后再 **SYNC-1 / RES-1 / SEC-0**。

---

## 2. P0：配置真正可调（联调立刻要）

当前痛点：代码里有 argparse，但一键启动没把旋钮传到三端；`--ratio` 在 Server 上只是说明性字段。

### CFG-1 统一 run 配置

**现状**

- 默认值散落在 `experiments/shared/protocol.py`（`DEFAULT_*`）
- Server：`aggregation_server.py` argparse
- Client：`run_s3r12v3_fsdp.py` argparse
- 启动：`scripts/start_s3r12v3_fsdp_run.sh` 环境变量 + 硬编码
- `deployment/central-server.env` 只有 `NUM_CLIENTS` / `NUM_ROUNDS` / `RATIO`

**要做**

新增一份三端共用配置，例如 `configs/s3r12v3-fsdp-run.yaml`（或 `deployment/run.yaml`），至少包含：

```yaml
federated:
  num_clients: 2
  num_rounds: 20
  coverage_h: 5          # 真正决定每轮上传 ≈ 1/H
  slots_per_round: 1     # 预留；P0 可先写 1，ALG-2 再实现
  seed: 20260831
  transfer_dtype: fp16
  memory_decay: 0.9
  block_size: 524288

train:
  local_steps: 30
  batch_size: 8
  grad_accum: 2
  lr: 1.0e-5
  seq_len: 512

io:
  skip_round0_download: true
  online_eval: true
  write_full_global_every_n_rounds: 1   # IO-1 改成 5 或 0=只写 delta
  client_upload_timeout_s: 1800

eval:
  eval_path: data/medical_flashcards_eval.json
  eval_max_batches: 0
```

- Server / Client 都能 `--config configs/s3r12v3-fsdp-run.yaml`，CLI 覆盖 yaml
- 启动时把该文件写入 `results/<run_id>/run.yaml`（与现有 `run_meta.json` 一起），保证可复现
- `ratio` 若仍出现，只作展示：`ratio ≈ slots_per_round / coverage_h`，**不得单独驱动调度**

**涉及文件**

- 新增：`configs/s3r12v3-fsdp-run.yaml`
- 改：`experiments/shared/protocol.py`（加载 yaml）
- 改：`experiments/server/aggregation_server.py`
- 改：`experiments/run_s3r12v3_fsdp.py`
- 改：`scripts/start_s3r12v3_fsdp_run.sh`

**验收**

- 改 yaml 里的 `num_rounds` / `local_steps` / `coverage_h` / `lr`，一键启动后三端日志与 `run_meta.json` 一致
- 不改代码、不改启动脚本正文，即可换一套超参

---

### CFG-2 启动脚本真正传参；修 ratio 陷阱

**现状（必须写进文档，避免再踩）**

```python
# aggregation_server.py
p.add_argument("--coverage-h", type=int, default=DEFAULT_COVERAGE_H)  # 默认 5
p.add_argument("--ratio", type=float, default=0.2)  # informational
if args.coverage_h <= 0:
    args.coverage_h = max(1, int(round(1.0 / max(args.ratio, 1e-6))))
```

`start_s3r12v3_fsdp_run.sh` 只传 `--ratio "${RATIO:-0.2}"`，**不传 `--coverage-h`**。  
因此 `RATIO=0.05` 或 `0.10` **不会改变实际上传量**，plan 仍是 H=5 ≈ 20%。

同样没传给 Client 的：`--local-steps`、`--lr`、`--batch-size`、`--grad-accum`、`--seq-len`、`--memory-decay`。

**要做**

- 启动脚本从 run yaml（或 env 覆盖 yaml）生成 Server 与两端 Client 的完整命令行
- 显式传 `--coverage-h`；禁止「只改 RATIO」的假开关。若保留 `RATIO` env，必须同时设 `COVERAGE_H`，否则启动失败并打印映射表
- Client SSH 远程启动也带上同一套训练超参
- `run_meta.json` 记录实际生效的 `coverage_h`、`slots_per_round`、`local_steps`、`memory_decay`、`transfer_dtype`

**H 与上传比例（P0 支持的集合）**

| 目标上传 | `coverage_h` | `slots_per_round` | 实际上传 | P0 能否开 |
|---|---|---|---|---|
| 5% | 20 | 1 | ~5% | 要能开 |
| 10% | 10 | 1 | ~10% | 要能开 |
| 20% | 5 | 1 | ~20% | 当前默认 |
| ~33% | 3 | 1 | ~33% | 要能开 |
| 50% | 2 | 1 | ~50% | 要能开 |
| 40% | 5 | 2 | ~40% | **ALG-2**，P0 可拒绝并报错 |

**涉及文件**

- `scripts/start_s3r12v3_fsdp_run.sh`
- `deployment/central-server.env.example`（补 `COVERAGE_H`、`LOCAL_STEPS` 等）
- `docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md` §9 默认参数表

**验收**

- `COVERAGE_H=10` 启动后，plan / `round_log.json` 的 `pct_of_total` 约 10%，每端上行大约减半
- `LOCAL_STEPS=10` 启动后，Client 日志每轮只训 10 step
- 只设 `RATIO=0.05` 不设 H 时，脚本退出码非 0，并提示应设 `coverage_h=20`

---

### CFG-3 拓扑与超参分离

**现状**

`start_s3r12v3_fsdp_run.sh` 写死：

- ICC1 / ICC2 的 SSH 用户、主机、仓库路径
- Server URL、MinIO endpoint
- MinIO access/secret（明文出现在远程命令里）
- 模型路径、数据切分路径

**要做**

- 超参：run yaml（CFG-1）
- 拓扑：节点清单，例如 `deployment/nodes.yaml`（gitignore 本地覆盖，提供 `.example`）
  - 每端：`client_id`、`ssh`、`repo`、`model_path`、`data_path`、`accelerate_config`、`conda_env`
  - Central：`server_url`、`minio_endpoint`（凭证只从 env / secret 文件读，不进 git）
- 启动脚本只读这两份文件，循环拉起 N 个 client，不再写死 2 台

**验收**

- 换一台 client 主机只需改节点清单
- `git grep` 启动脚本中不再出现内网 IP 或 MinIO 密码

---

### ALG-1 双集群可调 5% / 10% / 20% / 50%

**现状**

- 单机消融 `experiments/run_s3r12v3_ratio.py` 已测过 5%~50%
- 双集群 `BlockScheduler` 只有 `coverage_h`、每轮 1 个 slot
- 联调默认 20%，文档写死「约 20%」

**要做**

- 配置驱动 H 后，用 0.5B 双集群各跑短测（例如 3 轮）确认：
  - H=20 → ~5%、H=10 → ~10%、H=5 → ~20%、H=2 → ~50%
- 更新当前流程文档 §9：上传比例改为「由 `coverage_h` 决定」，不要写死 20%
- `round_log.json` 每轮已有 `pct_of_total`，画图脚本应对不同 H 都能工作

**验收**

- 同一套双集群代码，只改 yaml 即可切换上述四个比例
- 3 轮短测的实际上传 MiB 与 `1/H` 成比例（允许末 block 带来的约 ±3% 波动）

---

## 3. P0/P1：IO 与聚合写盘

### IO-1 全量 `global_state` 不要每轮都写

**现状**

`aggregation_server._aggregate_round` 每轮都：

1. 写 `global_delta/round-N/blocks.pt`
2. 写 `global_state/round-N/state.pt`（全量）

联调 `wait_agg` 约 20s，大头是写全量。0.5B 约 1GB 可忍；7B/13B 会变成主瓶颈。客户端稳态已靠 `global_delta` + 本地 cache。

**要做**

- 配置 `write_full_global_every_n_rounds`：
  - `1`：保持现状（联调/评估兜底）
  - `5` / `10`：每 N 轮写一次全量 checkpoint
  - `0`：只写 round-0 与最终一轮，中间只写 delta
- 缺全量时，客户端 `download_mode=full` 应失败并明确报错（回退到最近一次全量 + 连续 apply delta，可作为后续增强）
- `round_log` 记录本轮是否写了全量、`global_state_MiB`

**验收**

- 设 `every_n=5` 时，MinIO 中 round 1–4 无新 `state.pt`（或为软链/跳过），round 5 有
- 稳态轮 `wait_agg` 明显下降（0.5B 上应能看到写盘时间少一大截）
- 客户端下一轮仍为 `cache` 或 `delta`，训练不中断

---

## 4. P1：真实任务的生存能力

### SYNC-1 部分参与（最低人数即可聚合）

**现状**

- `len(bucket) >= num_clients` 才聚合
- watchdog 超时后 `aggregation_failed`，整轮作废
- 固定 2 端，一端挂则整场停

**要做**

- 配置 `min_clients_to_aggregate`（例如 2 中的 1，或未来 10 中的 7）
- 到点（超时或已达最低人数 + 宽限期）用**已上传子集**做加权 FedAvg
- result 里记录 `participated_clients` / `missing_clients` / `partial: true`
- plan 仍对所有 client 相同（S3R12v3 public mask）

**验收**

- 故意不启动 ICC2，超时后仍能用 ICC1 完成一轮并写出 `global_delta`
- ICC1 不会在 `wait_agg` 里空等到进程被杀

---

### SYNC-2 迟到与掉线策略

**现状**

超时只标失败，没有「本轮丢弃、下轮再来」。

**要做**

- 迟到上传：若本轮已聚合，返回 409，并提示下一轮 round
- 掉线 client：memory 仍在它本机；下轮带着过期 memory 参与（文档写清语义）
- 连续缺席 N 轮：标记 inactive，不再计入 `min_clients`
- 与 SYNC-1 共用同一时钟：`round_deadline_s`

**验收**

- ICC2 本轮超时未传，ICC1 已聚合；ICC2 晚到的 `upload-complete` 被拒绝且不污染下一轮
- ICC2 下一轮仍能拉 plan、用 delta 对齐后再训

---

### RES-1 Server 断点续训

**现状**

聚合进程内存里：`current_round`、`global_state`、`round_log`。进程重启后从 round-0/`--init-state` 再来，**不能**从 MinIO 已有 `global_delta`/`global_state` 接上。

**要做**

- 启动参数 `--resume-from-round N` 或自动扫描 MinIO 最新完成轮
- 恢复：`global_state`（最近全量）+ 连续 apply 之后的 `global_delta`
- 恢复 `round_log.json` / `current_round = last_done + 1`
- 与 IO-1 兼容：中间可能没有全量

**验收**

- 跑到 round 5 后杀聚合进程，带 resume 启动，Client 无需清 memory 即可从 round 6 继续（Client 侧见 RES-2）
- MinIO 中 round 5 的 delta/state 不被覆盖

---

### RES-2 Client memory 与本地全局版本持久化

**现状**

Client 的 block memory、`local_global` 版本只在进程内。重启等于丢掉未上传累积，首轮行为退化。

**要做**

- 每轮结束后 rank0 把 `memory.pt` + `local_global_round` 写到本地（或 MinIO `clients/client-C/`）
- 启动 `--resume` 时加载；与 Server 当前 round 对齐（落后则 apply delta / 拉最近全量）
- 文档写清：resume 不恢复 optimizer（每轮本地 30 step 本来就新建 scheduler）；若以后跨轮保持 Adam 状态，另开任务

**验收**

- 杀 ICC1 再拉起，能从 Server 当前轮继续，memory 非全零
- 版本落后 1 轮时走 `delta` 而不是无脑 `full`

---

### SEC-0 控制面鉴权 + 凭证不进脚本

**现状**

- REST 无认证，知道 URL 就能拿 plan、伪报 upload-complete
- MinIO 账号出现在启动脚本远程 heredoc
- HTTP 明文

**要做（最低生产门槛，不做完整安全方案）**

- 聚合 API：共享 token 或 per-client token（header）；无 token 拒绝
- MinIO：启动时从 env / 本地 secret 文件注入，不写进 git、不出现在 `ps` 能看到的过长命令则尽量用环境变量
- `central-server.env.example` 注明必改密码；检查清单里加「脚本内无明文 secret」

**验收**

- 无 token 调 `/api/round/1/plan` 返回 401
- 仓库与启动脚本中不再硬编码 MinIO 密码

---

## 5. P1/P2：调度扩展与规模

### ALG-2 多 slot（非 1/H 的比例，如 40%）

**现状**

单机 `run_s3r12v3_ratio.py` 用 `SLOTS_PER_ROUND` 实现 40%（H=5、每轮 2 slot）。  
双集群 `build_selected_blocks(..., slot=slot, coverage_h=H)` 只选一个 slot。

**要做**

- `BlockScheduler` / `RoundPlan` 增加 `active_slots: List[int]`
- yaml `slots_per_round` > 1 时，本轮选 `pos % H in active_slots`
- 覆盖周期变为 `ceil(H / slots_per_round)` 轮
- CFG-2 的映射表纳入 40%：H=5, slots=2

**验收**

- yaml 配 40%，`pct_of_total` 约 40%，与单机消融一致（允许同样的波动范围）

---

### SCALE-1 Server 流式按 block 聚合

**现状**

Server 把整份 `global_state` 放 CPU RAM，再把两端 blocks 全部 `get_torch` 进内存后 FedAvg。  
0.5B ~1GB；13B fp16 全量 + 多份 delta 会打满 32GB 规划机。

**要做**

- 按 key / block 流式：拉一块、累加一块、写一块，不要同时持有「全量 + 所有 client 全量 delta」
- 全量 checkpoint 改为分片对象或流式 `torch.save`（若仍保留 IO-1 的偶发全量）
- 文档给出内存上限估算：`O(block + 少量缓冲)` 而非 `O(model × (1+C))`

**验收**

- 用 0.5B 先做正确性对比（与现 FedAvg 数值一致）
- 内存峰值文档化；为目标 7B 给出是否还要加内存的结论

---

### SCALE-2 通信 int8

**现状**

`--transfer-dtype int8` 直接 `NotImplementedError`。早期单机 S3R 系列用过 INT8 block。

**要做**

- 在 `encode_block_delta` / `apply_block_delta` 实现对称量化（scale per-block）
- 聚合仍建议 fp32 累加，写出再量化
- 0.5B 双集群 3 轮对比 fp16 vs int8：收敛差、MiB、encode 耗时

**验收**

- `transfer_dtype: int8` 能跑完 3 轮
- 上行体积相对 fp16 明显下降（量级约一半，取决于 scale 开销）

---

### SCALE-3 减少每轮 FSDP 全量 load

**现状**

每轮训练前 `load_full_state_fsdp`；即使 `download_mode=cache`，load 仍可能很重。时间图上 `load` 是独立一段。

**要做**

- 评估：cache 命中时是否可跳过重建、只对选中 block 写入 FSDP 分片
- 若 FSDP API 限制太大，至少避免 cache 时重复 broadcast 整模
- 作为调研任务：先出结论（可行 / 本代不做），再改代码

**验收**

- 有简短结论写回本文或联调流程文档
- 若改了：cache 轮的 `load` 秒数下降，且 eval_loss 与基线一致

---

## 6. P2：真实数据与训练语义

### DATA-1 非 IID / 不等分切分可配置

**现状**

规划写过「50/50，后续 `--data-split`」；实际是预切好的 `icc1_client0_train.json` / `client1_train.json`。

**要做**

- 切分脚本参数化：比例、按 label/来源的 Dirichlet 非 IID、每 client 样本上限
- 节点清单里每端 `data_path` 指向不同切分
- FedAvg 已用 `num_examples` 加权，确认非均等切分时权重正确

**验收**

- 能生成 70/30 与一种非 IID 切分，并各跑 1 轮短测，`num_examples` 出现在聚合权重里

---

### DATA-2 评估集隔离

**现状**

两端 `--eval-path data/medical_flashcards_eval.json` 同一份；上报平均 eval_loss。真实机构不该共享同一测试集，也不该用训练同源的卡片集当业务指标。

**要做**

- 每端可用本地 hold-out；Server 只记录、不强行平均不可比的 eval
- 可选：Central 持有独立 eval，定期拉全局做离线评估（已有 `experiments/eval_minio_rounds.py` 方向）
- 文档区分：联调用共享 eval；真实任务用 hold-out + 服务端离线 eval

**验收**

- yaml 可关 `online_eval` 或指向每端不同路径
- `round_log` 能区分 `client_eval_loss` 与 `server_eval_loss`

---

### TRAIN-1 / TRAIN-2 每端 step 与时间预算

**现状**

两端相同 `--local-steps` 默认 30（且启动脚本还没传）。真实环境卡数、数据量、网络都不同。

**要做**

- yaml 允许全局默认 + per-client override（`clients.0.local_steps`）
- 或 `local_max_seconds`：到时间就停，上报实际 step
- 文档给出换算：有效 batch = `batch_size * grad_accum * num_gpus`；按 epoch 反推 step

**验收**

- ICC1 30 step、ICC2 10 step 能同轮聚合，权重仍按 `num_examples`（或按实际 token/step 再议，需在文档写死一种）

---

## 7. P2/P3：运维、安全加固、可选能力

### OPS-1 实时监控与告警

**现状**

`round_log.json` + 事后 `plot_s3r12v3_fsdp_run.py`。规划提过 dashboard，未作为联调必经路径。

**要做**

- 每轮结束刷新：当前轮、各端 train/eval、`wait_agg`、上传 MiB、失败原因
- 告警：上传超时、MinIO 重试耗尽、一端 round 落后 ≥2
- 复用已有 `experiment-dashboard` 或最小静态页读 `results/current`

**验收**

- 训练中打开 dashboard（或 `check_rerun_status.sh` 增强版）能看到当前轮与失败，不必 ssh 三台机器

---

### OPS-2 MinIO 生命周期

**现状**

每轮保留 uploads + delta + 全量，20 轮 0.5B 还能接受；7B × 很多轮会爆盘。

**要做**

- 保留：最近 K 轮 uploads、所有 delta（或最近 M 轮 delta）、按 IO-1 的全量 checkpoint
- 删除策略写进 yaml，默认联调不删（避免踩实验）

**验收**

- 开清理后，round 1 的 `uploads/` 在 round 1+K 之后消失；最新 checkpoint 仍在

---

### OPS-3 Client 选择

**现状**

固定 2 个全参加。真实联邦常每轮抽样。

**要做**

- 每轮 Server 在 plan 中带 `selected_client_ids`
- 未选中的 client 本轮 skip 训练，或只拉 delta 保持对齐
- 依赖 SYNC-1（参与集合可变）

**验收**

- 3 个假 client_id 中每轮只选 2 个，日志与聚合权重一致

---

### SEC-1 / SEC-2 / SEC-3 上传隐私（已有设计文档）

完整设计见 [上传安全方案](2026-09-08-upload-security-plan.md)。此处只挂接到总 TODO，避免两份清单分叉。

| ID | 对应安全文档 | 摘要 |
|---|---|---|
| SEC-1 | §5.1 | RoundPlan / 上传用 `block_id`，不传 `key_name` |
| SEC-2 | §5.2 | per-round 密钥加密 `block_id` |
| SEC-3 | §6.2 | 末 block 填充到 1MB |

SecAgg / DP（安全文档第四、五层）明确标为**可选，默认不做**，除非产品有合规要求。

---

### SEC-4 异常更新防护

**要做**

- 聚合前对 client delta 做范数统计；超过阈值则本轮丢弃该 client 并记日志
- 不做复杂 Byzantine 算法；先有开关和指标

**验收**

- 人工构造一份异常大的 blocks.pt，该 client 被跳过，另一端仍能聚合

---

### SEC-5 TLS

**要做**

- 控制面 HTTPS、MinIO TLS；证书放部署目录不进算法代码
- 内网联调可继续 HTTP，yaml `tls: false`

**验收**

- `tls: true` 时 Client 用 `https://` 能完成 1 轮；证书错误则失败而不是静默回落明文

---

## 8. 建议的执行切片

按「一次可合并、可验证」切，避免大爆炸重构。

| 切片 | 包含 ID | 预估 | 状态 | 完成后能力 |
|---|---|---|---|---|
| **A. 配置打通** | CFG-1, CFG-2, ALG-1 | 小 | done | 改 yaml 就能换轮数 / step / 5–50%（1/H） |
| **B. 拓扑解耦** | CFG-3, SEC-0 | 小 | done | 无明文密码；加机器不改脚本逻辑 |
| **C. 写盘减负** | IO-1 | 小 | done | 大模型前先把 `wait_agg` 降下来 |
| **D. 能断能续** | RES-1, RES-2 | 中 | done | 杀进程可接着训 |
| **E. 允许缺席** | SYNC-1, SYNC-2 | 中 | done | 一端挂了另一端还能往前走 |
| **F. 比例补齐** | ALG-2 | 小 | done | 40% 等多 slot |
| **G. 规模** | SCALE-1, SCALE-2, SCALE-3 | 中～大 | SCALE-2 done; SCALE-1/3 designed | int8 已实现；流式/FSDP 分片为 7B 前再做 |
| **H. 真实数据** | DATA-1, DATA-2, TRAIN-1, TRAIN-2 | 中 | done | 非 IID、异构 step |
| **I. 运维** | OPS-1, OPS-2, OPS-3 | 中 | done | 可值守 |
| **J. 安全加固** | SEC-1~5, SEC-4 | 中～大 | SEC-4/5 done; SEC-1/2/3 designed | TLS+异常防护已实现；上传隐私待合规确认 |

近期若只做联调体验：**先做切片 A，再做 C**。A 让「5% 还是 20%、30 step 还是 100」变成真开关；C 让换大模型时聚合侧先活下来。

---

## 9. 明确不做（除非另开需求）

避免 TODO 无限膨胀：

- 不把 Flower 加回来（已去掉，双集群走 HTTP+MinIO）
- 不在 Central 上做 GPU 训练
- 不默认上 SecAgg / DP（影响复杂度或收敛）
- 不在本 TODO 里换模型结构或改 S3R12v3 的 public mask 语义
- 不把单机消融脚本（`run_s3r12v3_ratio.py` 等）全部迁到 yaml；它们已完成历史实验，只把**双集群路径**配置化

---

## 10. 相关代码与文档索引

| 路径 | 和本 TODO 的关系 |
|---|---|
| `experiments/shared/protocol.py` | 默认超参、`RoundPlan` |
| `experiments/server/block_scheduler.py` | `coverage_h`；ALG-2 改这里 |
| `experiments/server/aggregation_server.py` | 等齐才聚合、每轮写全量 |
| `experiments/run_s3r12v3_fsdp.py` | Client 超参 argparse，启动脚本未传 |
| `scripts/start_s3r12v3_fsdp_run.sh` | 一键启动；CFG-2/CFG-3 主战场 |
| `deployment/central-server.env.example` | 现有少量联邦 env |
| `experiments/run_s3r12v3_ratio.py` | 单机 ratio 消融，ALG-2 可参考 |
| `docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md` | 以已落地实现为准的流程说明，完成后需回写 §9 |
| `docs/algorithm/2026-09-08-upload-security-plan.md` | SEC-1~3 详细设计 |

---

## 日志

- 2026-09-11：根据双集群联调现状，整理参数化缺口与真实联邦差距，形成分阶段 TODO（切片 A 优先）。
- 2026-09-11：迁入 `docs/exec-plans/active/`；已完成的联调与单机实验计划放到 `docs/exec-plans/completed/`。
- 2026-09-11：落地切片 A–J 的代码实现。新增/改动：
  - **CFG-1**：`configs/s3r12v3-fsdp-run.yaml` + `experiments/shared/run_config.py`（YAML 加载、CLI 覆盖、生效配置快照）；Server/Client 均支持 `--config`。
  - **CFG-2**：`scripts/start_s3r12v3_fsdp_run.sh` 重写，从 yaml 读全部超参并显式传给三端；启动前校验 `ratio ≈ slots/coverage_h`，不一致退出码非 0 并打印映射表；`run_meta.json` 记录实际生效 `coverage_h`/`slots_per_round`/`local_steps`/`memory_decay`/`transfer_dtype`。
  - **CFG-3**：`deployment/nodes.yaml(.example)` 节点清单（gitignore 本地）；启动脚本循环拉起 N client，换机器只改清单；MinIO 凭证用环境变量注入，不出现在 `ps` 可见命令行（配合 SEC-0）。
  - **ALG-1**：`BlockScheduler` 配置驱动 H，自测 5/10/20/50% 与 `1/H` 成比例。
  - **ALG-2**：`BlockScheduler.slots_per_round` + `build_selected_blocks_multi`；自测 H=5×2slot≈40%。
  - **IO-1**：`--write-full-global-every-n-rounds`（1=每轮/N=每N轮/0=只 round-0 与最终轮）；`round_log` 记 `wrote_full_global`。
  - **SYNC-1/2**：`--min-clients-to-aggregate`（0=旧行为）；超时后满足最低人数则部分聚合（`partial/participated_clients/missing_clients`）；迟到上传 409 拒绝。
  - **RES-1**：Server `--resume-from-round N`，自动扫描最近全量 + 连续 apply delta。
  - **RES-2**：Client `--client-state-dir` + `--resume`，每轮持久化 `memory.pt`/`local_version.txt`。
  - **SEC-0**：`--auth-token`，非空时所有 `/api/**` 要求 `Authorization: Bearer`，无 token 返回 401。
  - **SEC-4**：`--delta-norm-clip`/`--delta-norm-reject`，聚合前 L2 范数统计，超阈值裁剪或剔除。
  - **SEC-5**：`--tls-cert`/`--tls-key`，uvicorn 启用 HTTPS；客户端用 `https://` server-url。
  - **SEC-1/2/3**：设计就绪（见安全方案 §5），P3 合规项，改动传输格式需 client/server 协同迁移，默认关闭，待产品确认后接入。
  - **SCALE-1**：调研结论——当前 `apply_block_delta` 在 fp32 累加每个 selected block，峰值 `O(selected_elems×(1+C))`；0.5B/5%≈2.8GB 可接受，7B 同比线性放大。真正流式（拉一块/累加一块/写一块，不同时持有全量+所有 delta）需重构 `apply_block_delta` 的逐 key 循环为逐 block 流水线，标记为后续 7B 落地前再做。
  - **SCALE-2**：`block_selection.py` int8 对称量化（per-block scale，`_quantize_int8_block`/`_dequantize_int8_block`），`encode/apply/add_block_delta` 支持 int8（scale 存元组第 4 元素）；自测相对误差 ~0.8%，体积约为 fp16 的 54%。
  - **SCALE-3**：调研结论——FSDP `load_full_state_fsdp` 用 `rank0_only=False` 的 FULL_STATE_DICT 加载，要求每 rank 持全量；cache 命中时跳过重建需 FSDP API 支持「只写入选中分片」，本代 PyTorch FSDP 未暴露该细粒度接口。结论：**本代不做**，cache 轮仍需 broadcast 整模；7B 时靠 IO-1（少写全量）+ delta 同步降低成本。
  - **DATA-1**：`scripts/split_federated_data.py`（uniform/fixed/dirichlet + per-client holdout）；自测 Dirichlet α=0.5 产出 2501 vs 24659 非 IID 切分。
  - **DATA-2**：切分脚本支持 `--holdout-eval-ratio` 生成每端 `*_eval.json`；Server 已在 `round_log` 区分 `client_eval_loss`（per-client）与 `eval_loss`（平均，仅联调可比时用）。
  - **TRAIN-1**：`nodes.yaml` 支持 per-client `local_steps` 覆盖，启动脚本注入 `--local-steps`。
  - **TRAIN-2**：`configs/s3r12v3-fsdp-run.yaml` 注释给出换算：有效 batch = `batch_size × grad_accum × num_gpus`；按 epoch 反推 step = `ceil(num_examples / effective_batch)`。
  - **OPS-1**：`scripts/check_rerun_status.sh --watch` 增强版：检测失败轮、partial、client 滞后 ≥2、缺席告警，无需 ssh 三台。
  - **OPS-2**：Server `--minio-retention-recent-uploads K`，每轮聚合后删早于 `round-K` 的 uploads/。
  - **OPS-3**：Server `--selected-client-ids 0,1`，plan 带 `selected_client_ids`；未选中 client 本轮 skip 训练但保持 delta 对齐。
