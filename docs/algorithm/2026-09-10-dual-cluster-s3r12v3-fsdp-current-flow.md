# 双集群 S3R12v3 + FSDP 当前联调流程说明

- **日期**：2026-09-10
- **对应实现**：`deploy/central-server-minio`（增量 `global_delta`、fp16 传输、在线 eval、MinIO 假死恢复）
- **参考跑次**：`results/202609101345/`
- **相关代码**：`experiments/run_s3r12v3_fsdp.py`、`experiments/server/aggregation_server.py`、`experiments/shared/minio_client.py`
- **一键启动**：`bash scripts/start_s3r12v3_fsdp_run.sh`

本文用白话说明：**ICC1、ICC2、Central Server 各自做什么，数据怎么传**。算法细节见 [S3R12v3](2026-09-04-s3r12v3-block-uniform.md)；早期部署规划见 [双集群规划](2026-09-09-dual-cluster-fsdp-deployment.md)（部分内容已过时，以本文为准）。执行跟踪：已落地部分见 [completed 联调结项](../exec-plans/completed/2026-09-10-dual-cluster-s3r12v3-fsdp.md)；未完成项见 [active TODO](../exec-plans/active/2026-09-11-dual-cluster-to-production.md)。

---

## 1. 三角色一句话

| 角色 | 节点 | 干什么 |
|------|------|--------|
| **Central Server** | 聚合节点（无 GPU） | 跑聚合服务 + MinIO。决定「这一轮传哪些 block」、做 FedAvg、写回全局增量/全量权重、记日志 |
| **ICC1（Client 0）** | 训练节点 A（多卡 GPU） | FSDP 本地训练；用自己的一半数据；上传选中 blocks |
| **ICC2（Client 1）** | 训练节点 B（多卡 GPU） | 同上，另一半数据 |

> 具体主机名 / IP / 账号口令见本地运维配置（如 `deployment/central-server.env`、`deployment/dual-cluster-nodes.md`），**本文不记录内网地址与密钥**。

**原则**：大模型权重走 **MinIO（对象存储）**；小控制消息走 **HTTP REST**（问轮次、拿 plan、通知上传完、问聚合结果）。

```
        ICC1 (Client 0)                 ICC2 (Client 1)
        8×GPU 训练                       8×GPU 训练
              │  HTTP 控制面                    │
              │  S3 大文件                      │
              └────────────┬────────────────────┘
                           ▼
              ┌────────────────────────────┐
              │     Central Server          │
              │  Aggregation Server :8080   │
              │  MinIO               :9000  │
              └────────────────────────────┘
```

---

## 2. 启动前准备（只做一次或换模型时做）

1. Central 上 MinIO 健康、bucket 就绪。
2. 把初始权重放到 MinIO：`global_state/round-0/state.pt`（可用 `scripts/bootstrap_initial_state.sh`）。
3. 两端本地都有同一份底座模型目录（如 `model/Qwen/Qwen2.5-0.5B`）和各自训练数据切分。
4. 用启动脚本拉起：Server 聚合进程 → ICC1 → ICC2。当前默认：
   - `--transfer-dtype fp16`
   - `--skip-round0-download`（首轮不拉 ~1GB 全量，用本地模型当 round-0）
   - `--online-eval`（每轮训练后算 eval_loss 并上报）

结果写入：`results/YYYYMMDDHHMM/`（含 `round_log.json`、`figures/`、`logs/`）。

---

## 3. 一轮训练在干什么（核心）

把一轮想成一条流水线。两端 **并行** 做左边客户端步骤；Server 在「两端都上传完」之后做聚合。

```
【两端各自，大致同时】
  ① 问 Server：现在第几轮？拿本轮 plan（选哪些 block）
  ② 准备全局权重（见下一节「下载/缓存」）
  ③ 装进 FSDP 模型（多卡 load/broadcast）
  ④ 本地训练若干 step
  ⑤ 在线 eval（可选）→ 得到 eval_loss
  ⑥ 算更新：delta + memory → 只编码 plan 选中的 ~20% blocks
  ⑦ 上传到 MinIO：uploads/round-N/client-C/blocks.pt
  ⑧ HTTP 通知 Server：我传完了（附带 train/eval loss、分段耗时）

【Server】
  ⑨ 收齐 2 个 client 后：从 MinIO 拉两端 blocks
  ⑩ FedAvg 应用到 global_state
  ⑪ 写出：
        global_delta/round-N/blocks.pt   ← 本轮增量（客户端下一轮主要靠它）
        global_state/round-N/state.pt    ← 全量备份/兜底
  ⑫ 标记本轮 done；记入 round_log.json

【两端各自】
  ⑬ 轮询 /result 直到 done（这段时间叫 wait_agg）
  ⑭ 立刻下载并 apply 本轮 global_delta → 本地缓存版本变成 N
  ⑮ 进入下一轮（多数时候训练前 download=0，即 cache）
```

对应时间图 `figures/time_breakdown.png`（以 ICC1 为例）：

| 图上名字 | 实际阶段 |
|----------|----------|
| download | 训练前拉全局（cache 时为 0） |
| load | 权重装进 FSDP |
| train | 本地训练 |
| encode | 编码选中 blocks |
| upload | 传到 MinIO |
| wait_agg | 等对端 + 等 Server 聚合写完 |
| post_delta | 拉并应用本轮 global_delta |
| round_wall（虚线） | Server 视角整轮墙钟 |

说明：在线 eval 耗时目前未画进堆叠条，但已写入日志/上报。

---

## 4. 「全局权重」怎么在三端之间流动

### 4.1 首轮（Round 1）特殊：跳过 round-0 下载

两端启动后若开启 `--skip-round0-download`：

- 用**本地 model-path** 抽出一份 state，当作 `local_global`，版本号记为 `0`
- **不再**从 MinIO 下载 `global_state/round-0/state.pt`（省掉约 1GB）
- Server 自己仍从 MinIO 的 round-0 初始化聚合用的 global（两端底座模型需与之一致）

日志里会看到：`Seeded local_global ... skip round-0`，`mode=local_base`。

### 4.2 稳态轮次：优先 cache，其次 delta，最后全量

客户端本地始终尽量持有「上一轮聚合后的全局」：

| 模式 | 何时 | 从 MinIO 拉什么 |
|------|------|-----------------|
| **cache** | 本地版本已经是「本轮需要的上一版」 | 不拉（0 MiB） |
| **delta** | 本地落后一轮，且存在 `global_delta` | 只拉增量 blocks 并 apply |
| **full** | 冷启动/断档/缺 delta | 拉完整 `global_state/round-K/state.pt` |
| **local_base** | 仅首轮 skip-round0 | 不拉，用本地模型 |

聚合刚结束时两端会立刻 `post_delta`，所以**下一轮开始时通常已是 cache**。

### 4.3 上传的是什么（不是全模型）

每轮 Server 通过 plan 指定约 **20%** 的 block。客户端上传：

```
uploads/round-N/client-0/blocks.pt   ← ICC1
uploads/round-N/client-1/blocks.pt   ← ICC2
```

内容大致是：选中 block 的参数增量（当前联调为 **fp16**）+ `num_examples` / `train_loss` / `eval_loss` 等元数据。

通信量数量级（0.5B、ratio≈20%、fp16）：每端上行约 **220–260 MiB/轮**。

---

## 5. MinIO 里有哪些关键对象

```
fedscale-bucket/
├── global_state/
│   ├── round-0/state.pt          # 初始全量（Server 用；客户端可跳过下载）
│   ├── round-1/state.pt          # 第 1 轮聚合后全量（兜底）
│   └── round-N/state.pt
├── global_delta/
│   ├── round-1/blocks.pt         # 第 1 轮聚合增量（客户端稳态主要下这个）
│   └── round-N/blocks.pt
└── uploads/
    ├── round-N/client-0/blocks.pt
    └── round-N/client-1/blocks.pt
```

- **大文件**：几乎都在 MinIO。
- **HTTP**：只传 plan、完成通知、聚合是否 done、耗时指标等小 JSON。

---

## 6. HTTP 控制面（谁问谁答）

| 调用方 | 接口 | 作用 |
|--------|------|------|
| 客户端 | `GET /api/round/current` | 当前轮次 |
| 客户端 | `GET /api/round/{N}/plan` | 本轮选哪些 block |
| 客户端 | `POST /api/round/{N}/client/{C}/upload-complete` | 「我的 blocks 已在 MinIO」+ loss/timings |
| 客户端 | `GET /api/round/{N}/result` | 轮询聚合是否完成 |
| 客户端 | `POST /api/round/{N}/client/{C}/timing` | 聚合后再补报完整分段耗时（含 wait_agg、post_delta） |

Server 收齐两端 `upload-complete` 后才开始拉 MinIO、FedAvg、写回。

---

## 7. 时序示意（一轮）

```
ICC1                         Server                         ICC2
 │                             │                             │
 │── GET plan ────────────────►│◄──────────────── GET plan ──│
 │                             │                             │
 │  cache/load/train/eval      │      cache/load/train/eval  │
 │  encode                     │              encode         │
 │                             │                             │
 │── PUT blocks → MinIO ───────┼─────── PUT blocks → MinIO ──│
 │── upload-complete ─────────►│◄──────── upload-complete ───│
 │                             │                             │
 │  （wait_agg：轮询 result）   │  拉两端 → FedAvg            │
 │                             │  写 global_delta + state    │
 │◄──── result done ───────────┤────────── result done ─────►│
 │                             │                             │
 │── GET global_delta → apply ─┼── GET global_delta → apply ─│
 │                             │                             │
 │─────────── 进入下一轮 ───────┴─────── 进入下一轮 ──────────│
```

注意：

- ICC1 / ICC2 **互不直连**，只跟 Server + MinIO 说话。
- 先传完的一端会在 `wait_agg` 里多等一会儿（等另一端 + 等 Server 写盘）。
- `wait_agg` **不包含**本端自己的 upload（upload 已经在前面做完）。

---

## 8. 和 `time_breakdown` 相关的常见现象

| 现象 | 含义 |
|------|------|
| download 几乎为 0 | 增量 cache 生效，正常 |
| train 占比最大 | 算力主体，正常 |
| wait_agg 约 20s | 多数时间在等 Server 写 `global_delta`+全量 `global_state`，不是本端在算 |
| 虚线 round_wall 高于堆叠条 | 堆叠条只画了 client0 部分阶段；整轮还含对端进度差与 Server 聚合写盘 |
| 偶发 upload 变很长 | 跨机 MinIO 连接假死；现已硬超时+重建连接，失败后应在约 1 分钟内重试成功，而不应再卡十几分钟 |

---

## 9. 本轮联调默认参数（便于对照结果）

> 配置已迁移到 YAML 驱动（CFG-1/CFG-2）。下表是 `configs/s3r12v3-fsdp-run.yaml` 的默认值；改 yaml 即可换参，**不要再只改 `RATIO`**（已加启动校验，`ratio` 与 `coverage_h` 不一致会拒绝启动）。

| 项 | 值 | 配置来源 |
|----|-----|---------|
| 模型 | Qwen2.5-0.5B | `nodes.yaml` 每端 `model_path` |
| 客户端数 / 轮数 | 2 / 20 | yaml `federated.num_clients` / `num_rounds` |
| 上传比例 | ≈20% blocks | 由 `federated.coverage_h=5` 决定（≈1/H）；改 H=10→10%、H=20→5%、H=2→50% |
| 多 slot | `slots_per_round=1` | 设 2 + H=5 → ≈40%（ALG-2） |
| 传输精度 | fp16 | yaml `federated.transfer_dtype`（也支持 int8，SCALE-2） |
| 本地步数 | 30 step/轮 | yaml `train.local_steps`；`nodes.yaml` 可 per-client 覆盖（TRAIN-1） |
| 全量写盘频率 | 每轮 | yaml `io.write_full_global_every_n_rounds`（5/10/0 减负，IO-1） |
| 写盘超参 | timeout 1800s | yaml `io.client_upload_timeout_s` |
| 鉴权 | 关 | yaml `security.auth_token`（非空启用 Bearer，SEC-0） |
| 结果目录示例 | `results/202609101345/`（内含 `run.yaml` + `run_meta.json`） | — |
| 画图 | `python scripts/plot_s3r12v3_fsdp_run.py results/<id>` | — |
| 实时监控 | `bash scripts/check_rerun_status.sh --watch`（OPS-1） | — |

---

## 10. 相关文件索引

| 路径 | 说明 |
|------|------|
| `scripts/start_s3r12v3_fsdp_run.sh` | 一键起 Server + 两端 |
| `scripts/plot_s3r12v3_fsdp_run.py` | 出 train/eval/时间/传输图 |
| `scripts/check_rerun_status.sh` | 看当前轮进度 |
| `experiments/server/README.md` | 聚合服务 API 与增量协议要点 |
| `results/README.md` | 结果目录约定 |
| `deployment/dual-cluster-nodes.md` | 节点与启动顺序（运维） |
