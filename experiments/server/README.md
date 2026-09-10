# Central Server（本机）组件

对应规划：`docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md`  
运维：`deployment/central-server-prep.md`

## 文件

| 文件 | 职责 |
|---|---|
| `aggregation_server.py` | REST API + FedAvg 聚合 + round 调度 |
| `block_scheduler.py` | S3R12v3 block mask / RoundPlan |
| `minio_client.py` | 转发到 `shared.minio_client` |

## 启动前：上传 round-0 初始权重

在 **ICC1 或 ICC2**（有模型）执行：

```bash
source ~/miniconda3/bin/activate flwr-ft
cd ~/liuchao/fedscale-icc-1   # 或 fedscale-icc-2
bash scripts/bootstrap_initial_state.sh model/Qwen/Qwen2.5-0.5B http://192.168.235.42:9000
```

## 启动聚合服务（Central Server）

```bash
conda activate fedscale-server
cd /home/pcllgr/liuchao/fedscale-icc-server
python experiments/server/aggregation_server.py \
  --port 8080 \
  --minio-endpoint http://127.0.0.1:9000 \
  --num-clients 2 \
  --num-rounds 20 \
  --ratio 0.2
```

（若 MinIO 已有 `global_state/round-0/state.pt`，无需再传 `--model-path`。）

## REST API

- `GET  /health`
- `GET  /api/round/current`
- `GET  /api/round/{N}/plan`
- `POST /api/round/{N}/client/{C}/upload-complete`（可附带客户端分段 `timings`）
- `POST /api/round/{N}/client/{C}/timing`（聚合后补报完整客户端耗时）
- `GET  /api/round/{N}/result`（含 `timing_s`）

`results/.../round_log.json` 每轮会写入 `timing_s`：轮次墙钟、等客户端、上传先后差、汇聚下载/FedAvg/上传全局、以及各 client 分段耗时。

## 韧性（根治相关）

- MinIO 客户端：长 read timeout、multipart、失败指数退避重试
- FSDP 客户端：下载/上传/等聚合用 **短心跳 broadcast**，避免 rank0 做网络 I/O 时其它 rank 卡死在长 NCCL barrier
- 聚合服务：`FEDSCALE_CLIENT_UPLOAD_TIMEOUT_S`（默认 1800）内未收齐上传则 `aggregation_failed`，客户端可快速退出

## 增量全局下发（global_delta）

- 上传：仍只传本轮选中的 ~ratio blocks
- 聚合后 Server 额外写入 `global_delta/round-N/blocks.pt`（FedAvg 后的 block 增量），并继续保留全量 `global_state/round-N/state.pt` 作兜底/评估
- 客户端本地缓存上一轮全局；优先下载并 `apply` delta；冷启动或断档则回退拉全量
- 每轮聚合完成后客户端立刻拉取并应用本轮 `global_delta`，使下一轮多为 `cache`（0 下载）；指标 `download_mode`: 0=cache / 1=delta / 2=full，以及 `post_delta_MiB`
- `--transfer-dtype`：通信落盘精度，`auto`（默认，跟随模型）/ `fp16` / `fp32` / `bf16`；`int8` 预留。聚合内部仍可用 fp32 累加，写出再 cast
- 首轮 `--skip-round0-download`：客户端用本地 `model-path` 作 round-0，不再拉全量
- 在线 `--online-eval`：每轮本地训练后算 eval_loss 并随 upload-complete 上报，写入 `round_log.json`
- MinIO 上传：单次 attempt 硬超时 + 失败重建连接重试，避免假死拖十几分钟
