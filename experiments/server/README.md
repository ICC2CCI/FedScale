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
