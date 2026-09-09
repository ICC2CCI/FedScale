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
- `POST /api/round/{N}/client/{C}/upload-complete`
- `GET  /api/round/{N}/result`
