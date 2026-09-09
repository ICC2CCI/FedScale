# Central Server（本机）组件占位

本目录对应规划文档：
`docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md`

MinIO（成熟组件）已在 Central Server 用 Docker 部署，见：
`deployment/central-server-prep.md`

## 计划文件

| 文件 | 职责 | 状态 |
|---|---|---|
| `aggregation_server.py` | REST API + FedAvg 聚合 + round 调度 | **待实现** |
| `minio_client.py` | MinIO 读写封装 | **待实现** |
| `block_scheduler.py` | block mask / permutation 调度 | **待实现** |

## 启动（代码就绪后）

```bash
conda activate fedscale-server
source deployment/central-server.env
python experiments/server/aggregation_server.py \
  --port 8080 \
  --minio-endpoint http://localhost:9000 \
  --num-clients 2 \
  --num-rounds 20 \
  --ratio 0.2
```

## REST API（规划）

- `GET  /api/round/current`
- `GET  /api/round/{N}/plan`
- `POST /api/round/{N}/client/{C}/upload-complete`
- `GET  /api/round/{N}/result`
