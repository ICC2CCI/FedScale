# Central Server 本机准备说明

对应方案：[`docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md`](../docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md)

本机角色：**Central Server**（聚合 + MinIO，无需 GPU）

> 本文件记录**本机运维事实**（IP、已部署状态）。可复用的通用步骤见同目录 Compose / 脚本；勿把密钥写进会提交的文档。

## 本机信息

| 项 | 值 |
|---|---|
| IP | `192.168.235.42` |
| CPU / 内存 | 40 核 / 440 GB |
| GPU | 无（符合方案） |
| Aggregation（待代码） | `http://192.168.235.42:8080` |
| MinIO API | `http://192.168.235.42:9000` |
| MinIO Console | `http://192.168.235.42:9001` |
| 数据目录 | `data/minio/`（无 sudo，未用 `/data/minio`） |

## 部署原则

| 组件 | 是否改代码 | 部署方式 |
|---|---|---|
| MinIO | 否 | **Docker Compose**（`restart: "no"`） |
| aggregation_server | 是（自研） | conda / 进程，稳定后再容器化 |
| experiment-dashboard | 可选 | 稳定后可并进 Compose |

## 当前状态（2026-09-09）

| 项 | 状态 |
|---|---|
| Docker Engine | 已安装（24.0.5 + Compose v2.20.2） |
| MinIO 容器 | **已运行** `fedscale-minio` |
| 镜像 | `quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z`（Docker Hub 拉取失败时用 quay） |
| 重启策略 | `no`（不自动重启） |
| Bucket | `fedscale-bucket` |
| 旧二进制 / systemd MinIO | 已停用 |
| `aggregation_server.py` | **未实现**（`:8080` 未监听） |

## MinIO 常用命令

```bash
# 启动（会停掉旧的二进制 MinIO）
bash scripts/docker-central-up.sh

# 停止
bash scripts/docker-central-down.sh

# 状态
sg docker -c 'docker ps --filter name=fedscale-minio'
curl -fsS http://127.0.0.1:9000/minio/health/live && echo OK
```

配置模板：`deployment/central-server.env.example`  
本机密钥文件（gitignore）：`deployment/central-server.env`

Compose：`deployment/docker-compose.yml`

## ICC1 / ICC2 客户端连接参数

```bash
--server-url http://192.168.235.42:8080
--minio-endpoint http://192.168.235.42:9000
```

MinIO 账号密码见本机 `central-server.env`（不要提交到 git）。

## 待完成

- [ ] 实现 `experiments/server/aggregation_server.py` 等通信层
- [ ] 启动聚合服务并做健康检查
- [ ] （可选）监控看板

## 与 ICC 仓库协作（避免冲突）

见 [`central-server-git-notes.md`](central-server-git-notes.md)。
