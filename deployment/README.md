# 部署指南

本目录提供在新服务器上部署 **当前双集群**（ICC1/ICC2 + Central MinIO）实验环境的步骤。

## 文档

| 文档 | 内容 |
|---|---|
| [environment-setup.md](environment-setup.md) | Conda 环境、Python 依赖、CUDA 验证 |
| [model-download.md](model-download.md) | 基础模型下载（ModelScope） |
| [data-preparation.md](data-preparation.md) | 测试数据准备与切分 |
| [central-server-prep.md](central-server-prep.md) | Central：MinIO Docker 等 |
| [dual-cluster-nodes.md](dual-cluster-nodes.md) | ICC1/ICC2/Server 节点与启动顺序 |
| [central-server-git-notes.md](central-server-git-notes.md) | 与 ICC 上游协作、降低 git 冲突 |
| [troubleshooting.md](troubleshooting.md) | 常见问题 |
| [当前联调流程](../docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md) | **运行时以本文为准** |

> `run-experiment.md` 若仍写单机 `run_s3r12v3_block_uniform.py`，仅作历史参考；双集群请用下面「快速开始」。

## 快速开始（双集群）

```bash
# 0. 三端已克隆仓库；模型与数据按 ICC1_PATHS.md / ICC2_PREP.md 就位
# 1. Central：拉起 MinIO
bash scripts/docker-central-up.sh

# 2. Central：配置 deployment/nodes.yaml + central-server.env 后一键启动
#    非 SecAgg：
RUN_CONFIG=$PWD/configs/s3r12v3-fsdp-run.yaml \
  bash scripts/start_s3r12v3_fsdp_run.sh

#    SecAgg（常用 8081，避免与默认 8080 冲突）：
RUN_CONFIG=$PWD/configs/s3r12v3-fsdp-secagg-verify.yaml \
AGGREGATION_PORT_OVERRIDE=8081 \
NUM_ROUNDS_ENV=20 \
  bash scripts/start_s3r12v3_fsdp_run.sh

# 3. 看进度 / 画图
bash scripts/check_rerun_status.sh
python scripts/plot_s3r12v3_fsdp_run.py results/<RUN_ID>
```

脚本会：起 Aggregation Server → SSH 拉起 ICC1/ICC2 的 `accelerate launch … run_s3r12v3_fsdp.py` → 结果写入 `results/YYYYMMDDHHMM/`。

## 硬件要求（当前实测拓扑）

| 角色 | 配置 |
|---|---|
| Central | CPU；内存建议 ≥32GB（聚合+MinIO）；无 GPU |
| ICC1 / ICC2 | 多卡 GPU（联调为 8×V100 32GB）；conda `flwr-ft` |
| 模型 | Qwen2.5-0.5B ≈1GB 级本地缓存 |
| 磁盘 | 代码+数据+模型+多次 `results/`，建议 ≥50GB |

## 多集群部署

### 双集群 FSDP + MinIO（当前方案）

- 流程：[current-flow](../docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
- 早期规划（部分过时）：[2026-09-09 规划](../docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md)
- Compose / MinIO：`bash scripts/docker-central-up.sh`

### 旧版 K8s + Flower（已废弃作入口）

`deploy_multinode.sh`、`flowertune-llm/`、旧 K8s yaml 仅历史保留，**不要用于当前 SecAgg / S3R12v3 FSDP 实验**。

## 常见问题

详见 [troubleshooting.md](troubleshooting.md)。
