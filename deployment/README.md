# 部署指南

本目录提供在新服务器上部署 FedScale 联邦微调实验环境的完整步骤。

## 文档

| 文档 | 内容 |
|---|---|
| [environment-setup.md](environment-setup.md) | Conda 环境、Python 依赖、CUDA 验证 |
| [model-download.md](model-download.md) | 基础模型下载（ModelScope） |
| [data-preparation.md](data-preparation.md) | 测试数据准备 |
| [run-experiment.md](run-experiment.md) | 运行 S3R12v3 及其他实验的步骤 |

## 快速开始（5 步）

```bash
# 1. 克隆仓库
git clone https://github.com/ICC2CCI/FedScale.git
cd FedScale

# 2. 创建环境（详见 environment-setup.md）
conda create -n flwr-ft python=3.10 -y
conda activate flwr-ft
pip install torch transformers trl flwr datasets accelerate

# 3. 下载模型（详见 model-download.md）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir='/data/models')"

# 4. 准备数据（已在 data/ 目录中，无需额外下载）

# 5. 运行 S3R12v3 实验
python experiments/run_s3r12v3_block_uniform.py
```

## 硬件要求

| 配置 | 最低 | 推荐 |
|---|---|---|
| GPU | 1× 24GB（如 RTX 4090） | 1× 80GB（如 H100） |
| 内存 | 32 GB | 64 GB |
| 磁盘 | 10 GB（代码+数据+模型） | 50 GB（含实验输出） |
| CUDA | 12.1+ | 12.8 |

## 多集群部署

如需多集群联邦部署（K8s + Flower SuperNode），请参考：

- [`../deploy_multinode.sh`](../deploy_multinode.sh) — 多节点部署脚本
- [`../configs/`](../configs/) — K8s 部署配置
- [`../DESIGN.md`](../DESIGN.md) — 整体架构设计

## 常见问题

详见 [troubleshooting.md](troubleshooting.md)。
