# FedScale — 大模型联邦微调实验工作区

本仓库包含基于 Flower 框架的大模型联邦微调完整实验代码、数据、文档与结果，核心是 **S3R12v3 block 级分片上传算法**（在保证收敛的前提下将带宽降至全量上传的 20% 甚至更低）。

## 快速导航

| 你想… | 去这里 |
|---|---|
| 了解 S3R12v3 算法原理 | [`docs/algorithm/2026-09-04-s3r12v3-block-uniform.md`](docs/algorithm/2026-09-04-s3r12v3-block-uniform.md) |
| 看完整复现参数 | [`docs/reference/federated-training-reproduction-params.md`](docs/reference/federated-training-reproduction-params.md) |
| 在新服务器上部署 | [`deployment/README.md`](deployment/README.md) |
| 运行某个实验 | [`experiments/`](experiments/) 目录下的 `run_*.py` |
| 查看实验结果 | [`results/`](results/) 目录下的 round_logs 与 figures |
| 了解核心代码 | [`flowertune-llm/`](flowertune-llm/) 目录 |

## 仓库结构

```
FedScale/
├── flowertune-llm/            # 核心代码：ServerApp / ClientApp / 聚合 / 状态管理
├── experiments/               # 实验脚本（S2/S3/S3R*/FedRolex）
│   ├── run_s3r12v3_block_uniform.py   # 主力算法
│   ├── run_s3r12v3_ratio.py           # ratio 消融
│   └── plotting/                      # 画图脚本
├── data/                      # 测试数据（medical_flashcards）
├── configs/                   # 训练配置（LLaMA-Factory YAML）
├── docs/
│   ├── algorithm/             # 算法文档（S3R12v3 / S3R12v2 / FedRolex / 安全方案）
│   ├── experiment-records/    # 各实验的执行记录与结果
│   └── reference/             # 复现参数等参考文档
├── results/
│   ├── round_logs/            # 每个实验的逐轮日志（JSON）
│   └── figures/               # 对比图（PNG）
├── deployment/                # 多集群部署指南
├── scripts/                   # 原始部署/运维脚本
├── configs/                   # K8s 部署、对象存储配置
├── DESIGN.md                  # 整体设计说明
└── README.md                  # 本文件
```

## 实验场景概览

| 场景 | 说明 | 带宽 | eval loss |
|---|---|---|---|
| S1-D | 全量训练（非联邦，LLaMA-Factory） | 100% | ~0.86 |
| S2 | 联邦全量上传 | 100% | 1.0069 |
| S3 | 联邦固定分片 20% | 20% | 1.12（不收敛） |
| S3R5/6 | 论文方法（SGD memory + decay） | 20% | ~1.09 |
| S3R11 | 随机层选择 + memory + decay | 20% | 1.0462 |
| **S3R12v3** | **block 级均匀分片 + memory** | **20%** | **1.0514** |
| S3R12v3 10% | 低带宽消融 | 10% | 1.0979 |
| FedRolex | 部分训练（轮训层） | 100% | 1.1829 |

## 核心结果

- **S3R12v3 在 20% 带宽下 eval loss = 1.0514**，接近 S2 全量上传的 1.0069
- 在目标 loss 1.10 处，10% ratio 仅需 2.5 GB 带宽 vs S2 的 11.3 GB，**节省 78%**
- block 级分片使上传比例方差从 30%（key 级）降至 3%，保证每轮带宽稳定

## 环境要求

- Python 3.10
- PyTorch 2.8.0 + CUDA 12.8
- Transformers 4.45.2, TRL 0.8.6, Flower 1.18.0
- GPU: 1× H100 80GB（或同等显存）
- 基础模型: Qwen2.5-0.5B（`/data/models/Qwen/Qwen2.5-0.5B`）

详见 [`deployment/`](deployment/) 下的部署指南。

## License

见 [LICENSE](LICENSE)。
