# AGENTS.md — 项目边界与约定

## 当前并行的两个项目

### 1. SecAgg / S3R12v3（现有主力方案）

| 项 | 值 |
|---|---|
| 代码目录 | `experiments/` |
| 服务端 | `experiments/server/aggregation_server.py`（端口 8080） |
| 客户端 | `experiments/run_s3r12v3_fsdp.py` |
| 共享库 | `experiments/shared/`（secagg_*, block_*, fixed_point 等） |
| 配置 | `configs/s3r12v3-fsdp-*.yaml` |
| 结果 | `results/` |
| 文档 | `docs/algorithm/2026-09-*-secagg-*.md`, `2026-09-*-s3r12v3-*.md` |
| MinIO key | `uploads/`, `global_delta/`, `global_state/`（无 sketchopa 前缀） |
| 启动脚本 | `scripts/start_s3r12v3_fsdp_run.sh` |

> 算法：S3R12v3 block 分片 + Windowed SecAgg（Hadamard + INT16 + pairwise DH mask）。不复用、不混入 SketchOPA 代码。

### 2. SketchOPA（third-party 技术的大模型验证，独立实验）

| 项 | 值 |
|---|---|
| 代码目录 | `experiments_sketchopa/` |
| 协调器 | `experiments_sketchopa/coordinator.py`（端口 8090） |
| 客户端 | `experiments_sketchopa/client.py` |
| 核心库 | `experiments_sketchopa/sketchopa/` |
| 单进程测试 | `experiments_sketchopa/run_experiment.py` |
| 配置 | 命令行参数（无 yaml） |
| 结果 | `results_sketchopa/` |
| 文档 | `docs/algorithm/2026-10-12-sketchopa-*.md` |
| MinIO key | `sketchopa/` 前缀 |
| third-party 材料 | `third-party/`（交底书 + Clover 实验数据） |

> 算法：Top-k → 按层分区 CountSketch → 一轮式安全聚合（SHPRG mask + Shamir seed 分享）→ median-of-rows 恢复。不复用、不混入 SecAgg 代码。
>
> 仅复用：模型权重（`model/Qwen/`）、数据集（`data/splits/`）、FSDP 配置（`accelerate_config.yaml`）、MinIO 基础设施。

## 如何在对话中区分

- 说 **"SecAgg"** 或 **"现有方案"** → 操作 `experiments/` 体系
- 说 **"SketchOPA"** 或 **"third-party 技术"** → 操作 `experiments_sketchopa/` 体系
- 如果不确定，我会先问清楚再动手

## 共用资源（两项目都不碰对方代码，但可共用）

- 模型：`model/Qwen/Qwen2.5-0.5B`
- 数据：`data/splits/icc1_client0_train.json`, `data/splits/client1_train.json`, `data/medical_flashcards_eval.json`
- FSDP：`configs/accelerate_config.yaml`（8×V100）
- MinIO：Central `192.168.235.42:9000`，bucket `fedscale-bucket`（但 key 前缀隔离）
- ICC 节点：ICC1 `192.168.206.116`（`/home/pcl/liuchao/fedscale-icc-1`），ICC2 `192.168.205.130`（`/home/pcl/liuchao/fedscale-icc-2`）

## 环境约定

| 节点 | conda 环境 | 用途 |
|---|---|---|
| Central (本机) | `fedscale-server` | coordinator / 聚合（CPU 即可） |
| ICC1 / ICC2 | `flwr-ft` | client 训练（GPU + FSDP） |

## 运行检查

- 改代码后：本机 `python tests/test_sketchopa_core.py` + `python tests/test_smoke.py`（SketchOPA）；现有系统的测试见 `experiments/tests/`
- 提交前：确认只 stage 当前项目的文件，不混入对方结果
