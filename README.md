# FedScale — 双集群大模型联邦微调（S3R12v3 + SecAgg）

本仓库当前主力路径是 **ICC1 / ICC2 + Central Server（Aggregation + MinIO）**：  
用 **S3R12v3 block 分片** 降低上传带宽，可选 **Windowed SecAgg（Hadamard + INT16）** 做安全聚合。  
旧 Flower / K8s 代码仍在 `flowertune-llm/`，**不要当作当前入口**。

## 快速导航

| 你想… | 去这里 |
|---|---|
| **看当前整轮怎么跑（首选）** | [`docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md`](docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md) |
| SecAgg 完整流程 / 隐私 | [`docs/algorithm/2026-09-15-secagg-quantization-precision-issue.md`](docs/algorithm/2026-09-15-secagg-quantization-precision-issue.md) §3 |
| SecAgg 优化与 20 轮结论 | [`docs/algorithm/2026-09-17-secagg-quantization-optimization-survey.md`](docs/algorithm/2026-09-17-secagg-quantization-optimization-survey.md) |
| S3R12v3 算法原理 | [`docs/algorithm/2026-09-04-s3r12v3-block-uniform.md`](docs/algorithm/2026-09-04-s3r12v3-block-uniform.md) |
| 复现参数 | [`docs/reference/federated-training-reproduction-params.md`](docs/reference/federated-training-reproduction-params.md) |
| 部署 / 节点 | [`deployment/README.md`](deployment/README.md)、[`deployment/dual-cluster-nodes.md`](deployment/dual-cluster-nodes.md) |
| 一键启动 | `bash scripts/start_s3r12v3_fsdp_run.sh` |
| 实验结果 | [`results/`](results/)（如 `results/202609180941/`） |
| 文档索引 | [`docs/README.md`](docs/README.md) |

## 当前系统一览

| 项 | 当前默认 |
|---|---|
| 角色 | Central（聚合+MinIO）+ ICC1（client0）+ ICC2（client1） |
| 模型 | Qwen2.5-0.5B（各端 `model/Qwen/Qwen2.5-0.5B`） |
| 数据 | medical_flashcards；切分见 `data/splits/` |
| 训练 | Accelerate FSDP，`local_steps=30`，在线 eval |
| 上传 | 约 10% blocks（`coverage_h=10`）或按 yaml；SecAgg 时为 masked INT16 raw |
| 配置 | `configs/s3r12v3-fsdp-run.yaml` / `configs/s3r12v3-fsdp-secagg-verify.yaml` |
| 客户端脚本 | `experiments/run_s3r12v3_fsdp.py` |
| 服务端 | `experiments/server/aggregation_server.py` |
| 端口 | 聚合默认见 `deployment/central-server.env`；SecAgg 验证常用 **8081** |
| 结果 | `results/YYYYMMDDHHMM/`（`round_log.json` + `figures/`） |

## 仓库结构（当前相关）

```
fedscale-icc-server/
├── experiments/
│   ├── run_s3r12v3_fsdp.py          # 双集群 FSDP 客户端（主力）
│   ├── server/aggregation_server.py # 聚合服务
│   └── shared/                      # MinIO / SecAgg / block 选择等
├── configs/
│   ├── s3r12v3-fsdp-run.yaml        # 非 SecAgg 默认
│   ├── s3r12v3-fsdp-secagg-verify.yaml
│   └── accelerate_config.yaml
├── scripts/start_s3r12v3_fsdp_run.sh
├── data/                            # medical_flashcards + splits
├── model/                           # 各端本地模型目录约定
├── docs/algorithm/                  # 联调流程 + SecAgg + S3R12v3
├── results/                         # 每次 run 一个子目录
├── deployment/                      # 环境、节点、MinIO
└── flowertune-llm/                  # 【历史】Flower 路径，勿作当前入口
```

## 核心结果（双集群实测）

| 路径 | 参考目录 | R20 eval | 备注 |
|---|---|---|---|
| 非 SecAgg fp16 | `20260914-final-clean` | ≈0.987 | 明文 block 上传 |
| 旧 SecAgg（per-window amax） | `202609162025` | 1.328 | 已收敛但仍落后 |
| **SecAgg Hadamard（精度对照）** | **`202609180941`** | **0.985** | 对齐 fp16；`train≈38s` |
| SecAgg 时间优化 5 轮 | **`202609181151`** | R5=1.364 | 与 941 逐轮一致；整轮≈110–134s |

带宽：S3R12v3 只传选中 blocks（验证配置约 10%）；相对全量上传可大幅节省。算法消融历史见 `docs/experiment-records/`。

## 环境要求（当前双集群）

| 角色 | 要求 |
|---|---|
| Central | CPU 即可；Docker MinIO；conda `fedscale-server`（跑聚合） |
| ICC1 / ICC2 | 多卡 GPU（实测 8×V100）；conda `flwr-ft`；Accelerate FSDP |
| Python | 3.10 |
| 关键依赖 | PyTorch（CUDA）、Transformers、Accelerate、MinIO 客户端 |
| 模型路径 | 各端 `model/Qwen/Qwen2.5-0.5B`（见 `ICC1_PATHS.md` / `ICC2_PREP.md`） |
| 数据路径 | `data/medical_flashcards_*.json`，`data/splits/icc1_client0_train.json` 等 |

启动 SecAgg 20 轮示例：

```bash
RUN_CONFIG=$PWD/configs/s3r12v3-fsdp-secagg-verify.yaml \
AGGREGATION_PORT_OVERRIDE=8081 \
NUM_ROUNDS_ENV=20 \
bash scripts/start_s3r12v3_fsdp_run.sh
```

正式对照请保持 `--detailed-train-metrics` **关闭**（默认关），否则 `train_local_s` 会膨胀到 ~300s。

## License

见 [LICENSE](LICENSE)。
