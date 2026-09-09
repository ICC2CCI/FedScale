# 实验脚本

本目录包含所有联邦微调实验脚本。

## 脚本列表

| 脚本 | 场景 | 说明 |
|---|---|---|
| `run_s2_federated_full.py` | S2 | 联邦全量上传（100% 带宽基线） |
| `run_s3_federated_shard20.py` | S3 | 固定分片 20%（不收敛基线） |
| `run_s3r_federated_shard20_residual.py` | S3R | 残差分片 |
| `run_s3r5_sgd_memory.py` | S3R5 | 论文方法（SGD memory） |
| `run_s3r6_sgd_memory_decay.py` | S3R6 | memory + decay |
| `run_s3r11_layer_random20.py` | S3R11 | 随机层选择 + memory + decay |
| `run_s3r12v2_block_permutation.py` | S3R12v2 | key 级 block 排列（true 20%） |
| `run_s3r12v3_block_uniform.py` | **S3R12v3** | **block 级均匀分片（主力算法）** |
| `run_s3r12v3_ratio.py` | S3R12v3 ratio | ratio 消融（5%/10%/30%/40%/50%） |
| `run_fedrolex_partial_training.py` | FedRolex | 部分训练（轮训层） |

## 画图脚本

| 脚本 | 说明 |
|---|---|
| `plotting/plot_s3r12v3_vs_all.py` | S3R12v3 vs 所有场景 |
| `plotting/plot_s3r12v3_ratio_ablation.py` | ratio 消融对比 |
| `plotting/plot_s3r12v3_equal_convergence.py` | 等收敛分析 |
| `plotting/plot_s3r12v2_vs_ratios.py` | S3R12v2 vs S3R11 ratios |
| `plotting/plot_fedrolex_vs_all.py` | FedRolex vs 所有场景 |

## 运行方法

```bash
# 主力实验
python experiments/run_s3r12v3_block_uniform.py

# ratio 消融
python experiments/run_s3r12v3_ratio.py --ratio 0.10

# 画图
python experiments/plotting/plot_s3r12v3_vs_all.py
```

## 修改模型路径

脚本顶部修改 `MODEL_PATH`：

```python
MODEL_PATH = "/data/models/Qwen/Qwen2.5-0.5B"  # 改为你的路径
```

## 输出位置

默认输出到 `output/` 目录（自动创建）。每个实验一个子目录：

```
output/s3r12v3-block-uniform/
├── round_log.json
├── final_global.pt
└── r{N}-c{cid}/
```

详见 [deployment/run-experiment.md](../deployment/run-experiment.md)。
