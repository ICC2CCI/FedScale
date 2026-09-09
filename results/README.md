# 实验结果

## 目录结构

| 子目录 | 内容 |
|---|---|
| [`round_logs/`](round_logs/) | 每个实验的逐轮日志（JSON 格式） |
| [`figures/`](figures/) | 对比图（PNG） |

## round_logs

每个 JSON 文件记录了一个实验的逐轮数据：

```json
[
  {
    "round": 1,
    "eval_loss": 1.3665,
    "upload_ratio": 0.205,
    "upload_bytes": 1320000000,
    "round_time_sec": 120
  },
  ...
]
```

| 文件 | 实验 | 最终 eval loss | 带宽 |
|---|---|---|---|
| `s2-federated-full.json` | S2 全量 | 1.0069 | 100% |
| `s3-federated-shard20.json` | S3 固定分片 | — | 20% |
| `s3r11-layer-random20.json` | S3R11 随机层 | 1.0462 | 20% |
| `s3r12v2-block-permutation.json` | S3R12v2 key级 | 1.0569 | 20% |
| `s3r12v3-block-uniform.json` | **S3R12v3 block级** | **1.0514** | **20%** |
| `s3r12v3-ratio-5pct.json` | 5% 带宽 | 1.1298 | 5% |
| `s3r12v3-ratio-10pct.json` | 10% 带宽 | 1.0979 | 10% |
| `s3r12v3-ratio-30pct.json` | 30% 带宽 | 1.0326 | 30% |
| `s3r12v3-ratio-40pct.json` | 40% 带宽 | 1.0316 | 40% |
| `s3r12v3-ratio-50pct.json` | 50% 带宽 | 1.0196 | 50% |
| `fedrolex-partial-training.json` | FedRolex | 1.1829 | 100% |

## figures

| 图片 | 说明 |
|---|---|
| `s3r12v3-vs-all.png` | S3R12v3 与所有场景的 loss 对比 |
| `s3r12v3-ratio-ablation.png` | S3R12v3 不同带宽比例的消融对比 |
| `s3r12v3-equal-convergence.png` | 等收敛分析（带宽 vs 目标 loss） |
| `s3r12v2-vs-all.png` | S3R12v2 与所有场景对比 |
| `fedrolex-vs-all.png` | FedRolex 与所有场景对比 |

## 注意

- 模型权重（`*.pt`）因体积过大未上传，如需请重新运行实验生成
- round_logs 可独立用于画图分析，不依赖模型权重
