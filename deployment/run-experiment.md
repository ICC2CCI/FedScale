# 运行实验

## 1. 前置条件

确保已完成：
- [x] 环境搭建（`environment-setup.md`）
- [x] 模型下载（`model-download.md`）
- [x] 数据准备（`data-preparation.md`，仓库已含数据）

## 2. 修改模型路径

实验脚本中默认模型路径为 `/data/models/Qwen/Qwen2.5-0.5B`。如果你的模型路径不同，修改脚本顶部的 `MODEL_PATH`：

```python
MODEL_PATH = "/data/models/Qwen/Qwen2.5-0.5B"  # 改为你的路径
```

## 3. 运行 S3R12v3（主力算法，20% 带宽）

```bash
python experiments/run_s3r12v3_block_uniform.py
```

输出：

```
output/s3r12v3-block-uniform/
├── round_log.json       # 逐轮日志
├── final_global.pt      # 最终权重（1.2 GB）
├── r1-c0/               # 第1轮 client 0 的训练输出
├── r1-c1/
├── r1-eval/             # 第1轮评估
├── ...
└── r20-eval/
```

预期结果：20 轮后 eval loss ≈ 1.0514

## 4. 运行 S3R12v3 ratio 消融

```bash
# 5% 带宽
python experiments/run_s3r12v3_ratio.py --ratio 0.05

# 10% 带宽
python experiments/run_s3r12v3_ratio.py --ratio 0.10

# 30% 带宽
python experiments/run_s3r12v3_ratio.py --ratio 0.30

# 40% 带宽
python experiments/run_s3r12v3_ratio.py --ratio 0.40

# 50% 带宽
python experiments/run_s3r12v3_ratio.py --ratio 0.50
```

## 5. 运行其他场景

```bash
# S2 全量上传基线
python experiments/run_s2_federated_full.py

# S3 固定分片
python experiments/run_s3_federated_shard20.py

# S3R11 随机层 + memory + decay
python experiments/run_s3r11_layer_random20.py

# FedRolex 部分训练
python experiments/run_fedrolex_partial_training.py
```

## 6. 画图

```bash
# S3R12v3 vs 所有场景
python experiments/plotting/plot_s3r12v3_vs_all.py

# ratio 消融
python experiments/plotting/plot_s3r12v3_ratio_ablation.py

# 等收敛分析
python experiments/plotting/plot_s3r12v3_equal_convergence.py
```

图片输出到 `output/` 目录。

## 7. 运行 S1-D 全量训练基线（LLaMA-Factory）

需要安装 LLaMA-Factory：

```bash
pip install llamafactory
```

```bash
llamafactory-cli train configs/qwen25-0.5b-medical-flashcards-full.yaml
```

输出到 `output/s1d-qwen25-0.5b-medical-flashcards/`。

## 8. 实验参数说明

详见 [`docs/reference/federated-training-reproduction-params.md`](../docs/reference/federated-training-reproduction-params.md)。

关键参数：

| 参数 | 值 |
|---|---|
| 联邦轮数 | 20 |
| 每轮本地步数 | 30 |
| 客户端数 | 2 |
| batch size | 8 × 2 (grad_accum) = 16 |
| learning rate | 1e-5 |
| lr scheduler | cosine |
| warmup ratio | 0.1 |
| seq_len | 512 |
| seed | 20260831 |

## 9. 预期结果对照

| 实验 | eval loss | 带宽 | 运行时间（H100） |
|---|---|---|---|
| S1-D 全量 | ~0.86 | 100% | ~30 min |
| S2 联邦全量 | 1.0069 | 100% | ~45 min |
| S3R12v3 20% | 1.0514 | 20% | ~45 min |
| S3R12v3 10% | 1.0979 | 10% | ~45 min |
| FedRolex | 1.1829 | 100% | ~45 min |

## 10. 常见问题

### GPU 显存不足

```python
# 减小 batch size
per_device_train_batch_size=4,
gradient_accumulation_steps=4,  # 保持 effective batch = 16
```

### 模型路径错误

```python
# 确认模型存在
ls /data/models/Qwen/Qwen2.5-0.5B/model.safetensors
```

### gradient_checkpointing 报错

FedRolex 实验中如遇到 gradient_checkpointing 报错，设置 `gradient_checkpointing=False`。
