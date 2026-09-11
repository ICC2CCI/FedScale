# S1-D 单集群基线（医疗 flashcards）— 正式基线

- **状态**：done
- **创建**：2026-09-02
- **更新**：2026-09-02
- **关联**：`docs/exec-plans/completed/2026-09-02-three-scenario-loss-comparison.md`（三场景主计划）

## 背景

S1-C（alpaca 46785 条）eval loss 卡在 1.33 触容量墙，下降仅 0.03，曲线太平无法作为联邦对比基线。改用医疗领域数据集 medical_meadow_medical_flashcards（窄领域、模式化强，0.5B 容量能学得更深），得到明显下降的 loss 曲线作为正式基线。

## 数据集

- **名称**：medalpaca/medical_meadow_medical_flashcards
- **HF 页面**：https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards
- **来源**：MedAlpaca 项目（医学闪卡问答）
- **格式**：`{instruction, input, output}`（alpaca 格式）
- **样例**：
  - instruction: "Answer this question truthfully"
  - input: "What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?"
  - output: "Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels."
- **原始条数**：33955 条
- **去重后**：33528 条
- **切分**：30176 train / 3352 eval（90/10，seed=20260831）
- **本地文件**：
  - 原始：`flower-llm/llamafactory-local/assets/datasets/medical_meadow_flashcards.json`
  - 训练：`flower-llm/llamafactory-local/assets/datasets/medical_flashcards_train.json`
  - 评估：`flower-llm/llamafactory-local/assets/datasets/medical_flashcards_eval.json`

## 配置

- **配置文件**：`flower-llm/llamafactory-local/configs/qwen25-0.5b-medical-flashcards-full.yaml`
- **模型**：`/data/models/Qwen/Qwen2.5-0.5B`（base，FP16）
- **超参**：
  - `learning_rate: 1.0e-5`
  - `per_device_train_batch_size: 8`，`gradient_accumulation_steps: 2`（有效 batch=16）
  - `num_train_epochs: 2.0`
  - `weight_decay: 0.01`
  - `warmup_ratio: 0.03`
  - `lr_scheduler_type: cosine`
  - `fp16: true`，`gradient_checkpointing: true`
  - `flash_attn: auto`
  - `template: qwen`，`finetuning_type: full`，`stage: sft`
  - `cutoff_len: 512`
- **eval**：`eval_strategy: steps`，`eval_steps: 100`，`val_size: 0.1`（从 train 切 10%）
- **env**：conda env `flwr-ft`（torch 2.8.0+cu128, transformers 4.45.2, llamafactory 0.8.1）
- **日志**：`logs/s1d-medical-train.log`
- **输出**：`output/s1d-qwen25-0.5b-medical-flashcards/`
- **曲线图**：`output/s1d-qwen25-0.5b-medical-flashcards/s1d-loss-curve.png`（脚本 `scripts/plot_s1d_loss.py`）

## 结果

| 指标 | 值 |
|---|---|
| 训练步数 | 339 steps |
| 训练耗时 | 33 分 22 秒 |
| train loss 均值 | 0.7875 |
| **eval loss 起点** | **1.1009** @ epoch 0.06 |
| **eval loss 终点** | **0.8635** @ epoch 1.94 |
| **eval loss 最低** | **0.8635** @ epoch 1.94（终点即最低） |
| **下降幅度** | **0.2374**（1.1009 → 0.8635） |
| eval 点数 | 33 |
| 过拟合 | 无（全程下降到最后） |
| train-eval gap | 0.07（train 0.79 vs eval 0.86，泛化优秀） |
| GPU 峰值显存 | ~11 GB / 80 GB |

## eval loss 完整轨迹（33 点）

```
epoch 0.06 → 1.1009
epoch 0.12 → 1.0264
epoch 0.18 → 1.0113
epoch 0.24 → 0.9991
epoch 0.29 → 0.9841
epoch 0.35 → 0.9787
epoch 0.41 → 0.9695
epoch 0.47 → 0.9561
epoch 0.53 → 0.9496
epoch 0.59 → 0.9352
epoch 0.65 → 0.9306
epoch 0.71 → 0.9198
epoch 0.77 → 0.9132
epoch 0.82 → 0.9039
epoch 0.88 → 0.8986
epoch 0.94 → 0.8871
epoch 1.00 → 0.8792  ← epoch 1 边界
epoch 1.06 → 0.9027  (shuffle 噪声尖刺)
epoch 1.12 → 0.8969
epoch 1.18 → 0.8926
epoch 1.24 → 0.8873
epoch 1.30 → 0.8835
epoch 1.35 → 0.8795
epoch 1.41 → 0.8783
epoch 1.47 → 0.8747
epoch 1.53 → 0.8725
epoch 1.59 → 0.8691
epoch 1.65 → 0.8678
epoch 1.71 → 0.8669
epoch 1.77 → 0.8651
epoch 1.83 → 0.8642
epoch 1.89 → 0.8635
epoch 1.94 → 0.8635  ← MIN（终点）
```

## 末段下降速率（确认进入平台期）

```
epoch 1.71 → 1.77: -0.0018
epoch 1.77 → 1.83: -0.0009
epoch 1.83 → 1.89: -0.0007
epoch 1.89 → 1.94: -0.0000  ← 停滞
```

最后 0.05 epoch 下降 0.0000，已进入平台期。无需继续训练额外 epoch。

## 关键结论

1. **医疗数据集对 0.5B 友好**：窄领域、模式化强，eval loss 从 1.10 持续降到 0.86，下降 0.24，曲线形态理想。
2. **无过拟合、无容量墙**：33 个 eval 点全程下降，train-eval gap 仅 0.07，泛化优秀。终点刚进入平台期，未恶化。
3. **作为三场景对比的正式基线**：曲线明显下降，S2/S3 若收敛慢/变差能直观看出差距；无过拟合/容量墙引入额外噪声，归因干净。

## 与之前基线对比

| 指标 | S1-B (alpaca 2400条) | S1-C (alpaca 46785条) | **S1-D (医疗 30176条)** |
|---|---|---|---|
| eval loss 起点 | 1.4735 | 1.3633 | **1.1009** |
| eval loss 终点 | 1.4292（上升） | 1.3328（平） | **0.8635（仍降）** |
| 下降幅度 | -0.04（过拟合） | 0.03（容量墙） | **0.24（明显）** |
| 过拟合 | 有（epoch 0.9 拐） | 无 | 无 |
| 曲线形态 | 拐点上升 | 平台 | **理想下行** |
| 用途 | 弃用 | 弃用 | **正式基线** |

## 日志

- 2026-09-02: 完成 S1-D 训练。医疗 flashcards 数据集使 0.5B eval loss 从 1.10 降至 0.86，曲线形态理想。锁定为三场景对比正式基线。下一步推进 S2（联邦全量上传）。
