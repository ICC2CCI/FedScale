# S2 联邦全量上传 — 结果记录

- **状态**：done
- **创建**：2026-09-02
- **更新**：2026-09-02
- **关联**：`docs/exec-plans/completed/2026-09-02-three-scenario-loss-comparison.md`

## 配置

- **脚本**：`scripts/run_s2_federated_full.py`（轻量联邦模拟，不依赖 Flower runtime）
- **拓扑**：本机单卡模拟 2 客户端 + 1 server，顺序执行
- **模型**：`/data/models/Qwen/Qwen2.5-0.5B`（base，**bf16**）
- **数据**：medical_meadow_flashcards，30176 train IID 切 2 份（各 15088），3352 eval
- **联邦**：20 轮，每轮每客户端本地训练 **30 步**，FedAvg 加权平均（按样本数）
- **上传策略**：**全量 FP16/BF16 state_dict**（无压缩，每轮上传完整模型 ~988MB）
- **超参**：lr=1e-5, bs=8, grad_accum=2, cosine+warmup 0.1, bf16, gradient_checkpointing, seq_len=512
- **聚合**：复用 `fedavg_state_dicts` 同款逻辑（FP32 累加 → 原 dtype）
- **日志**：`logs/s2-federated-full.log`、`logs/s2-federated-full-outer.log`
- **输出**：`output/s2-federated-full/`（含 `round_log.json`、`final_global.pt`）
- **曲线图**：`output/s2-federated-full/s2-loss-curve.png`（脚本 `scripts/plot_s2_loss.py`）

## 结果

| 指标 | S1-D 基线（单集群） | **S2（联邦全量上传）** |
|---|---|---|
| 训练总步数 | 339 steps（2 epoch） | 20 轮 × 2 客户端 × 30 步 = 1200 local steps |
| eval loss 起点 | 1.1009 | 1.2389 |
| **eval loss 终点** | **0.8635** | **1.0091** |
| 下降幅度 | 0.2374 | 0.2298 |
| 过拟合 | 无 | 无 |
| 最终 gap vs S1-D | — | +0.1456（联邦代价） |

## eval loss 完整轨迹（20 轮）

```
round  1 → 1.2389
round  2 → 1.1912
round  3 → 1.1697
round  4 → 1.1517
round  5 → 1.1399
round  6 → 1.1289
round  7 → 1.1194
round  8 → 1.1084
round  9 → 1.0999
round 10 → 1.0896
round 11 → 1.0822
round 12 → 1.0727
round 13 → 1.0644
round 14 → 1.0552
round 15 → 1.0465
round 16 → 1.0386
round 17 → 1.0300
round 18 → 1.0229
round 19 → 1.0152
round 20 → 1.0091  ← 终点（仍下降）
```

## 关键结论

1. **S2 收敛正常**：eval loss 从 1.24 持续降至 1.01，20 轮无过拟合，曲线形态健康。
2. **联邦代价 = +0.1456**：S2 最终 eval loss 1.01 比 S1-D 基线 0.86 高 0.15。原因：
   - 每客户端只有一半数据（15088 vs 30176），单轮信息量少
   - FedAvg 平均两个客户端的更新，等效学习率被稀释
   - 总 local steps 1200 vs S1-D 的 339（但 S1-D 是全量数据，S2 每客户端半量）
3. **下降幅度相近**（0.23 vs 0.24）：说明联邦机制本身没破坏学习能力，只是效率略低。
4. **曲线仍下降**：20 轮末还在降，未触平台——可继续训练或增加轮数。

## 实现说明（轻量模拟脚本）

不使用 Flower runtime，自写编排复用核心逻辑：
- `get_full_state` / `load_full_state`：保存/加载完整 state_dict（bf16）
- `fedavg`：复用 `aggregation.py:fedavg_state_dicts` 同款 FP32 累加逻辑
- 每轮：2 客户端顺序训练（各 30 步 SFTTrainer）→ 收集 state → FedAvg → eval
- 修复：fp16→bf16（H100 支持，避免 FP16 梯度 unscale 错误）、constant→cosine lr（避免 round 2 崩溃）

## 日志

- 2026-09-02: 完成 S2 训练。20 轮联邦全量上传，eval loss 1.24→1.01，比 S1-D 基线高 0.15（联邦代价）。下一步推进 S3（分片 20% 上传）。
