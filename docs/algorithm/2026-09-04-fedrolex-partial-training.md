# FedRolex Partial Training — 结果记录

- **状态**：done
- **创建**：2026-09-04
- **更新**：2026-09-04
- **关联**：`docs/exe-plans/active/2026-09-02-three-scenario-loss-comparison.md`

## 背景

FedRolex（NeurIPS 2022）提出 rolling sub-model extraction：每轮只训练模型的一部分层（partial training），其余层冻结。与 S3R11 的"全训只传部分层"不同，FedRolex 是"只训部分层、只传部分层"，同时省带宽和算力。

## 配置

- **脚本**：`scripts/run_fedrolex_partial_training.py`
- **拓扑**：本机单卡模拟 2 客户端 + 1 server，顺序执行
- **模型**：`/data/models/Qwen/Qwen2.5-0.5B`（base，bf16，24 层，~494M 参数）
- **数据**：medical_meadow_flashcards，30176 train IID 切 2 份，3352 eval
- **联邦**：20 轮，每轮每客户端本地训练 30 步
- **partial training 策略**：
  - 每轮随机选 `mask_ratio_layers = 0.2`（5 层 / 24 层）训练，其余层 `requires_grad=False`
  - 选层用 `random.sample`，每轮独立采样
  - 只上传被训练层的 delta（INT8 量化）
  - 不使用 memory 机制（未训层无更新，无需累积）
- **可训练参数**：~59.6M / 494M（约 12%，即 5 层）
- **关键修复**：`gradient_checkpointing=True` 与冻结层冲突（`None of the inputs have requires_grad=True`），改为 `False`。partial training 只训 5 层，显存充裕。
- **超参**：lr=1e-5, bs=8, grad_accum=2, cosine+warmup 0.1, bf16, seq_len=512
- **日志**：`logs/fedrolex-partial-training.log`
- **输出**：`output/fedrolex-partial-training/`（含 `round_log.json`、`final_global.pt`）
- **曲线图**：`output/fedrolex-vs-all.png`（脚本 `scripts/plot_fedrolex_vs_all.py`）

## 结果

| 指标 | S2 全量上传 | S3 分片20%(无mem) | S3R11 随机20%层+mem | **FedRolex partial-train** |
|---|---|---|---|---|
| eval loss 起点 | 1.2389 | 1.5164 | 1.4896 | **1.3665** |
| **eval loss 终点** | **1.0091** | **1.1501** | **1.1190** | **1.1829** |
| 下降幅度 | 0.23 | 0.37 | 0.37 | 0.18 |
| gap vs S2 | — | +0.14 | +0.11 | **+0.17** |
| 上传/轮 | 988 MB | 99 MB | 99 MB | 99 MB |
| 训练参数/轮 | 494M (100%) | 494M (100%) | 494M (100%) | **59.6M (12%)** |
| 需要 memory | 否 | 否 | 是 | **否** |

### 全轮次 eval loss

```
r 1: 1.3665   r 6: 1.2383   r11: 1.2084   r16: 1.1918
r 2: 1.3005   r 7: 1.2294   r12: 1.2036   r17: 1.1890
r 3: 1.2772   r 8: 1.2229   r13: 1.2002   r18: 1.1873
r 4: 1.2609   r 9: 1.2163   r14: 1.1968   r19: 1.1846
r 5: 1.2487   r10: 1.2120   r15: 1.1939   r20: 1.1829
```

## 分析

### 两种思路的本质区别

```
S3R11（全训只传部分）: 训 24 层 → 传 5 层 → 19 层白训但 memory 保留 → 信息不浪费
FedRolex（只训部分）  : 训 5 层  → 传 5 层 → 19 层没训 → 省算力但信息少
```

- **S3R11**：浪费算力但保留信息（memory）→ 收敛好（1.119）
- **FedRolex**：省算力但信息少（无 memory）→ 收敛差一点（1.183）但更简单

### FedRolex 的独特优势

1. **不需要 memory**——没训的层没有更新，不存在丢弃问题，机制最简单
2. **省算力 88%**——只训 5 层（60M/494M 参数），训练速度 ~8 步/秒 vs 全训的 ~3 步/秒
3. **全程稳定**——无波动无发散，曲线平滑单调下降

### 适用场景

- **带宽是唯一瓶颈、算力充足** → S3R11（收敛好）
- **带宽+算力都瓶颈（边缘设备）** → FedRolex（省算力 88% + 省带宽 90%）
- **要最简单实现** → FedRolex（不需要 memory 机制）

## 结论

FedRolex partial training 终点 1.183，比 S3R11（1.119）差 0.06，但省算力 88% 且无需 memory。在边缘设备场景（算力也受限）有独特价值，但纯收敛效果不如 S3R11。

## 日志

- 2026-09-04: 初版脚本编写，因 `freeze_layers()` 冻结所有参数导致 `RuntimeError: element 0 of tensors does not require grad` 崩溃
- 2026-09-04: 修复——关闭 `gradient_checkpointing`（partial training 只训 5 层显存充裕），重新运行成功
- 2026-09-04: 完成 20 轮训练，终点 eval loss 1.1829，生成对比图 `output/fedrolex-vs-all.png`
