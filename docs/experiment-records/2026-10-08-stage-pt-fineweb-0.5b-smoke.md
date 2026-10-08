# STAGE-PT 冒烟：0.5B FineWeb-Edu + SecAgg（5 轮）

- **日期**：2026-10-08
- **跑次**：`202610081654`
- **TAG**：`pt-fineweb-0.5b-smoke`
- **计划**：[STAGE-PT](../exec-plans/active/2026-10-08-stage-pt-federated-continued-pretrain.md)

## 设定

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-0.5B base |
| 目标 | `causal_pt`（纯文本，无 chat template） |
| 数据 | FineWeb-Edu `sample/10BT` 抽样 ~200M train + 20M eval |
| 切分 | `data/splits/fineweb-edu-pt/` 文档级 50/50 |
| 协议 | 10% `public_random` + Windowed SecAgg（Hadamard INT16，`memory_decay=1.0`） |
| 超参 | lr=`5e-6`，seq_len=1024，local_steps=30，batch 8×2，5 轮 |

## 结果

| round | avg_train_loss | eval_loss | 备注 |
|---|---|---|---|
| 1 | 2.620 | **2.614** | ~PPL 13.7 |
| 2 | 2.637 | — | eval 每 5 轮 |
| 3 | 2.629 | — | |
| 4 | 2.640 | — | |
| 5 | 2.610 | **2.612** | SecAgg 全程成功 |

验收：协议通路 OK（训练 → SecAgg → 聚合 → 下发）；eval 非 nan。5 轮 eval 几乎持平属预期（步数/LR 小，冒烟不追求降 PPL）。

## 下一步

`PT-RUN-20`：同配置 20 轮正式跑；数字仍不进 SFT 主表。
