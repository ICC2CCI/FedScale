# STAGE-PT：0.5B FineWeb-Edu + SecAgg（20 轮）

- **日期**：2026-10-08
- **跑次**：`202610081718`
- **TAG**：`pt-fineweb-0.5b-r20`
- **计划**：[STAGE-PT](../exec-plans/active/2026-10-08-stage-pt-federated-continued-pretrain.md)
- **冒烟对照**：`202610081654`（5 轮）

## 设定

与冒烟相同：Qwen2.5-0.5B base，`causal_pt`，FineWeb-Edu ~200M/20M，10% SecAgg（Hadamard INT16，`memory_decay=1.0`），lr=`5e-6`，seq_len=1024，local_steps=30，eval 每 5 轮。

## 主数字（eval CE；约 PPL=exp(CE)）

| round | avg_train | eval_loss | ~PPL |
|---|---|---|---|
| 1 | 2.620 | **2.614** | 13.65 |
| 5 | 2.610 | **2.612** | 13.63 |
| 10 | 2.627 | **2.611** | 13.60 |
| 15 | 2.599 | **2.609** | 13.59 |
| **20** | 2.595 | **2.608** | **13.58** |

SecAgg 全程成功。eval 相对 R1 略降（2.614→2.608）；降幅小，符合当前 token/步预算偏保守。数字**不进** SFT 主表。

## 结果目录

`results/202610081718/`（`metrics.jsonl`、`run.yaml`、logs）
