# FedRolex partial training 对照

- **状态**：completed
- **创建**：2026-09-04
- **更新**：2026-09-11
- **结项**：2026-09-04
- **记录**：[算法/结果](../../algorithm/2026-09-04-fedrolex-partial-training.md)
- **前序**：[三场景主线](2026-09-02-three-scenario-loss-comparison.md)

## 目标

对照「只训部分层」vs S3 系「全训只传部分」。单机 2 client、同一医疗数据与 0.5B 模型。

## 交付

- `experiments/run_fedrolex_partial_training.py` 与 round log
- 作为历史对照，不进入双集群主路径
