# 三场景 loss 对比（S1 / S2 / S3）

- **状态**：completed
- **创建**：2026-09-02
- **更新**：2026-09-11
- **结项**：2026-09-02
- **设计 / 记录**：
  - [S1-D 医疗基线](../../experiment-records/2026-09-02-s1d-baseline-medical.md)
  - [S2 全量上传联邦](../../experiment-records/2026-09-02-s2-federated-full.md)
  - [S3 shard 20%](../../experiment-records/2026-09-02-s3-federated-shard20.md)
  - [S3R residual](../../experiment-records/2026-09-02-s3r-residual.md)

## 目标

在同一数据与模型上对比：单机全量微调（S1）、联邦全量上传（S2）、联邦分片上传（S3），得到可画 loss 曲线的正式基线。

## 交付

- 数据集定为 medical flashcards（S1-D），替换 alpaca 容量墙
- S2 / S3 / S3R 单机模拟 2 client 联邦跑通并留下 `round_log`
- 后续 S3R11/S3R12 系列从此主线分出（见同目录其它结项）

## 结项说明

单机对比主线已完成。后续算法迭代不再改本计划，另开执行计划。
