# S3R12v3 — block 级均匀上传

- **状态**：completed
- **创建**：2026-09-04
- **更新**：2026-09-11
- **结项**：2026-09-04
- **设计**：[S3R12v3 算法](../../algorithm/2026-09-04-s3r12v3-block-uniform.md)
- **前序**：[S3R12v2](2026-09-04-s3r12v2-block-permutation.md)

## 目标

把每个 key 切成约 1MB block，组内 Fisher-Yates + 无放回轮转，使每轮上传量接近恒定（默认 H=5 ≈ 20%）。

## 交付

- 单机 2 client 实验完成；上传波动约 18.5%~21.5%
- 终点 eval loss 约 1.0514（20% 带宽）
- 当前双集群实现沿用此算法（见 [联调结项](2026-09-10-dual-cluster-s3r12v3-fsdp.md)）
