# S3R12v2 — key 级分层排列轮转

- **状态**：completed
- **创建**：2026-09-04
- **更新**：2026-09-11
- **结项**：2026-09-04
- **设计**：[S3R12v2 算法](../../algorithm/2026-09-04-s3r12v2-block-permutation.md)
- **前序**：[三场景主线](2026-09-02-three-scenario-loss-comparison.md)

## 目标

相对 S3R11：把 non_layer 纳入轮转，在 **key 级**做到真正约 20% 带宽。

## 交付

- 单机实验脚本与 round log
- 终点 eval loss 约 1.057；实际上传接近 20%
- 遗留：key 大小不均导致每轮上传 7%~37% 波动 → 由 [S3R12v3](2026-09-04-s3r12v3-block-uniform.md) 用 block 级选择解决
