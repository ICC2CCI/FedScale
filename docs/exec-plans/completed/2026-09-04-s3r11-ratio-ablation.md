# S3R11 mask ratio 消融

- **状态**：completed
- **创建**：2026-09-04
- **更新**：2026-09-11
- **结项**：2026-09-04
- **记录**：[实验记录](../../experiment-records/2026-09-04-s3r11-ratio-ablation.md)
- **前序**：[三场景主线](2026-09-02-three-scenario-loss-comparison.md)

## 目标

S3R11（随机选层 + memory）在 10%~50% mask 下的收敛与带宽权衡。

## 交付

- 各比例 round log 与带宽–loss 曲线
- 后续主路径改为 S3R12v2/v3（block 级），本消融不再迭代
