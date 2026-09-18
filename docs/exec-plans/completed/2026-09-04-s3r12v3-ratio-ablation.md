# S3R12v3 ratio 消融（5%~50%）

- **状态**：completed
- **创建**：2026-09-04
- **更新**：2026-09-11
- **结项**：2026-09-04
- **记录**：[实验记录](../../experiment-records/2026-09-04-s3r12v3-ratio-ablation.md)
- **基线**：[S3R12v3 20%](2026-09-04-s3r12v3-block-uniform.md)

## 目标

在单机模拟联邦上测 H / slots 映射的 5%、10%、20%、30%、40%、50% 上传，画出带宽–收敛权衡。

## 交付

- 脚本 `experiments/run_s3r12v3_ratio.py` 与各比例 round log
- 结论：20% 是带宽与 loss 的常用折中；更低比例明显变差
- **未做（历史）**：把同一套比例开关接到双集群启动路径；跟踪见 [联调 → 真实联邦](2026-09-11-dual-cluster-to-production.md) ALG-1 / ALG-2（该计划已迁入 completed）
