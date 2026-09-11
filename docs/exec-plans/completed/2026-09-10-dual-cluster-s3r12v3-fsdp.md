# 双集群 S3R12v3 + FSDP 联调落地

- **状态**：completed
- **创建**：2026-09-09
- **更新**：2026-09-11
- **结项**：2026-09-10
- **流程说明（以代码为准）**：[当前联调流程](../../algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
- **早期规划（部分过时）**：[部署规划](../../algorithm/2026-09-09-dual-cluster-fsdp-deployment.md)
- **算法**：[S3R12v3](../../algorithm/2026-09-04-s3r12v3-block-uniform.md)
- **后续未完成项**：[active：联调 → 真实联邦](../active/2026-09-11-dual-cluster-to-production.md)

## 目标

在 ICC1、ICC2（各 8 卡 FSDP）+ Central（聚合 + MinIO）上跑通 S3R12v3，去掉 Flower，权重走对象存储、控制面走 HTTP。

## 交付

- 聚合服务、FSDP 客户端、增量 `global_delta`、fp16 传输、在线 eval
- MinIO 假死：硬超时 + 重建连接
- 一键启动 `scripts/start_s3r12v3_fsdp_run.sh`；参考跑次 `results/202609101345/`
- 默认联调：2 client、20 轮、H=5 ≈ 20%、约 30 step/轮、Qwen2.5-0.5B

## 结项边界（以下不算本计划范围）

本计划只证明 **测试床链路能跑**。下列已单独记在 active TODO，不在本结项内：

- 统一 yaml 配置；改 `RATIO` 实际改 H
- 部分参与 / 断点续训 / 控制面鉴权
- 每 N 轮才写全量 `global_state`
- 7B/13B 内存与 int8
