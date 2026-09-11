# 文档目录

## 目录结构

| 子目录 | 内容 |
|---|---|
| [`exec-plans/`](exec-plans/) | 执行计划：[`active/`](exec-plans/active/) 进行中，[`completed/`](exec-plans/completed/) 已结项（不要混放） |
| [`algorithm/`](algorithm/) | 算法设计文档（S3R12v3 / S3R12v2 / FedRolex / 安全方案） |
| [`experiment-records/`](experiment-records/) | 各实验的执行记录与结果分析 |
| [`reference/`](reference/) | 复现参数等参考文档 |

## 重点文档

| 文档 | 说明 |
|---|---|
| [当前双集群联调流程](algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md) | ICC1/ICC2/Server 操作与传数流程（白话，以已落地实现为准） |
| [进行中：联调 → 真实联邦](exec-plans/active/2026-09-11-dual-cluster-to-production.md) | 配置参数化、部分参与、续训、安全与规模化（active TODO） |
| [已完成：双集群 FSDP 联调](exec-plans/completed/2026-09-10-dual-cluster-s3r12v3-fsdp.md) | 测试床落地结项 |
| [S3R12v3 算法](algorithm/2026-09-04-s3r12v3-block-uniform.md) | block 级均匀分片算法详细设计 |
| [双集群 FSDP 部署规划](algorithm/2026-09-09-dual-cluster-fsdp-deployment.md) | 早期架构规划（部分已过时） |
| [S3R12v3 ratio 消融](experiment-records/2026-09-04-s3r12v3-ratio-ablation.md) | 5%~50% 带宽消融实验 |
| [复现参数](reference/federated-training-reproduction-params.md) | 完整复现参数表 |
| [上传安全方案](algorithm/2026-09-08-upload-security-plan.md) | 分片上传的隐私保护设计 |
