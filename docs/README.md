# 文档目录

## 目录结构

| 子目录 | 内容 |
|---|---|
| [`exec-plans/`](exec-plans/) | 执行计划：[`active/`](exec-plans/active/) 进行中，[`completed/`](exec-plans/completed/) 已结项 |
| [`algorithm/`](algorithm/) | 算法与联调流程（S3R12v3 / SecAgg / 安全方案） |
| [`experiment-records/`](experiment-records/) | 各实验的执行记录与结果分析 |
| [`reference/`](reference/) | 复现参数等参考文档 |
| [`ai-design/`](ai-design/) | SecAgg / Block Mask 协议草稿（设计输入，非运维手册） |

## 重点文档（当前系统）

| 文档 | 说明 |
|---|---|
| [当前双集群联调流程](algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md) | **首选**：ICC1/ICC2/Server 操作与传数（含 SecAgg §3.1） |
| [SecAgg 量化精度问题](algorithm/2026-09-15-secagg-quantization-precision-issue.md) | 完整 SecAgg 流程、amax 隐私；问题已解决 |
| [SecAgg 量化优化调研](algorithm/2026-09-17-secagg-quantization-optimization-survey.md) | Hadamard / Issue #1；精度对照 `202609180941` R20=0.985；时间优化 5 轮 `202609181151` / 20 轮 `202609181406` |
| [S3R12v3 算法](algorithm/2026-09-04-s3r12v3-block-uniform.md) | block 级均匀分片算法详细设计 |
| [复现参数](reference/federated-training-reproduction-params.md) | 完整复现参数表 |
| [已完成：双集群 FSDP 联调](exec-plans/completed/2026-09-10-dual-cluster-s3r12v3-fsdp.md) | 测试床落地结项 |
| [已完成：联调 → 真实联邦](exec-plans/completed/2026-09-11-dual-cluster-to-production.md) | CFG/SEC/SCALE 等切片结项 |
| [已完成：SecAgg 时间效率](exec-plans/completed/2026-09-18-secagg-time-efficiency.md) | 通用加速（不特化 0.5B）；5 轮 `202609181151` / 20 轮 `202609181406` |

## 历史 / 部分过时（勿当操作手册）

| 文档 | 说明 |
|---|---|
| [双集群 FSDP 部署规划](algorithm/2026-09-09-dual-cluster-fsdp-deployment.md) | 早期规划；架构方向仍对，细节以 current-flow 为准 |
| [上传安全方案](algorithm/2026-09-08-upload-security-plan.md) | SEC 方案设计；实现状态见 completed 计划 |
| `experiment-records/*`、`algorithm/2026-09-04-*` | 算法演进与消融记录，有效但不是当前运维入口 |
| `flowertune-llm/`、`scripts/run-federated*.sh` | **旧 Flower/K8s 路径**，与当前 ICC+MinIO 实验无关 |
