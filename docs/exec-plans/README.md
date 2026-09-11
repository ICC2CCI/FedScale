# 执行计划（exec-plans）

执行跟踪与 TODO **不要和算法设计、实验记录混放**。算法细节仍在 [`docs/algorithm/`](../algorithm/)，跑次结果仍在 [`docs/experiment-records/`](../experiment-records/)。

## 目录约定

| 目录 | 放什么 | 不要放什么 |
|---|---|---|
| [`active/`](active/) | **正在做**或**下一步要做**的执行计划 / TODO | 已结项的计划 |
| [`completed/`](completed/) | **已经做完**的执行计划（结项说明 + 指向算法/实验记录） | 仍开放的任务 |

**一条计划只存在一处。** 整份做完后把文件从 `active/` **移到** `completed/`（改状态为 `completed`，补结项日期与交付物），不要两边各留一份。

若一份 active 计划里有多条切片（例如生产 TODO 的 A/C/D）：某一切片做完时，**在原文件里把对应 ID 标成 `done`**，并 vis 可在 `completed/` 另写一篇短结项；**未完成切片仍留在 `active/` 那一份里**，不要把未完成任务拷进 completed。

历史文档里若出现 `docs/exe-plans/`，一律视为本目录的旧名。

## 文件命名

`YYYY-MM-DD-<短标题>.md`

文首建议包含：

```markdown
- **状态**：active | completed
- **创建**：YYYY-MM-DD
- **更新**：YYYY-MM-DD
- **结项**：YYYY-MM-DD   # 仅 completed
```

## active

| 计划 | 说明 |
|---|---|
| [双集群联调 → 真实联邦](active/2026-09-11-dual-cluster-to-production.md) | 配置参数化、部分参与、续训、安全与规模化（未完成项） |

## completed

| 计划 | 结项 | 说明 |
|---|---|---|
| [三场景 loss 对比](completed/2026-09-02-three-scenario-loss-comparison.md) | 2026-09-02 | S1/S2/S3 单机联邦对比主线 |
| [S3R12v2 key 级排列](completed/2026-09-04-s3r12v2-block-permutation.md) | 2026-09-04 | 真 20% 带宽（key 级） |
| [S3R12v3 block 均匀](completed/2026-09-04-s3r12v3-block-uniform.md) | 2026-09-04 | block 级均匀上传 |
| [S3R12v3 ratio 消融](completed/2026-09-04-s3r12v3-ratio-ablation.md) | 2026-09-04 | 5%~50% 带宽消融 |
| [S3R11 ratio 消融](completed/2026-09-04-s3r11-ratio-ablation.md) | 2026-09-04 | 层随机 mask 比例消融 |
| [FedRolex 部分训练](completed/2026-09-04-fedrolex-partial-training.md) | 2026-09-04 | 对照实验 |
| [双集群 S3R12v3 + FSDP 联调](completed/2026-09-10-dual-cluster-s3r12v3-fsdp.md) | 2026-09-10 | ICC1/ICC2 + Central 测试床落地 |
