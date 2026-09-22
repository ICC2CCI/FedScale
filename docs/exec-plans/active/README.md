# active — 进行中的执行计划

此处只放 **尚未结项** 的计划。做完后把对应文件移到 [`../completed/`](../completed/)，并在文首把状态改为 `completed`。

当前进行中：

| 计划 | 说明 |
|---|---|
| [Non-IID + 更大模型](2026-09-18-non-iid-and-larger-models.md) | 3B 全量 S2 `202609221148` eval=0.790 已完成。剩余见正文看板：BASE-S2-0.5B、ABL-MASK、DATA-C1、QUANT-8、STAGE-PT、ABL-MDEC |

已归档：

- `2026-09-18-secagg-time-efficiency` → [`../completed/2026-09-18-secagg-time-efficiency.md`](../completed/2026-09-18-secagg-time-efficiency.md)
- `2026-09-11-dual-cluster-to-production` → [`../completed/`](../completed/)

新实验 / SecAgg 对照以这些为准：

- [当前双集群联调流程](../../algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
- [SecAgg 量化精度问题](../../algorithm/2026-09-15-secagg-quantization-precision-issue.md)（已解决）
- [SecAgg 量化优化调研](../../algorithm/2026-09-17-secagg-quantization-optimization-survey.md)（精度 `202609180941`；时间 5 轮 `202609181151` / 20 轮 `202609181406`）
- [FSDP scatter-load](../../algorithm/2026-09-20-fsdp-scatter-load.md)
- [3B 医学 SecAgg](../../experiment-records/2026-09-20-qwen25-3b-medical-secagg.md)
- [7B 医学 SecAgg](../../experiment-records/2026-09-21-qwen25-7b-medical-secagg.md)
- [V100 7B QK fp32](../../algorithm/2026-09-21-v100-7b-qk-fp32.md)
