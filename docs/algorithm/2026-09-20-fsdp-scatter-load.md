# FSDP 同步：rank0 scatter-load（禁止 8 份整模）

- **状态**：done（3B SecAgg 20 轮验证）
- **创建**：2026-09-20
- **关联**：
  - [当前联调流程](2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
  - [3B 医学 SecAgg 记录](../experiment-records/2026-09-20-qwen25-3b-medical-secagg.md)
  - 代码：`experiments/shared/state_dict_utils.py` `load_full_state_fsdp`、`experiments/run_s3r12v3_fsdp.py` sync-global

## 1. 问题

FSDP 训练时 GPU 上本来就是分片的。联邦每轮对齐全局时，旧路径却按「没有 FSDP」来搬权重：

1. rank0 在 CPU 上持有完整 `global_state`
2. `run_rank0_io_with_heartbeat` 用 `broadcast_object` 把 **整份 state_dict pickle 到每个 rank**
3. `load_full_state_fsdp(..., rank0_only=False)` 要求 **每张卡 CPU 再持一份全量**，再灌进分片

0.5B 全量约 1 GiB，8 份拷贝被机器余量盖住。3B 放大后，第 1 轮 `local_base` 尚能训完；第 2 轮 GPU 已常驻约 28 GiB/卡（计入 RSS），再叠 8 份 CPU 全量，ICC1（503 GiB、无 swap）把 rank 6 **OOM kill**（`exitcode -9`，单进程 RSS ≈ 61 GiB）。ICC2 随后 `timeout waiting for peer keys`，那是连带症状。

生产切片 SCALE-3 曾写「本代不做、cache 轮仍需 broadcast 整模」。那是 0.5B 捷径，和 FSDP 的拆卡设计相反，3B 已证明不能再拖。

## 2. 做法（不改 SecAgg / mask / 量化）

- **sync-global 只广播 meta**（bytes / mode / version）。完整 tensor 留在 rank0 的 `local_global`。
- **`load_full_state_fsdp`**：只有 rank0 传入全量 dict；按 FSDP **叶子 unit**（`Qwen2DecoderLayer`）逐个 `summon`（峰值约一层）+ `writeback`；根节点 `recurse=False` 只写 embed / lm_head / norm。NCCL 不能广播 CPU tensor 时经当前 GPU 中转一层。
- **Round 1 `local_base`**：GPU 分片已是 round-0，**跳过 reload**。
- 3B/7B **不要抄** 0.5B verify yaml 里的 `scale3_sharded_extract: false` 当加载策略；加载路径已与该开关无关。

日志：`Skip FSDP reload: local_base...`；第 2 轮起 `FSDP scatter-load: units=37 leaves=36 copied=434 missing=0`。

## 3. 验收

| | 旧路径（3B R2） | 新路径（`202609201530`） |
|---|---|---|
| ICC1 主机 | 打满 503 GiB，SIGKILL | 训练中约 108 GiB / 503 GiB |
| R2 | 挂 | 过，eval 继续降 |
| R20 | — | eval=0.880，20 轮跑完 |

7B 走同一条路径：主机内存跟 **1 份 CPU 全量 + 一层临时** 走，不是 ×8。Central 3B 聚合 RSS 约 13 GiB；7B 全量 fp32 约 14–28 GiB，Central 440 GiB 先能量，SCALE-1 流式仍未做。
