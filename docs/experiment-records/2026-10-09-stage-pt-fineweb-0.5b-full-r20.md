# STAGE-PT：0.5B FineWeb-Edu + SecAgg 完整 PT（20 轮，阶段二）

- **日期**：2026-10-09
- **跑次**：`202610091639`
- **TAG**：`pt-fineweb-0.5b-full-r20`
- **计划**：[STAGE-PT](../exec-plans/active/2026-10-08-stage-pt-federated-continued-pretrain.md)
- **冒烟对照**：`202610091412`（5 轮，同配置）
- **阶段一对照**：`202610081718`（local_steps=30，无 packing，200M 语料）

## 设定

| 项 | 值 | 与阶段一区别 |
|---|---|---|
| 模型 | Qwen2.5-0.5B base | 同 |
| objective | `causal_pt`（纯文本续预训练） | 同 |
| **packing** | **true**（`PackedDataset`：EOS 拼接 + 切块到 seq_len） | 阶段一无 packing |
| **local_steps** | **200** | 阶段一 30 |
| **train tokens** | **~500M**（每端 250M，210401/210402 文档） | 阶段一 200M |
| **eval tokens** | **~50M**（41967 文档） | 阶段一 20M |
| batch_size × grad_accum | 8 × 2 | 同 |
| lr | 5e-6 | 同 |
| seq_len | 1024 | 同 |
| SecAgg | Hadamard + INT16，`memory_decay=1.0` | 同 |
| coverage_h | 10（~10%） | 同 |
| eval_every_n_rounds | **2** | 阶段一 5 |
| client_upload_timeout_s | **3600** | 阶段一 1800（阶段二因 tokenize + 训练时间更长而提高） |
| num_rounds | 20 | 同 |

**数据**：FineWeb-Edu `sample/10BT` 第 0 分片（1 个 parquet，~2GB），chars/4 抽样 500M train + 50M eval，文档级随机 50/50 切分 ICC1/ICC2。

**PackedDataset 效果**：21 万文档经 EOS 拼接后切成 1024-token 块，每端 ~21 万个 packed chunk。相比阶段一 per-doc padding（短文档大量 pad 浪费 40-60% tokens），packing 后每个 chunk 全是有效 token，token 密度大幅提升。

## 主数字（eval CE；约 PPL=exp(CE)）

| round | avg_train | eval_loss | ~PPL |
|---|---|---|---|
| 1 | 2.6817 | **2.6181** | 13.71 |
| 2 | 2.6818 | **2.6176** | 13.70 |
| 4 | 2.6868 | **2.6165** | 13.69 |
| 6 | 2.6851 | **2.6154** | 13.68 |
| 8 | 2.6737 | **2.6142** | 13.67 |
| 10 | 2.6749 | **2.6133** | 13.65 |
| 12 | 2.6680 | **2.6124** | 13.64 |
| 14 | 2.6620 | **2.6111** | 13.62 |
| 16 | 2.6532 | **2.6105** | 13.61 |
| 18 | 2.6710 | **2.6101** | 13.61 |
| **20** | 2.6686 | **2.6101** | **13.60** |

SecAgg 全程成功（20/20 轮），无 nan/OOM/error。

## 与阶段一对比

| | 阶段一（`202610081718`） | 阶段二（`202610091639`） |
|---|---|---|
| packing | 无 | **有**（PackedDataset） |
| local_steps | 30 | **200** |
| train tokens | 200M | **500M** |
| eval tokens | 20M | **50M** |
| R1 eval | 2.614 | 2.618 |
| R20 eval | 2.608 | **2.610** |
| 降幅（R1→R20） | 0.006 | **0.008** |
| 平均整轮墙钟 | ~385s | **~943s**（~15.7 min） |
| 每轮上传 | ~140 MiB | ~117-140 MiB |

## 分析

### eval 降幅仍偏小

eval_loss 从 R1 的 2.6181 降到 R20 的 2.6101，20 轮降了 **0.008**——比阶段一（0.006）好，但远未达到"明显下降（>0.02）"的验收标准。

**根因分析**：`lr=5e-6` 对于续预训练来说太小。Qwen2.5-0.5B base 已在 FineWeb-Edu 上预训练过，以如此小的 lr 续训，模型权重几乎不动。train_loss 也仅从 2.68 降到 2.66，佐证了学习率不足。

### train_loss 趋势

train_loss 在 R1-R5 有轻微上升（2.6817→2.6879），随后逐渐下降到 R16 的 2.6532，R17-R20 回升到 ~2.67。这种非单调行为可能与 block mask 轮转（epoch 0→1 时 coverage 重置）和 lr 过小导致训练信号弱有关。

### SecAgg 信号质量

SecAgg SQNR 从 R1 的 54dB 逐渐降到 R20 的 ~41dB，cosine 从 0.9999 降到 0.991——仍在可接受范围，但表明随着训练进行，delta 的动态范围在增大，量化噪声占比上升。这与 lr 过小导致 delta 整体很小、相对量化噪声增大一致。

### 整轮墙钟

平均 ~943s/轮（~15.7 min），其中训练 ~680s、eval ~375s（eval 轮）、SecAgg 编码+上传 ~30s、聚合等待 ~10s。相比阶段一 ~385s/轮增长约 2.4 倍，与 local_steps 从 30→200（6.7 倍）不完全成比例，因为 eval 和通信开销是固定的。

## 代码改动

本次实验涉及以下代码变更（相对阶段一）：

| 文件 | 改动 |
|---|---|
| `experiments/run_s3r12v3_fsdp.py` | 新增 `PackedDataset` 类（EOS 拼接 + 切块到 seq_len）；新增 `--packing` CLI 参数；训练数据按 `packing && causal_pt` 选择 `PackedDataset`（eval 不 pack） |
| `experiments/shared/run_config.py` | `RUN_CONFIG_SCHEMA` train section 加 `packing` |
| `scripts/start_s3r12v3_fsdp_run.sh` | 从 yaml 读 `PACKING`，传 `--packing`/`--no-packing` 给客户端 |
| `configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml` | `packing: true`、`local_steps: 200`、`eval_every_n_rounds: 2`、`client_upload_timeout_s: 3600` |

## 冒烟过程

- 首次冒烟 `202610091001` 因 `client_upload_timeout_s=1800` 不足（tokenize 21 万文档 ~15 min + 训练 200 steps ~18 min = ~33 min 超过 30 min 限制），server 在 upload timeout 后标记 round 失败。
- 修复：`client_upload_timeout_s` 从 1800 提到 3600。
- 第二次冒烟 `202610091412`（5 轮）成功完成。

## 后续建议

| 方向 | 说明 |
|---|---|
| **提高 lr** | `5e-6` → `1e-5` 或 `2e-5`，使续预训练信号更强。当前 lr 下模型几乎不动。 |
| **增加轮次** | 如果 lr 不变，可能需要 50-100 轮才能看到明显下降。但加 lr 更高效。 |
| **PT-BASE-S2 对照** | 同数据无 SecAgg 全量 FedAvg，量化隐私聚合代价。当前 eval 降幅小，S2 对照可以确认"降幅小是 lr 问题还是协议问题"。 |
| **更大数据量** | 500M tokens 对 0.5B CPT 可能不够（社区常用 1B+）。可抽到 1B 再试。 |

## 结果目录

`results/202610091639/`

- `metrics.jsonl`：每轮 train/eval loss
- `round_log.json`：每轮详细日志（含 SecAgg 指标、墙钟分解）
- `run.yaml` / `run_meta.json`：生效配置
- `figures/`：eval_loss.png、train_loss.png、time_breakdown.png、download_mode.png、transfer_mib.png
- `logs/`：aggregation_server.log、client0_fsdp.log、client1_fsdp.log
