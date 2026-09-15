# FedScale 评估清单与数据清单

## 一、评估清单（四大类别）

### 类别 1: 集群内训练性能

- **聚合模块**: `evaluation/training_performance.py`
- **数据来源**: `flowertune_llm/metrics.py` → `metrics_detailed.json`
- **补齐工具**: `measure_sharded_state_export()`, `measure_checkpoint_restore()`

| 指标 | 说明 | 数据来源 |
|---|---|---|
| `avg_forward_ms` | 前向传播时间（含 loss 计算） | `StepMetricsCallback.record_forward()` + `timed_compute_loss` |
| `avg_backward_ms` | 反向传播时间（total − forward − comm − optimizer） | `StepMetricsCallback.on_step_end()` |
| `avg_comm_ms` | 集群内通信总时间 | monkey-patch `dist.all_reduce` / `all_gather` / `reduce_scatter` |
| `avg_all_reduce_ms` | DDP All-Reduce 时间 | `ddp_comm_hook` + `dist.all_reduce` patch |
| `avg_all_gather_ms` | FSDP All-Gather 时间 | `dist.all_gather` / `all_gather_into_tensor` patch |
| `avg_reduce_scatter_ms` | FSDP Reduce-Scatter 时间 | `dist.reduce_scatter` / `reduce_scatter_tensor` patch |
| `avg_optimizer_ms` | 优化器更新时间 | `on_pre_optimizer_step` / `on_optimizer_step` |
| `avg_step_ms` | 单 step 总时间 | `on_step_begin` / `on_step_end` |
| `total_train_time_s` | 总训练时间 | `on_train_begin` → `on_train_end` |
| `throughput_tokens_per_s` | 吞吐量 (tokens/s) | `total_tokens / total_train_time_s` |
| `total_all_reduce_bytes` | DDP All-Reduce 总流量 | collective hook 统计 |
| `total_all_gather_bytes` | FSDP All-Gather 总流量 | collective hook 统计 |
| `total_reduce_scatter_bytes` | FSDP Reduce-Scatter 总流量 | collective hook 统计 |
| `full_state_export_s` | FSDP full-state 导出时间 | `distributed_trainer.py` 已有计时 |
| `sharded_state_export_s` | FSDP sharded-state 导出时间 | `measure_sharded_state_export()` **（新增）** |
| `state_dict_conversion_s` | 状态字典转换时间 | `distributed_trainer.py` 已有计时 |
| `state_serialization_s` | 状态序列化保存时间 | `distributed_trainer.py` 已有计时 |
| `state_bytes` | 状态字典字节数 | `distributed_trainer.py` 已有计时 |
| `checkpoint_save_s` | checkpoint 保存时间 | `server_app.py` `checkpoint_save_s` |
| `checkpoint_bytes` | checkpoint 字节数 | `server_app.py` `checkpoint_bytes` |
| `checkpoint_restore_s` | checkpoint 恢复时间 | `distributed_trainer.py` **（新增）** |

---

### 类别 2: 集群内资源使用

- **聚合模块**: `evaluation/resource_usage.py`
- **数据来源**: `flowertune_llm/metrics.py` `ResourceMonitor` → `metrics_detailed.json`

| 指标 | 说明 | 数据来源 |
|---|---|---|
| `gpu_memory_peak_mb` | GPU 显存峰值 | `torch.cuda.max_memory_allocated()` + 后台采样 |
| `gpu_utilization_avg_pct` | GPU 利用率均值 | NVML / `nvidia-smi` 后台采样 |
| `cpu_utilization_avg_pct` | CPU 利用率均值 | `psutil.cpu_percent()` 后台采样 |
| `cpu_memory_peak_mb` | CPU 内存峰值 | `psutil.Process().memory_info().rss` |
| `network_rx_bytes` | 网络接收字节数 | `/proc/net/dev` 前后差值（排除 lo） |
| `network_tx_bytes` | 网络发送字节数 | `/proc/net/dev` 前后差值（排除 lo） |
| `network_total_bytes` | 网络总流量 | `rx + tx` |
| `total_nccl_bytes` | NCCL 通信总字节数 | collective hook 统计 |
| `nccl_collective_calls` | NCCL collective 调用次数 | collective hook 统计 |
| `avg_nccl_comm_ms` | 平均 NCCL 通信时间 | collective hook 统计 |

---

### 类别 3: 跨智算中心联邦更新

- **聚合模块**: `evaluation/federated_timing.py`
- **数据来源**: `client_app.py` + `server_app.py` → `federated_metrics_round_N.json` + `experiment_summary.json`

| 指标 | 说明 | 数据来源 |
|---|---|---|
| `model_delta_export_s` | 本地模型更新导出时间 | `client_app.py` `t_model_delta_export_s` |
| `full_update_compression_s` | 压缩编码时间（Top-K / FedScale INT8） | `client_app.py` `compression_seconds` |
| `wan_download_s` | 集群间下行传输时间（global model → client） | `client_app.py` **（新增）** |
| `wan_upload_s` | 集群间上行传输时间（client → server） | `client_app.py` **（新增）** |
| `wan_transfer_total_s` | WAN 传输总时间 | `wan_download_s + wan_upload_s` **（新增，派生）** |
| `server_fedavg_aggregation_s` | 明文 FedAvg 聚合时间 | `server_app.py` `TimedFedAvg` / `FedScaleBlockFedAvg` |
| `federated_cycle_s` | 单次联邦轮总时间 | `server_app.py` `round_timings` |
| `server_post_aggregation_s` | 服务端后聚合处理时间（含 checkpoint） | `server_app.py` |
| `checkpoint_save_s` | 服务端 checkpoint 保存时间 | `server_app.py` |
| `client_training_s` | 客户端训练时间 | `client_app.py` `client_training_seconds` |
| `client_evaluation_s` | 客户端评估时间 | `client_app.py` `client_evaluation_seconds` |
| `client_round_s` | 客户端整轮时间 | `client_app.py` `client_round_seconds` |
| `model_delta_bytes` | 模型增量字节数 | `client_app.py` `model_delta_bytes` |
| `object_store_uploaded_bytes` | 对象存储上传字节数 | `client_app.py` |

---

### 类别 4: 模型微调准确度

- **模块**: `validation_metrics.py`, `generation_metrics.py`, `baseline_comparison.py`, `mmlu_evaluator.py`
- **对比模块**: `comparison_report.py`

#### 4a. 验证损失 / PPL（路径 A）

| 指标 | 说明 |
|---|---|
| `val_loss` | 仅对 assistant response tokens 计算的交叉熵损失 |
| `perplexity` | `exp(val_loss)` |
| `evaluated_samples` / `skipped_samples` | 有效 / 跳过样本数 |
| `answer_tokens` | 评估的 response token 总数 |

**关键设计**: 使用 `return_offsets_mapping=True` 精确定位 prompt/response 边界（P0 修复）。

#### 4b. 自由生成下游指标（路径 B）

| 指标 | 说明 |
|---|---|
| `rouge_l` (precision / recall / f1) | ROUGE-L，优先用 `rouge_score` 库，无依赖时回退到 LCS |
| `bertscore` (f1 / precision / recall) | BERTScore，库未安装返回 `None`（P0 修复） |
| `token_overlap_accuracy` | ≥30% reference tokens 出现在 prediction 中算正确 |
| `macro_f1` | Token 级别宏平均 F1 |
| `exact_match` | 预测与参考完全匹配的比例 |
| `generation_quality` | 空预测数、平均生成 token 数等 |

**关键设计**: 确定性解码 `do_sample=False, num_beams=1`（§13）；左 padding（P1 修复）。

#### 4c. 基线对比与 Delta

| 指标 | 说明 |
|---|---|
| `delta.val_loss` | 联邦 − 基座（负 = 改善） |
| `delta.perplexity` | PPL 差 |
| `delta.rouge_l_f1` | ROUGE-L F1 差 |
| `delta.bertscore_f1` | BERTScore F1 差 |
| `delta.token_overlap_accuracy` | 准确率差 |
| `delta.macro_f1` | 宏 F1 差 |
| `delta.exact_match` | EM 差 |

#### 4d. MMLU / MMLU-Pro 外部准确率

| 指标 | 说明 |
|---|---|
| `accuracy` | 总体多项选择准确率 |
| `num_questions` / `correct` | 评估题数 / 正确数 |
| `per_subject` | 按学科分组的准确率明细 |

#### 4e. DDP vs FSDP 对比报告

- **模块**: `comparison_report.py`
- **入口**: `generate_comparison_report(ddp_dir, fsdp_dir)` → `ComparisonReport`
- **输出**: JSON + Markdown，含四类别对比表

在相同 global batch size、训练步数和随机种子下对比 DDP 与 FSDP 的最终模型质量:

| Metric | DDP | FSDP | Delta |
|---|---|---|---|
| Val Loss ↓ | ... | ... | ... |
| Perplexity ↓ | ... | ... | ... |
| ROUGE-L F1 ↑ | ... | ... | ... |
| BERTScore F1 ↑ | ... | ... | ... |
| Token Acc ↑ | ... | ... | ... |
| Macro F1 ↑ | ... | ... | ... |
| Exact Match ↑ | ... | ... | ... |

---

### 扩展实验

#### E3: 压缩-效用报告

- **模块**: `compression_report.py`

| 指标 | 说明 |
|---|---|
| `original_bytes` / `compressed_bytes` | 原始 / 压缩字节数 |
| `compression_ratio` | 压缩比 |
| `quantization_error_l2` / `_relative` | INT8 量化误差 |
| `param_coverage` | 掩码后参数比例 |

配置矩阵（9 种）: no_compression / int8_only / mask_25%/10%/5% / mask+int8 / mask+int8+residual

#### E4: DP 隐私预算报告

- **模块**: `dp_report.py`

| 指标 | 说明 |
|---|---|
| `epsilon` | (ε, δ)-DP 中的 ε |
| `delta` | 目标 δ |
| `alpha_optimal` | 最优 Rényi 阶 α |
| `rdp_per_round` | 每轮 RDP 值 |
| `sigma_agg` / `sensitivity` | 聚合噪声 / L2 敏感度 |

#### 三方数据划分

- **模块**: `data_split.py`
- `stable_sample_key` — SHA-256(instruction + NUL + input)
- `create_train_val_test_split` — 5% val / 5% test / 90% train
- `reconstruct_held_out_test_set` — 重建联邦各分区 held-out test 并集

---

## 二、数据清单

| # | 数据 | 来源 | 用途 | 需要的列 / 字段 |
|---|---|---|---|---|
| 1 | **Alpaca-GPT4** | `vicgalle/alpaca-gpt4` (HuggingFace) | 类别 4: 路径 A/B + 基线对比 | `instruction`, `input`, `output` → `text`（含 `### Response:`） |
| 2 | **MMLU** | `cais/mmlu` (HuggingFace) | 类别 4: 外部准确率 | `question`, `choices` (4), `answer` (0–3), `subject` |
| 3 | **MMLU-Pro** | `TIGER-Lab/MMLU-Pro` (HuggingFace) | 类别 4: 外部准确率（可选） | `question`, `options` (≤10), `answer_index`, `category` |
| 4 | **基座模型** | `openlm-research/open_llama_3b_v2` 或本地缓存 | 类别 4: 基线 + 联邦模型底座 | 模型权重 + tokenizer |
| 5 | **联邦模型状态** | 训练产出的 checkpoint | 类别 4: 评估对象 + 类别 1: 压缩报告 | PEFT adapter 或 state_dict `.pt` |
| 6 | **DP 参数** | CLI / 环境变量 | E4: DP 报告 | `num_rounds`, `clip_norm`, `sigma_client`, `cohort_size`, `delta` |
| 7 | **metrics_detailed.json** | `distributed_trainer.py` 产出 | 类别 1 + 2: 训练性能 + 资源使用 | `training` + `resources` + `federated` 字段 |
| 8 | **federated_metrics_round_N.json** | `server_app.py` 产出 | 类别 3: 联邦更新时序 | `federated_cycle_s`, `server_fedavg_aggregation_s` 等 |
| 9 | **experiment_summary.json** | `server_app.py` 产出 | 类别 3 + 4: 联邦时序 + 最终评估 | `round_timings`, `final_evaluation_metrics` |

---

## 三、数据依赖关系

```
                    ┌─── metrics_detailed.json ──→ training_performance (类别 1)
                    │                          └─→ resource_usage (类别 2)
                    │
训练实验产出 ───────┼─── federated_metrics_round_N.json ──→ federated_timing (类别 3)
                    │
                    ├─── experiment_summary.json ──→ federated_timing (类别 3)
                    │                            └─→ comparison_report (类别 4e)
                    │
                    └─── model_weights.pt / adapter ──→ validation_metrics (类别 4a)
                                                   ──→ generation_metrics (类别 4b)
                                                   ──→ baseline_comparison (类别 4c)
                                                   ──→ mmlu_evaluator (类别 4d)
                                                   ──→ compression_report (E3)

Alpaca-GPT4 ──→ data_split ──→ test set ──→ 类别 4a/4b/4c
MMLU/MMLU-Pro ──→ mmlu_evaluator ──→ 类别 4d
DP 参数 (无需数据) ──→ dp_report ──→ E4
```

---

## 四、运行时数据获取方式

| 数据 | 获取方式 | 是否需要联网 | 本地缓存位置 |
|---|---|---|---|
| Alpaca-GPT4 | `flwr_datasets.FederatedDataset` 分区加载 | 首次需要 | `$HF_HOME/hub/datasets--vicgalle--alpaca-gpt4` |
| MMLU | `datasets.load_dataset("cais/mmlu", "all", split="test")` | 首次需要 | `$HF_HOME/hub/datasets--cais--mmlu` |
| MMLU-Pro | `datasets.load_dataset("TIGER-Lab/MMLU-Pro", "default", split="test")` | 首次需要 | `$HF_HOME/hub/datasets--TIGER-Lab--MMLU-Pro` |
| 基座模型 | `AutoModelForCausalLM.from_pretrained(local_files_only=True)` | 否 | `$HF_HOME/hub/models--*` |
| 联邦 checkpoint | `MODEL_STATE_PATH` / `--adapter-path` | 否 | 训练产出路径 |
| DP 参数 | 环境变量 / CLI 参数 | 否 | — |
| metrics_detailed.json | 训练 Job 自动产出 | 否 | `$OUTPUT_DIR/$JOB_NAME/` |
| federated_metrics_round_N.json | ServerApp 自动产出 | 否 | `$RESULTS_DIR/` |
| experiment_summary.json | ServerApp 自动产出 | 否 | `$RESULTS_DIR/` |

> **K8s 部署**: 基座模型和数据集在镜像构建时预下载到 `$HF_HOME`，运行时 `local_files_only=True`。

---

## 五、入口与运行方式

### 方式 1: 联邦运行时自动触发（K8s Job）

`final_evaluator.py` — 由 Flower ClientApp 的 evaluate handler 启动。

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `MODEL_NAME` | 基座模型名 | — |
| `MODEL_STATE_PATH` | 联邦模型状态文件路径 | — |
| `EVALUATION_OUTPUT_PATH` | 评估结果输出路径 | — |
| `PARTITION_ID` / `NUM_PARTITIONS` | 联邦分区参数 | — |
| `DATASET_NAME` | HuggingFace 数据集名 | — |
| `EVAL_VAL_RATIO` / `EVAL_TEST_RATIO` | 验证 / 测试集比例 | `0.05` / `0.05` |
| `NUM_EVAL_SAMPLES` | 评估样本数 | `50` |
| `EVAL_MAX_LENGTH` / `EVAL_MAX_NEW_TOKENS` | 最大长度 / 最大生成 token 数 | `512` / `128` |
| `EVAL_SKIP_BASE` | 跳过基线评估 | `false` |
| `EVAL_MMLU_DATASET` | MMLU 数据集名（空 = 不评估） | `""` |
| `EVAL_MMLU_SAMPLES` | MMLU 评估题数 | `100` |
| `EVAL_COMPRESSION_REPORT` | 生成压缩报告 | `false` |
| `EVAL_DP_SIGMA_CLIENT` | DP σ 值（空 = 不生成 DP 报告） | `""` |
| `EVAL_DP_NUM_ROUNDS` / `EVAL_DP_CLIP_NORM` / `EVAL_DP_COHORT_SIZE` / `EVAL_DP_DELTA` | DP 参数 | `50` / `1.0` / `2` / `1e-5` |

### 方式 2: 独立模型评估 CLI

`run_evaluation.py` — 在任何有 GPU 的机器上独立运行模型准确度评估。

```bash
python -m evaluation.run_evaluation \
    --adapter-path /path/to/checkpoint \
    --base-model openlm-research/open_llama_3b_v2 \
    --dataset vicgalle/alpaca-gpt4 \
    --output-dir ./eval_results \
    --max-samples 100 \
    --with-baseline \
    --mmlu-dataset cais/mmlu \
    --compression-report \
    --dp-sigma-client 0.5
```

### 方式 3: 统一报告生成 CLI（新增）

`generate_report.py` — 从实验产出目录生成四类别统一报告。

```bash
# 单次实验报告
python -m evaluation.generate_report \
    --results-dir /app/results/experiment-001 \
    --output report.json

# DDP vs FSDP 对比报告（Markdown + JSON）
python -m evaluation.generate_report \
    --ddp-dir /app/results/ddp-experiment \
    --fsdp-dir /app/results/fsdp-experiment \
    --output-dir ./reports \
    --format both
```

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--results-dir` | 单次实验结果目录 | — |
| `--ddp-dir` | DDP 实验结果目录（对比模式） | — |
| `--fsdp-dir` | FSDP 实验结果目录（对比模式） | — |
| `--output` | 输出文件路径 | `report.json` / `comparison_report.json` |
| `--output-dir` | 输出目录（`--format both` 时） | — |
| `--format` | 输出格式: `json` / `markdown` / `both` | `json` |

---

## 六、代码文件一览

### `evaluation/` 包（评估代码收拢目录）

| 文件 | 类别 | 轻量? | 说明 |
|---|---|---|---|
| `__init__.py` | — | ✅ | 包入口，轻量 eager + 重型 lazy |
| `data_split.py` | 基础 | ✅ | 三方数据划分（SHA-256 稳定哈希） |
| `validation_metrics.py` | 4a | ❌ | 路径 A: teacher-forced loss / PPL |
| `generation_metrics.py` | 4b | ❌ | 路径 B: ROUGE-L / BERTScore / Acc / F1 / EM |
| `baseline_comparison.py` | 4c | ❌ | 基线对比与 delta |
| `mmlu_evaluator.py` | 4d | ❌ | MMLU / MMLU-Pro 外部准确率 |
| `comparison_report.py` | 4e | ✅ | DDP vs FSDP 四类别对比报告 |
| `training_performance.py` | 1 | ✅ | 训练性能聚合 + sharded-state export |
| `resource_usage.py` | 2 | ✅ | 资源使用聚合 |
| `federated_timing.py` | 3 | ✅ | 联邦更新时序聚合 + WAN 拆分 |
| `compression_report.py` | E3 | ✅ | 压缩-效用报告 |
| `dp_report.py` | E4 | ✅ | DP RDP 核算 + DP-utility 曲线 |
| `final_evaluator.py` | 4 | ❌ | 联邦运行时评估入口 (K8s Job) |
| `run_evaluation.py` | 4 | ❌ | 独立模型评估 CLI |
| `generate_report.py` | 全部 | ✅ | 统一报告生成 CLI |

### `flowertune_llm/` 中的数据采集代码（已被评估包聚合）

| 文件 | 采集内容 | 写入文件 |
|---|---|---|
| `metrics.py` `StepMetricsCallback` | 类别 1: forward/backward/comm/optimizer/step | `metrics_detailed.json` |
| `metrics.py` `ResourceMonitor` | 类别 2: GPU/CPU/网络 | `metrics_detailed.json` |
| `distributed_trainer.py` | 类别 1: state export + checkpoint restore **(新增)** | `metrics_detailed.json` + `metrics.json` |
| `client_app.py` | 类别 3: delta export + WAN download/upload **(新增)** + compression | `metrics_detailed.json` + `federated_metrics` |
| `server_app.py` | 类别 3: FedAvg aggregation + federated cycle + checkpoint | `federated_metrics_round_N.json` + `experiment_summary.json` |
