# 评估与数据清单

## 第一部分：评估清单

> 评估代码全部在 `evaluation/` 包中，只读取运行时记录的数据，不嵌入训练代码。

### 类别 1: 集群内训练性能

**评估模块**: `evaluation/training_performance.py`
**入口函数**: `aggregate_training_performance(metrics_paths)`

| # | 评估点 | 输出字段 | 依赖数据 | 数据来源文件 |
|---|---|---|---|---|
| 1.1 | 前向传播时间 | `avg_forward_ms` | `training.steps[].forward_ms` | `metrics_detailed.json` |
| 1.2 | 反向传播时间 | `avg_backward_ms` | `training.steps[].backward_ms` | `metrics_detailed.json` |
| 1.3 | 集群内通信总时间 | `avg_comm_ms` | `training.steps[].comm_ms` | `metrics_detailed.json` |
| 1.4 | DDP All-Reduce 时间 | `avg_all_reduce_ms` | `training.steps[].all_reduce_ms` | `metrics_detailed.json` |
| 1.5 | FSDP All-Gather 时间 | `avg_all_gather_ms` | `training.steps[].all_gather_ms` | `metrics_detailed.json` |
| 1.6 | FSDP Reduce-Scatter 时间 | `avg_reduce_scatter_ms` | `training.steps[].reduce_scatter_ms` | `metrics_detailed.json` |
| 1.7 | DDP All-Reduce 流量 | `total_all_reduce_bytes` | `training.steps[].all_reduce_bytes` | `metrics_detailed.json` |
| 1.8 | FSDP All-Gather 流量 | `total_all_gather_bytes` | `training.steps[].all_gather_bytes` | `metrics_detailed.json` |
| 1.9 | FSDP Reduce-Scatter 流量 | `total_reduce_scatter_bytes` | `training.steps[].reduce_scatter_bytes` | `metrics_detailed.json` |
| 1.10 | 优化器更新时间 | `avg_optimizer_ms` | `training.steps[].optimizer_ms` | `metrics_detailed.json` |
| 1.11 | 单 step 时间 | `avg_step_ms` | `training.steps[].total_ms` | `metrics_detailed.json` |
| 1.12 | 总训练时间 | `total_train_time_s` | `training.total_train_time_s` | `metrics_detailed.json` |
| 1.13 | 吞吐量 (tokens/s) | `throughput_tokens_per_s` | `training.throughput_tokens_per_s` | `metrics_detailed.json` |
| 1.14 | 总 token 数 | `total_tokens` | `training.total_tokens` | `metrics_detailed.json` |
| 1.15 | FSDP full-state 导出时间 | `full_state_export_s` | `federated.full_state_export_s` | `metrics_detailed.json` |
| 1.16 | FSDP sharded-state 导出时间 | `sharded_state_export_s` | `federated.sharded_state_export_s` | `metrics_detailed.json` |
| 1.17 | 状态字典转换时间 | `state_dict_conversion_s` | `federated.state_dict_conversion_s` | `metrics_detailed.json` |
| 1.18 | 状态序列化保存时间 | `state_serialization_s` | `federated.state_serialization_s` | `metrics_detailed.json` |
| 1.19 | 状态字典字节数 | `state_bytes` | `federated.state_bytes` | `metrics_detailed.json` |
| 1.20 | checkpoint 保存时间 | `checkpoint_save_s` | `federated.checkpoint_save_s` | `metrics_detailed.json` |
| 1.21 | checkpoint 字节数 | `checkpoint_bytes` | `federated.checkpoint_bytes` | `metrics_detailed.json` |
| 1.22 | checkpoint 恢复时间 | `checkpoint_restore_s` | `federated.checkpoint_restore_s` | `metrics_detailed.json` |

**运行时补齐工具** (需要 torch，在训练侧调用):
- `measure_sharded_state_export(model, output_path)` — 计时 FSDP sharded-state 导出
- `measure_checkpoint_restore(model, checkpoint_path)` — 计时 checkpoint 恢复

---

### 类别 2: 集群内资源使用

**评估模块**: `evaluation/resource_usage.py`
**入口函数**: `aggregate_resource_usage(metrics_paths)`

| # | 评估点 | 输出字段 | 依赖数据 | 数据来源文件 |
|---|---|---|---|---|
| 2.1 | GPU 显存峰值 | `gpu_memory_peak_mb` | `resources.gpu_memory_peak_mb` | `metrics_detailed.json` |
| 2.2 | GPU 利用率均值 | `gpu_utilization_avg_pct` | `resources.gpu_utilization_avg_pct` | `metrics_detailed.json` |
| 2.3 | CPU 利用率均值 | `cpu_utilization_avg_pct` | `resources.cpu_utilization_avg_pct` | `metrics_detailed.json` |
| 2.4 | CPU 内存峰值 | `cpu_memory_peak_mb` | `resources.cpu_memory_peak_mb` | `metrics_detailed.json` |
| 2.5 | 网络接收流量 | `network_rx_bytes` | `resources.network_rx_bytes` | `metrics_detailed.json` |
| 2.6 | 网络发送流量 | `network_tx_bytes` | `resources.network_tx_bytes` | `metrics_detailed.json` |
| 2.7 | 网络总流量 | `network_total_bytes` | `resources.network_total_bytes` | `metrics_detailed.json` |
| 2.8 | NCCL 通信总字节数 | `total_nccl_bytes` | `training.total_nccl_bytes` | `metrics_detailed.json` |
| 2.9 | NCCL collective 调用次数 | `nccl_collective_calls` | `training.nccl_collective_calls` | `metrics_detailed.json` |
| 2.10 | 平均 NCCL 通信时间 | `avg_nccl_comm_ms` | `training.avg_nccl_comm_ms` | `metrics_detailed.json` |
| 2.11 | FSDP 状态导出开销 | `full_state_export_s`, `sharded_state_export_s` | `federated.full_state_export_s`, `federated.sharded_state_export_s` | `metrics_detailed.json` |
| 2.12 | checkpoint 保存开销 | `checkpoint_save_s`, `checkpoint_bytes` | `federated.checkpoint_save_s`, `federated.checkpoint_bytes` | `metrics_detailed.json` |
| 2.13 | checkpoint 恢复开销 | `checkpoint_restore_s` | `federated.checkpoint_restore_s` | `metrics_detailed.json` |

---

### 类别 3: 跨智算中心联邦更新

**评估模块**: `evaluation/federated_timing.py`
**入口函数**: `aggregate_federated_timing(results_dir)`

| # | 评估点 | 输出字段 | 依赖数据 | 数据来源文件 |
|---|---|---|---|---|
| 3.1 | 本地 model delta 导出时间 | `model_delta_export_s` | `federated.t_model_delta_export_s` 或 `aggregated_client_train_metrics.model_delta_export_seconds` | `metrics_detailed.json` 或 `experiment_summary.json` |
| 3.2 | 压缩编码时间 | `full_update_compression_s` | `federated.t_full_update_compression_s` 或 `aggregated_client_train_metrics.full_update_compression_seconds` | `metrics_detailed.json` 或 `experiment_summary.json` |
| 3.3 | WAN 下行传输时间 | `wan_download_s` | `federated.wan_download_s` 或 round 文件 `wan_download_s` | `metrics_detailed.json` 或 `federated_metrics_round_N.json` |
| 3.4 | WAN 上行传输时间 | `wan_upload_s` | `federated.wan_upload_s` 或 round 文件 `wan_upload_s` | `metrics_detailed.json` 或 `federated_metrics_round_N.json` |
| 3.5 | WAN 传输总时间 | `wan_transfer_total_s` | 派生: `wan_download_s + wan_upload_s` | 派生 |
| 3.6 | 明文 FedAvg 聚合时间 | `server_fedavg_aggregation_s` | round 文件 `server_fedavg_aggregation_s` | `federated_metrics_round_N.json` |
| 3.7 | 服务端后聚合处理时间 | `server_post_aggregation_s` | round 文件 `server_post_aggregation_s` | `federated_metrics_round_N.json` |
| 3.8 | checkpoint 保存时间 | `checkpoint_save_s` | round 文件 `checkpoint_save_s` | `federated_metrics_round_N.json` |
| 3.9 | 单次联邦轮总时间 | `federated_cycle_s` | round 文件 `federated_cycle_s` | `federated_metrics_round_N.json` |
| 3.10 | 客户端训练时间 | `client_training_s` | `aggregated_client_train_metrics.client_training_seconds` | `experiment_summary.json` |
| 3.11 | 客户端评估时间 | `client_evaluation_s` | `aggregated_client_train_metrics.client_evaluation_seconds` | `experiment_summary.json` |
| 3.12 | 客户端整轮时间 | `client_round_s` | `aggregated_client_train_metrics.client_round_seconds` | `experiment_summary.json` |
| 3.13 | model delta 字节数 | `model_delta_bytes` | round 文件 `model_delta_bytes` 或 `aggregated_client_train_metrics.model_delta_bytes` | `federated_metrics_round_N.json` 或 `experiment_summary.json` |
| 3.14 | 联邦总时间 | `total_federated_time_s` | 派生: Σ `federated_cycle_s` | 派生 |
| 3.15 | 平均每轮时间 | `avg_round_s` | 派生: `total_federated_time_s / num_rounds` | 派生 |

---

### 类别 4: 模型微调准确度

**评估模块**: `evaluation/validation_metrics.py`, `generation_metrics.py`, `baseline_comparison.py`, `mmlu_evaluator.py`

> 类别 4 的评估需要**运行模型推理**（不是纯数据聚合），因此在 GPU 环境中运行。

| # | 评估点 | 输出字段 | 依赖数据 | 获取方式 |
|---|---|---|---|---|
| 4.1 | 验证损失 (Loss) | `val_loss` | model + tokenizer + evalset | 运行 `compute_validation_metrics()` |
| 4.2 | 困惑度 (PPL) | `perplexity` | 同上 | 运行 `compute_validation_metrics()` |
| 4.3 | ROUGE-L | `rouge_l.f1` | model + tokenizer + evalset | 运行 `compute_generation_metrics()` |
| 4.4 | BERTScore | `bertscore.f1` | 同上 + `bert_score` 库 | 运行 `compute_generation_metrics()` |
| 4.5 | Token 重叠准确率 | `token_overlap_accuracy` | 同上 | 运行 `compute_generation_metrics()` |
| 4.6 | Macro-F1 | `macro_f1` | 同上 | 运行 `compute_generation_metrics()` |
| 4.7 | Exact Match | `exact_match` | 同上 | 运行 `compute_generation_metrics()` |
| 4.8 | 基线 delta | `delta.*` | federated 结果 + base 结果 | 运行 `evaluate_with_baseline()` |
| 4.9 | MMLU 准确率 | `accuracy`, `per_subject` | model + MMLU 数据集 | 运行 `evaluate_mmlu()` |
| 4.10 | DDP vs FSDP 对比 | 四类别对比表 | 类别 1-4 全部数据 | 运行 `generate_comparison_report()` |

**类别 4 的运行时数据依赖**:

| 数据 | 说明 | 来源 |
|---|---|---|
| model | 训练后的模型 (PEFT adapter 或 full state) | `MODEL_STATE_PATH` / `--adapter-path` |
| tokenizer | 基座模型 tokenizer | `AutoTokenizer.from_pretrained(base_model)` |
| evalset | 测试集 (Alpaca-GPT4 的 test split) | `data_split.create_train_val_test_split()` |
| base model | 未修改的基座模型 (基线对比用) | `AutoModelForCausalLM.from_pretrained(model_name)` |
| MMLU 数据集 | 多项选择题 | `datasets.load_dataset("cais/mmlu")` |

**类别 4 也可从已有结果文件聚合** (对比报告用):

| 评估点 | 读取的 JSON key | 来源文件 |
|---|---|---|
| val_loss / perplexity | `final_evaluation_metrics.assistant_only.loss/ppl` | `experiment_summary.json` |
| rouge_l_f1 | `final_evaluation_metrics.rouge_l.f1` | `experiment_summary.json` |
| bertscore_f1 | `final_evaluation_metrics.bertscore.f1` | `experiment_summary.json` |
| accuracy / macro_f1 / exact_match | `final_evaluation_metrics.generation_quality.*` | `experiment_summary.json` |
| val_loss / perplexity (备选) | `validation.val_loss/perplexity` | `evaluation.json` |
| rouge_l / bertscore (备选) | `downstream.rouge_l/bertscore_f1` | `evaluation.json` |

---

### 扩展实验

| # | 评估点 | 评估模块 | 依赖数据 | 获取方式 |
|---|---|---|---|---|
| E3.1 | 压缩字节数 | `compression_report.py` | model state dict | `compute_compression_report(state)` |
| E3.2 | 压缩比 | 同上 | 同上 | 同上 |
| E3.3 | INT8 量化误差 | 同上 | 同上 | 同上 |
| E3.4 | 掩码参数覆盖率 | 同上 | 同上 | 同上 |
| E4.1 | DP epsilon | `dp_report.py` | `num_rounds`, `clip_norm`, `sigma_client`, `cohort_size`, `delta` | `compute_rdp_epsilon()` (纯计算，无需运行时数据) |
| E4.2 | DP-utility 曲线 | 同上 | 上述参数 + `utility_fn` | `dp_utility_curve()` |

---

## 第二部分：数据清单

> 每个数据项列出：字段名、含义、在哪个环节、哪个文件、哪行代码记录。

### A. 训练过程数据（每步记录）

**记录代码**: `flowertune_llm/metrics.py` → `StepMetricsCallback`
**记录环节**: 训练 Job 中每个 optimizer step
**输出文件**: `metrics_detailed.json` 的 `training.steps[]` 数组

| # | 字段名 | 含义 | 记录方式 | 记录环节 |
|---|---|---|---|---|
| A.1 | `step` | 全局步号 | `state.global_step` | `on_step_end` |
| A.2 | `forward_ms` | 前向传播 + loss 计算时间 (ms) | `time.perf_counter()` 包裹 `compute_loss` | `record_forward()` (由 `distributed_trainer.py` 的 `timed_compute_loss` 调用) |
| A.3 | `backward_ms` | 反向传播时间 (ms) | 派生: `total_ms - forward_ms - comm_ms - optimizer_ms` | `on_step_end` |
| A.4 | `comm_ms` | NCCL 通信总时间 (ms) | monkey-patch `dist.all_reduce` / `all_gather` / `reduce_scatter`，`time.perf_counter()` 计时 | 每个 collective 调用时 `record_collective()` |
| A.5 | `nccl_bytes` | 本步 NCCL 通信字节数 | collective hook 中 `tensor.numel() * tensor.element_size()` | 同上 |
| A.6 | `nccl_collective_calls` | 本步 collective 调用次数 | `record_collective()` 递增 | 同上 |
| A.7 | `optimizer_ms` | 优化器更新时间 (ms) | `time.perf_counter()` 包裹 `on_pre_optimizer_step` → `on_optimizer_step` | `on_pre_optimizer_step` / `on_optimizer_step` |
| A.8 | `total_ms` | 单步总时间 (ms) | `time.perf_counter()` 从 `on_step_begin` 到 `on_step_end` | `on_step_end` |
| A.9 | `loss` | 当前步训练 loss | `state.log_history[-1]["loss"]` | `on_step_end` |
| A.10 | `lr` | 当前步学习率 | `state.log_history[-1]["learning_rate"]` | `on_step_end` |

**每步派生的 per-collective 字段** (由 `record_collective` 累积到 `_collective_totals_all`):

| # | 字段名 | 含义 |
|---|---|---|
| A.11 | `all_reduce_ms` / `all_reduce_bytes` | DDP All-Reduce 时间和流量 |
| A.12 | `all_gather_ms` / `all_gather_bytes` | FSDP All-Gather 时间和流量 |
| A.13 | `reduce_scatter_ms` / `reduce_scatter_bytes` | FSDP Reduce-Scatter 时间和流量 |
| A.14 | `all_gather_into_tensor_ms` / `_bytes` | All-Gather into tensor (FSDP 变体) |
| A.15 | `reduce_scatter_tensor_ms` / `_bytes` | Reduce-Scatter tensor (FSDP 变体) |

> **记录机制**: `StepMetricsCallback._hook_all_reduce()` 在 `on_train_begin` 时 monkey-patch `torch.distributed` 的 collective 函数，用 `time.perf_counter()` 计时并调用 `record_collective()`。DDP 额外注册 `model.register_comm_hook(state, ddp_comm_hook)` 捕获 C++ 层 All-Reduce。`on_train_end` 时恢复原始函数。

---

### B. 训练汇总数据（整轮训练结束记录）

**记录代码**: `flowertune_llm/metrics.py` → `StepMetricsCallback.get_summary()`
**记录环节**: 训练 Job 中 `trainer.train()` 返回后
**输出文件**: `metrics_detailed.json` 的 `training` 字段

| # | 字段名 | 含义 | 计算方式 |
|---|---|---|---|
| B.1 | `total_train_time_s` | 总训练时间 (s) | `time.perf_counter()` 从 `on_train_begin` 到 `on_train_end` |
| B.2 | `avg_step_time_ms` | 平均步时间 (ms) | Σ `total_ms` / num_steps |
| B.3 | `avg_forward_ms` | 平均前向时间 | Σ `forward_ms` / N |
| B.4 | `avg_backward_ms` | 平均反向时间 | Σ `backward_ms` / N |
| B.5 | `avg_comm_ms` | 平均通信时间 | Σ `comm_ms` / N |
| B.6 | `avg_optimizer_ms` | 平均优化器时间 | Σ `optimizer_ms` / N |
| B.7 | `avg_nccl_comm_ms` | 同 `avg_comm_ms` | 同上 |
| B.8 | `avg_nccl_bytes` | 平均每步 NCCL 流量 | Σ `nccl_bytes` / N |
| B.9 | `total_nccl_bytes` | NCCL 总流量 | Σ `nccl_bytes` |
| B.10 | `nccl_collective_calls` | 总 collective 调用数 | Σ `nccl_collective_calls` |
| B.11 | `throughput_tokens_per_s` | 吞吐量 (tokens/s) | `total_tokens / total_train_time_s` |
| B.12 | `total_tokens` | 总训练 token 数 | Σ `batch_size * gradient_accumulation_steps * seq_length` (每步累加) |
| B.13 | `avg_all_reduce_ms` | 平均 All-Reduce 时间 | `total_all_reduce_ms / N` |
| B.14 | `total_all_reduce_bytes` | All-Reduce 总流量 | 累积 |
| B.15 | `avg_all_gather_ms` | 平均 All-Gather 时间 | `total_all_gather_ms / N` |
| B.16 | `total_all_gather_bytes` | All-Gather 总流量 | 累积 |
| B.17 | `avg_reduce_scatter_ms` | 平均 Reduce-Scatter 时间 | `total_reduce_scatter_ms / N` |
| B.18 | `total_reduce_scatter_bytes` | Reduce-Scatter 总流量 | 累积 |

---

### C. 资源使用数据（后台采样）

**记录代码**: `flowertune_llm/metrics.py` → `ResourceMonitor`
**记录环节**: 训练 Job 中 `trainer.train()` 前后（后台线程每秒采样）
**输出文件**: `metrics_detailed.json` 的 `resources` 字段

| # | 字段名 | 含义 | 记录方式 | 记录环节 |
|---|---|---|---|---|
| C.1 | `gpu_memory_peak_mb` | GPU 显存峰值 (MB) | `max(torch.cuda.memory_allocated())` 采样 + `torch.cuda.max_memory_allocated()` | 训练期间后台线程每秒采样；`stop()` 时取 `max()` |
| C.2 | `gpu_utilization_avg_pct` | GPU 利用率均值 (%) | NVML `pynvml.nvmlDeviceGetUtilizationRates` 或 `nvidia-smi` 子进程 | 同上 |
| C.3 | `cpu_utilization_avg_pct` | CPU 利用率均值 (%) | `psutil.cpu_percent(interval=None)` | 同上 |
| C.4 | `cpu_memory_peak_mb` | CPU 内存峰值 (MB) | `max(psutil.Process().memory_info().rss)` | 同上 |
| C.5 | `network_rx_bytes` | 网络接收字节数 | `/proc/net/dev` 前后差值（排除 lo） | `start()` 时读基线，`stop()` 时读终值 |
| C.6 | `network_tx_bytes` | 网络发送字节数 | 同上 | 同上 |
| C.7 | `network_total_bytes` | 网络总流量 | `rx + tx` | 派生 |

---

### D. 状态导出与 checkpoint 数据

**记录代码**: `flowertune_llm/distributed_trainer.py`
**记录环节**: 训练 Job 中 `trainer.train()` 返回后、状态导出和序列化阶段
**输出文件**: `metrics_detailed.json` 的 `federated` 字段 + `metrics.json`

| # | 字段名 | 含义 | 记录方式 | 记录环节 |
|---|---|---|---|---|
| D.1 | `state_export_type` | 状态导出类型 | 常量: `"full_state_dict"` (FSDP) 或 `"replicated_full_state_dict"` (DDP) | 训练后状态导出阶段开始时 |
| D.2 | `full_state_export_s` | FSDP full-state 导出时间 (s) | `time.perf_counter()` 包裹 `FSDP.state_dict_type(FULL_STATE_DICT)` + `model.state_dict()` | FSDP 路径: 训练后 full-state gather |
| D.3 | `state_dict_conversion_s` | 联邦状态字典转换时间 (s) | `time.perf_counter()` 包裹 `get_federated_state_dict()` | 状态导出后 |
| D.4 | `state_serialization_s` | 状态序列化保存时间 (s) | `time.perf_counter()` 包裹 `torch.save()` | `model_weights.pt` 写入 |
| D.5 | `state_bytes` | 状态字典文件字节数 | `os.path.getsize(weights_path)` | `torch.save` 后 |
| D.6 | `checkpoint_restore_s` | checkpoint 恢复时间 (s) | `time.perf_counter()` 包裹 `torch.load()` + `set_federated_state_dict()` | **训练开始前**加载初始权重时 |

> D.6 是新增字段。当 `model_weights_path` 存在时，在 `distributed_trainer.py` 加载初始权重处计时。

---

### E. 客户端联邦时序数据

**记录代码**: `flowertune_llm/client_app.py`
**记录环节**: ClientApp `train()` 函数中，从接收全局模型到返回训练结果
**输出文件**: `metrics_detailed.json` 的 `federated` 字段（回写增强）+ 返回给 Server 的 `MetricRecord`

| # | 字段名 | 含义 | 记录方式 | 记录环节 |
|---|---|---|---|---|
| E.1 | `t_model_delta_export_s` | model delta 导出时间 (s) | `time.perf_counter()` 包裹 `collect_training_results()` (加载 weights) | K8s Job 完成后收集结果时 |
| E.2 | `t_collect_results_s` | 结果收集时间 (s) | 同上 | 同上 |
| E.3 | `t_full_update_compression_s` | 压缩编码时间 (s) | `time.perf_counter()` 包裹 `encode_topk_int8_delta()` 或 `encode_fedscale_int8_delta()` | 压缩编码阶段 |
| E.4 | `t_total_round_s` | 客户端整轮时间 (s) | `time.perf_counter()` 从 `train()` 开始到结束 | `train()` 函数全程 |
| E.5 | `wan_download_s` | WAN 下行传输时间 (s) | `time.perf_counter()` 包裹对象存储下载或 Flower RPC 反序列化 | `train()` 开始时接收全局模型 |
| E.6 | `wan_upload_s` | WAN 上行传输时间 (s) | `time.perf_counter()` 包裹对象存储上传 | `train()` 结束时上传训练结果 |
| E.7 | `model_delta_bytes` | model delta 字节数 | Σ `value.numel() * value.element_size()` | 压缩编码后 |
| E.8 | `object_store_uploaded_bytes` | 对象存储上传字节数 | `artifact.size` 或 `os.path.getsize()` | 上传完成后 |

> E.5 和 E.6 是新增字段。E.5 在对象存储下载或 Flower RPC `to_torch_state_dict()` 处计时；E.6 在对象存储上传处计时（Flower RPC 路径下为 0）。

**返回给 Server 的 `MetricRecord` 字段** (经 Flower 聚合后进入 `experiment_summary.json`):

| # | 字段名 | 含义 | 对应 federated_metrics 字段 |
|---|---|---|---|
| E.9 | `client_training_seconds` | 客户端训练时间 | 来自 `metrics.json` 的 `training_only_s` |
| E.10 | `client_evaluation_seconds` | 客户端评估时间 | 来自 `metrics.json` 的 `evaluation_s` |
| E.11 | `client_round_seconds` | 客户端整轮时间 | E.4 `t_total_round_s` |
| E.12 | `client_non_training_seconds` | 客户端非训练时间 | `t_total_round_s - training_only_s` |
| E.13 | `full_update_compression_seconds` | 压缩编码时间 | E.3 |
| E.14 | `wan_download_seconds` | WAN 下行时间 | E.5 |
| E.15 | `wan_upload_seconds` | WAN 上行时间 | E.6 |
| E.16 | `model_delta_export_seconds` | delta 导出时间 | E.1 |
| E.17 | `model_delta_bytes` | delta 字节数 | E.7 |
| E.18 | `object_store_uploaded_bytes` | 上传字节数 | E.8 |

---

### F. 服务端联邦时序数据

**记录代码**: `flowertune_llm/server_app.py`
**记录环节**: ServerApp 每轮聚合后和实验结束时
**输出文件**: `federated_metrics_round_N.json`, `federated_timings.json`, `experiment_summary.json`

| # | 字段名 | 含义 | 记录方式 | 记录环节 | 输出文件 |
|---|---|---|---|---|---|
| F.1 | `server_fedavg_aggregation_s` | FedAvg 聚合时间 (s) | `time.perf_counter()` 包裹 `aggregate_train()` | `TimedFedAvg` / `FedScaleBlockFedAvg.aggregate_train()` 的 `finally` 块 | `federated_metrics_round_N.json` |
| F.2 | `federated_cycle_s` | 单次联邦轮总时间 (s) | 上一次 `evaluate()` 回调结束到本次回调开始 | `get_evaluate_fn().evaluate()` | `federated_metrics_round_N.json` + `federated_timings.json` |
| F.3 | `server_post_aggregation_s` | 服务端后聚合时间 (s) | `time.perf_counter()` 从 `evaluate()` 开始到结束 | `get_evaluate_fn().evaluate()` | `federated_metrics_round_N.json` |
| F.4 | `checkpoint_save_s` | 服务端 checkpoint 保存时间 (s) | `time.perf_counter()` 包裹 `torch.save()` / `_write_lora_checkpoint()` | `get_evaluate_fn().evaluate()` 中 checkpoint 保存 | `federated_metrics_round_N.json` |
| F.5 | `checkpoint_bytes` | checkpoint 字节数 | `os.path.getsize()` 或 `os.walk()` | 同上 | `federated_metrics_round_N.json` |
| F.6 | `final_model_evaluation_s` | 最终模型评估时间 (s) | `time.perf_counter()` 从评估开始到结束 | 最终轮 `evaluate()` 回调 | `experiment_summary.json` |
| F.7 | `duration_seconds` | 实验总时长 (s) | `time.perf_counter()` 从 `main()` 开始到训练完成 | `main()` 的 `else` 块 | `experiment_summary.json` |
| F.8 | `initial_state_load_s` | 初始状态加载时间 (s) | `time.perf_counter()` | `main()` 中策略初始化 | `experiment_summary.json` |
| F.9 | `server_base_state_load_s` | 服务端基座模型加载时间 (s) | `time.perf_counter()` | `main()` 中 `_load_full_base_state()` | `experiment_summary.json` |
| F.10 | `aggregated_client_train_metrics` | 客户端训练指标聚合 | Flower `train_metrics_aggr_fn` | 每轮聚合后 | `experiment_summary.json` |
| F.11 | `aggregated_client_evaluate_metrics` | 客户端评估指标聚合 | Flower evaluate 回调 | 最终轮 | `experiment_summary.json` |

---

### G. 最终模型评估数据

**记录代码**: `flowertune_llm/server_app.py` → `_write_final_evaluation_summary()`
**记录环节**: 实验完成后，客户端最终评估结果回传到服务端
**输出文件**: `evaluation_summary.json` + `experiment_summary.json` 的 `final_evaluation_metrics`

| # | 字段名 | 含义 | 记录方式 | 记录环节 |
|---|---|---|---|---|
| G.1 | `assistant_only.loss` | 最终模型验证损失 | 客户端 `evaluate()` 回传的 `val_loss` | 最终轮 ClientApp evaluate Job |
| G.2 | `assistant_only.ppl` | 最终模型 PPL | 客户端回传的 `perplexity` | 同上 |
| G.3 | `rouge_l.f1` | ROUGE-L F1 | 客户端回传的 `rouge_l` | 同上 |
| G.4 | `bertscore.f1` | BERTScore F1 | 客户端回传的 `bertscore_f1` | 同上 |
| G.5 | `generation_quality.accuracy` | Token 重叠准确率 | 客户端回传的 `accuracy` | 同上 |
| G.6 | `generation_quality.macro_f1` | Macro-F1 | 客户端回传的 `macro_f1` | 同上 |
| G.7 | `generation_quality.exact_match` | Exact Match | 客户端回传的 `exact_match` | 同上 |
| G.8 | `evaluation_seconds` | 评估耗时 | `time.perf_counter()` | `evaluate()` 回调 |

> G.1-G.7 的数据流: ClientApp `evaluate()` → K8s Job (`final_evaluator.py`) → `evaluation.json` → ClientApp 读取并包装为 `MetricRecord` → ServerApp `_write_final_evaluation_summary()` 写入 `evaluation_summary.json`。

---

### H. 运行时数据文件总览

| 文件 | 路径 | 写入者 | 写入环节 | 包含的数据项 |
|---|---|---|---|---|
| `metrics_detailed.json` | `{output_dir}/{job_name}/` | `distributed_trainer.py` (rank 0) | 训练 Job 结束时 | A (全部 steps), B (汇总), C (资源), D (状态导出) |
| | | `client_app.py` (回写增强) | K8s Job 完成后 | + E.1-E.8 (客户端联邦时序) |
| `metrics.json` | `{output_dir}/{job_name}/` | `distributed_trainer.py` (rank 0) | 训练 Job 结束时 | B (摘要), C (网络), D (状态导出摘要) |
| `federated_metrics_round_N.json` | `/app/results/{experiment_id}/` | `server_app.py` | 每轮聚合后 | F.1-F.5 |
| `federated_timings.json` | `/app/results/{experiment_id}/` | `server_app.py` | 每轮聚合后 (覆盖写) | F.2 (累积列表) |
| `experiment_summary.json` | `/app/results/{experiment_id}/` | `server_app.py` | 实验结束时 | F.6-F.11, G.1-G.8 |
| `evaluation_summary.json` | `/app/results/{experiment_id}/` | `server_app.py` | 实验完成后 | G.1-G.8 |
| `evaluation.json` | `/app/outputs/{eval_job}/` | `final_evaluator.py` | 评估 Job 结束时 | G.1-G.8 (客户端侧) |
| `summary.json` | CLI 输出目录 | `run_evaluation.py` | CLI 评估完成后 | G.1-G.8 (CLI 侧) + delta |

---

### I. 数据缺口

以下数据**当前未记录**，需要在未来补齐：

| # | 缺口数据 | 应在哪个环节记录 | 应在哪个文件记录 | 影响 |
|---|---|---|---|---|
| I.1 | `sharded_state_export_s` | FSDP sharded-state 导出时 | `metrics_detailed.json` → `federated.sharded_state_export_s` | 类别 1.16 缺失（已有 `measure_sharded_state_export()` 工具函数，但训练代码未调用） |
| I.2 | WAN 传输字节数（独立于 model_delta_bytes） | 客户端下载/上传时 | `metrics_detailed.json` → `federated.wan_rx_bytes` / `wan_tx_bytes` | 类别 3 的 WAN 流量缺少独立字节数（当前只有 `model_delta_bytes`） |

> I.1 的工具函数 `measure_sharded_state_export()` 已在 `evaluation/training_performance.py` 中实现，但需要在 `distributed_trainer.py` 的状态导出阶段调用。I.2 的网络字节数目前只有容器级 `/proc/net/dev` 统计 (C.5-C.7)，缺少 WAN 级别（跨中心）的独立统计。
