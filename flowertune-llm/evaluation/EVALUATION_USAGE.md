# 评估运行使用说明

## 概述

评估分为两大块：

1. **运行时数据采集** — 在训练业务代码中自动完成，不需要额外操作。训练过程中自动记录训练性能、资源使用、联邦时序等数据到 JSON 文件。
2. **评估分析** — 基于 `evaluation/` 包，读取运行时记录的数据进行聚合分析和报告生成。模型准确度评估需要额外运行 GPU 推理。

```
训练 Job                          评估 (evaluation/ 包)
┌─────────────────────┐           ┌──────────────────────────┐
│ metrics.py          │           │ training_performance.py  │ ← 类别 1: 训练性能
│  · StepMetricsCallback           │ resource_usage.py        │ ← 类别 2: 资源使用
│  · ResourceMonitor  │           │ federated_timing.py      │ ← 类别 3: 联邦时序
│ distributed_trainer │           │                          │
│  · checkpoint_restore_s          │ validation_metrics.py    │ ← 类别 4: 模型准确度
│ client_app.py       │           │ generation_metrics.py    │
│  · wan_download_s    │           │ baseline_comparison.py   │
│  · wan_upload_s     │           │ mmlu_evaluator.py        │
│ server_app.py       │           │ compression_report.py    │ ← E3: 压缩报告
│  · aggregation_timings           │ dp_report.py             │ ← E4: DP 报告
│  · round_timings    │           │ comparison_report.py     │ ← DDP vs FSDP 对比
└────────┬────────────┘           │ generate_report.py       │ ← 统一报告 CLI
         │                        └──────────────────────────┘
         ▼ 写入 JSON 文件                      ▲ 读取 JSON 文件
   ┌─────────────────────────────────────────────┐
   │ metrics_detailed.json                       │
   │ metrics.json                                │
   │ federated_metrics_round_N.json              │
   │ federated_timings.json                      │
   │ experiment_summary.json                     │
   │ evaluation_summary.json / evaluation.json   │
   └─────────────────────────────────────────────┘
```

---

## 一、运行时数据采集（自动，无需操作）

训练过程中以下文件自动生成，评估代码依赖这些文件。

### 1.1 训练 Job 产出文件

训练 Job 结束后，在 `{output_dir}/{job_name}/` 目录下生成：

| 文件 | 内容 | 生成条件 |
|---|---|---|
| `metrics_detailed.json` | 每步训练时间、资源使用、状态导出、联邦时序 | 训练 Job 完成 |
| `metrics.json` | 训练摘要（兼容旧格式） | 训练 Job 完成 |
| `model_weights.pt` | 训练后的模型权重 | 训练 Job 完成 |

### 1.2 ServerApp 产出文件

联邦实验运行过程中，在 `/app/results/{experiment_id}/` 目录下生成：

| 文件 | 内容 | 生成条件 |
|---|---|---|
| `federated_metrics_round_N.json` | 第 N 轮的服务端聚合时间、联邦周期时间、checkpoint 时间 | 每轮聚合后 |
| `federated_timings.json` | 累积的各轮联邦周期时间数组 | 每轮聚合后（覆盖写） |
| `experiment_summary.json` | 实验总览：各轮时间、客户端指标聚合、最终评估结果 | 实验结束时 |
| `evaluation_summary.json` | 最终模型评估结果（loss/ppl/rouge/bertscore 等） | 最终评估完成后 |

### 1.3 无需额外配置

上述文件全部由训练代码自动生成，不需要手动触发或配置。训练代码中只新增了 3 个计时点（不影响训练逻辑）：

- `distributed_trainer.py`: 加载初始权重时记录 `checkpoint_restore_s`
- `client_app.py`: 接收全局模型时记录 `wan_download_s`
- `client_app.py`: 上传训练结果时记录 `wan_upload_s`

---

## 二、评估运行方式

### 方式 1: 联邦运行时自动触发（K8s Job）

**场景**: 联邦训练结束后，自动在 GPU 上运行模型准确度评估。

**触发方**: Flower ClientApp 的 `evaluate()` handler，自动启动 K8s Job。

**入口**: `evaluation/final_evaluator.py`

**环境变量配置**:

| 环境变量 | 必需 | 说明 | 默认值 |
|---|---|---|---|
| `MODEL_NAME` | ✅ | 基座模型名（如 `Qwen/Qwen3-14B`） | — |
| `MODEL_STATE_PATH` | ✅ | 联邦模型状态文件路径 | — |
| `EVALUATION_OUTPUT_PATH` | ✅ | 评估结果输出 JSON 路径 | — |
| `PARTITION_ID` | ✅ | 联邦分区 ID | — |
| `NUM_PARTITIONS` | ✅ | 联邦分区总数 | — |
| `DATASET_NAME` | ✅ | HuggingFace 数据集名 | — |
| `FINETUNING_TYPE` | | 微调类型 (`lora` / `full`) | `lora` |
| `QUANTIZATION` | | 量化级别 (0/4/8) | `4` |
| `LORA_R` / `LORA_ALPHA` | | LoRA 参数 | `32` / `64` |
| `EVAL_VAL_RATIO` / `EVAL_TEST_RATIO` | | 验证/测试集比例 | `0.05` / `0.05` |
| `NUM_EVAL_SAMPLES` | | 评估样本数 | `50` |
| `EVAL_MAX_LENGTH` | | 最大 tokenization 长度 | `512` |
| `EVAL_MAX_NEW_TOKENS` | | 最大生成 token 数 | `128` |
| `EVAL_SKIP_BASE` | | 跳过基座模型基线评估 | `false` |
| `EVAL_MMLU_DATASET` | | MMLU 数据集名（空=不评估） | `""` |
| `EVAL_MMLU_SAMPLES` | | MMLU 评估题数 | `100` |
| `EVAL_COMPRESSION_REPORT` | | 生成压缩报告 | `false` |
| `FEDSCALE_BLOCK_SIZE` | | 压缩规范块大小 | `1048576` |
| `EVAL_DP_SIGMA_CLIENT` | | DP σ 值（空=不生成 DP 报告） | `""` |
| `EVAL_DP_NUM_ROUNDS` | | DP 联邦轮数 | `50` |
| `EVAL_DP_CLIP_NORM` | | DP 裁剪阈值 | `1.0` |
| `EVAL_DP_COHORT_SIZE` | | DP 参与者数 | `2` |
| `EVAL_DP_DELTA` | | DP δ 值 | `1e-5` |

**输出**: 写入 `EVALUATION_OUTPUT_PATH` 指定的 JSON 文件，包含：

```json
{
  "status": "completed",
  "federated": { "validation": {...}, "generation": {...} },
  "base": { "validation": {...}, "generation": {...} },
  "delta": { "val_loss": ..., "rouge_l_f1": ..., ... },
  "mmlu": { "accuracy": ..., "per_subject": {...} },
  "compression": [ { "config_name": ..., "compression_ratio": ..., ... } ],
  "dp": { "epsilon": ..., "alpha_optimal": ..., ... }
}
```

**说明**: 此方式由 `client_app.py` 的 `@app.evaluate()` 自动触发，用户无需手动操作。ClientApp 会将全局模型状态保存到 PVC，启动一个单 GPU K8s Job 运行 `final_evaluator.py`，完成后读取结果并回传给 ServerApp。

---

### 方式 2: 独立 CLI 模型评估

**场景**: 在任何有 GPU 的机器上，独立评估一个训练好的 checkpoint。

**入口**: `python -m evaluation.run_evaluation`

**前提条件**:
- CUDA GPU 可用
- `torch`、`transformers`、`peft`、`datasets` 已安装
- 基座模型已在本地缓存 (`$HF_HOME`)
- Alpaca-GPT4 数据集已缓存（或允许联网下载）

#### 基本用法

```bash
# 评估 PEFT adapter
python -m evaluation.run_evaluation \
    --adapter-path /path/to/checkpoint \
    --base-model Qwen/Qwen3-14B \
    --output-dir ./eval_results \
    --max-samples 100

# 评估完整模型 state dict
python -m evaluation.run_evaluation \
    --full-state-path /path/to/model_state.pt \
    --base-model Qwen/Qwen3-14B \
    --output-dir ./eval_results
```

#### 完整评估（基线对比 + MMLU + 压缩报告 + DP 报告）

```bash
python -m evaluation.run_evaluation \
    --adapter-path /path/to/checkpoint \
    --base-model Qwen/Qwen3-14B \
    --dataset vicgalle/alpaca-gpt4 \
    --output-dir ./eval_results \
    --max-samples 100 \
    --max-length 512 \
    --max-new-tokens 128 \
    --with-baseline \
    --mmlu-dataset cais/mmlu \
    --mmlu-samples 200 \
    --compression-report \
    --dp-sigma-client 0.5 \
    --dp-num-rounds 50 \
    --dp-clip-norm 1.0 \
    --dp-cohort-size 2
```

#### CLI 参数一览

| 参数 | 必需 | 说明 | 默认值 |
|---|---|---|---|
| `--adapter-path` | ✅* | PEFT adapter checkpoint 目录 | — |
| `--full-state-path` | ✅* | 完整模型 state_dict 文件 | — |
| `--base-model` | | 基座模型名 | `openlm-research/open_llama_3b_v2` |
| `--label` | | 结果文件名标签 | `federated_peft` 或 `full_checkpoint` |
| `--quantization` | | 量化级别 (0/4/8) | `0` |
| `--lora-r` / `--lora-alpha` | | LoRA 参数 | `32` / `64` |
| `--dataset` | | HuggingFace 数据集名 | `vicgalle/alpaca-gpt4` |
| `--num-partitions` | | 联邦分区数 | `2` |
| `--val-ratio` / `--test-ratio` | | 验证/测试集比例 | `0.05` / `0.05` |
| `--split-seed` | | 分割随机种子 | `42` |
| `--max-samples` | | 评估样本数 | `100` |
| `--max-length` | | 最大 tokenization 长度 | `512` |
| `--max-new-tokens` | | 最大生成 token 数 | `128` |
| `--bertscore-model` | | BERTScore backbone 模型 | `None` |
| `--with-baseline` | | 启用基座模型基线对比 | `False` |
| `--skip-base` | | 跳过基线（覆盖 `--with-baseline`） | `False` |
| `--mmlu-dataset` | | MMLU 数据集名 | `None` |
| `--mmlu-samples` | | MMLU 评估题数 | `100` |
| `--compression-report` | | 生成压缩报告 | `False` |
| `--block-size` | | 压缩规范块大小 | `1048576` |
| `--dp-sigma-client` | | DP σ 值 | `None` |
| `--dp-num-rounds` | | DP 联邦轮数 | `50` |
| `--dp-clip-norm` | | DP 裁剪阈值 | `1.0` |
| `--dp-cohort-size` | | DP 参与者数 | `2` |
| `--dp-delta` | | DP δ 值 | `1e-5` |
| `--output-dir` | ✅ | 输出目录 | — |
| `--local-files-only` | | 仅使用本地缓存 | `False` |

> *`--adapter-path` 和 `--full-state-path` 二选一，互斥。

#### 输出文件

| 文件 | 内容 |
|---|---|
| `{output-dir}/summary.json` | 完整评估结果（含 federated/base/delta/mmlu/compression/dp） |
| `{output-dir}/predictions_{label}.jsonl` | 每个样本的生成预测（prompt + reference + prediction） |
| `{output-dir}/predictions_base.jsonl` | 基座模型的生成预测（如启用基线） |

---

### 方式 3: 统一报告生成 CLI

**场景**: 训练实验已完成，从产出目录生成四类别统一报告。不需要 GPU。

**入口**: `python -m evaluation.generate_report`

**前提条件**:
- 实验结果目录中存在 `metrics_detailed.json`、`federated_metrics_round_N.json`、`experiment_summary.json` 等产出文件
- Python 环境（不需要 torch/transformers）

#### 单次实验报告

```bash
python -m evaluation.generate_report \
    --results-dir /app/results/experiment-001 \
    --output report.json
```

输出包含：
- `training_performance`: 类别 1（前向/反向/通信/优化器时间、吞吐量、状态导出等）
- `resource_usage`: 类别 2（GPU/CPU/网络/NCCL）
- `federated_timing`: 类别 3（每轮 WAN 传输、FedAvg 聚合、联邦周期时间）
- `model_accuracy`: 类别 4（从 `experiment_summary.json` 或 `evaluation.json` 读取）

#### DDP vs FSDP 对比报告

```bash
# JSON + Markdown 双格式
python -m evaluation.generate_report \
    --ddp-dir /app/results/ddp-experiment \
    --fsdp-dir /app/results/fsdp-experiment \
    --output-dir ./reports \
    --format both
```

输出文件：
- `comparison_report.json` — 结构化对比数据
- `comparison_report.md` — Markdown 格式的四类别对比表

#### 仅 Markdown 对比报告

```bash
python -m evaluation.generate_report \
    --ddp-dir /app/results/ddp-experiment \
    --fsdp-dir /app/results/fsdp-experiment \
    --output comparison_report.md \
    --format markdown
```

#### CLI 参数一览

| 参数 | 必需 | 说明 | 默认值 |
|---|---|---|---|
| `--results-dir` | ✅* | 单次实验结果目录 | — |
| `--ddp-dir` | ✅* | DDP 实验结果目录（对比模式） | — |
| `--fsdp-dir` | * | FSDP 实验结果目录（对比模式） | — |
| `--output` | | 输出文件路径 | `report.json` 或 `comparison_report.json` |
| `--output-dir` | | 输出目录（`--format both` 时使用） | — |
| `--format` | | 输出格式: `json` / `markdown` / `both` | `json` |

> *`--results-dir` 和 `--ddp-dir` 互斥。使用 `--ddp-dir` 时必须同时提供 `--fsdp-dir`。

---

### 方式 4: Python API 直接调用

**场景**: 在脚本或 Notebook 中直接调用评估函数。

#### 4.1 轻量聚合（不需要 GPU）

```python
from evaluation import (
    aggregate_training_performance,
    aggregate_resource_usage,
    aggregate_federated_timing,
    generate_comparison_report,
    format_comparison_report,
)

# 类别 1: 训练性能
tp = aggregate_training_performance("/path/to/metrics_detailed.json")
print(f"吞吐量: {tp.throughput_tokens_per_s} tokens/s")
print(f"平均步时间: {tp.avg_step_ms} ms")

# 类别 2: 资源使用
ru = aggregate_resource_usage("/path/to/metrics_detailed.json")
print(f"GPU 显存峰值: {ru.gpu_memory_peak_mb} MB")
print(f"GPU 利用率: {ru.gpu_utilization_avg_pct}%")

# 类别 3: 联邦时序
ft = aggregate_federated_timing("/path/to/results_dir")
for r in ft.rounds:
    print(f"Round {r.round}: cycle={r.federated_cycle_s}s, "
          f"WAN={r.wan_transfer_total_s}s, "
          f"FedAvg={r.server_fedavg_aggregation_s}s")

# 类别 4e: DDP vs FSDP 对比
report = generate_comparison_report(
    ddp_results_dir="/app/results/ddp-experiment",
    fsdp_results_dir="/app/results/fsdp-experiment",
)
print(format_comparison_report(report))
```

#### 4.2 DP 隐私预算计算（纯计算，不需要 GPU）

```python
from evaluation import compute_rdp_epsilon, dp_utility_curve, format_dp_report

# 计算单次实验的 DP 保证
rdp = compute_rdp_epsilon(
    num_rounds=50,
    clip_norm=1.0,
    sigma_client=0.5,
    cohort_size=2,
    delta=1e-5,
)
print(f"ε = {rdp.epsilon}, α* = {rdp.alpha_optimal}")

# 生成 DP-utility 曲线
curve = dp_utility_curve(
    sigma_values=[0, 0.1, 0.5, 1.0, 2.0],
    utility_fn=lambda sigma: {"val_loss": 1.5 + sigma * 0.3},  # 替换为实际评估函数
    num_rounds=50,
    clip_norm=1.0,
    cohort_size=2,
)
print(format_dp_report(curve))
```

#### 4.3 压缩报告（需要 model state dict）

```python
from evaluation import compute_compression_report

# state 是 CPU 上的模型 state dict
results = compute_compression_report(state)
for r in results:
    print(f"{r.config_name}: {r.compression_ratio}x, "
          f"误差={r.quantization_error_l2}")
```

#### 4.4 模型准确度评估（需要 GPU + model）

```python
from evaluation import (
    compute_validation_metrics,
    compute_generation_metrics,
    evaluate_with_baseline,
    evaluate_mmlu,
)

# 路径 A: 验证损失 / PPL
val = compute_validation_metrics(model, tokenizer, evalset, "cuda:0", max_samples=100)
print(f"Loss: {val['val_loss']}, PPL: {val['perplexity']}")

# 路径 B: 生成指标
gen = compute_generation_metrics(model, tokenizer, evalset, "cuda:0", max_samples=100)
print(f"ROUGE-L: {gen['rouge_l']['f1']}, BERTScore: {gen['bertscore']['f1']}")

# 基线对比
results = evaluate_with_baseline(
    model_name="Qwen/Qwen3-14B",
    federated_model=model,
    tokenizer=tokenizer,
    evalset=evalset,
    device="cuda:0",
    max_samples=100,
)
print(f"Delta: {results['delta']}")

# MMLU
mmlu = evaluate_mmlu(model, tokenizer, "cuda:0", dataset_name="cais/mmlu", max_samples=200)
print(f"MMLU accuracy: {mmlu['accuracy']}")
```

#### 4.5 数据划分（轻量，不需要 GPU）

```python
from evaluation import stable_sample_key, create_train_val_test_split

# 单样本哈希
key = stable_sample_key({"instruction": "Translate", "input": "hello"})

# 三方划分
split = create_train_val_test_split(dataset, val_ratio=0.05, test_ratio=0.05)
print(f"Train: {len(split.train)}, Val: {len(split.validation)}, Test: {len(split.test)}")
```

---

## 三、典型工作流

### 工作流 A: 联邦训练 + 自动评估

```
1. 提交联邦训练实验
   ↓ ServerApp 自动调度多轮训练
2. 每轮训练自动产出:
   - metrics_detailed.json (训练性能 + 资源)
   - federated_metrics_round_N.json (联邦时序)
   ↓ 训练完成后
3. ClientApp 自动触发最终评估 K8s Job
   ↓ final_evaluator.py 运行
4. 产出 evaluation.json (模型准确度)
   ↓ ServerApp 汇总
5. 产出 experiment_summary.json + evaluation_summary.json
   ↓
6. 生成报告:
   python -m evaluation.generate_report \
       --results-dir /app/results/experiment-001 \
       --output report.json
```

### 工作流 B: DDP vs FSDP 对比实验

```
1. 分别运行 DDP 和 FSDP 实验
   - DDP:  配置 train.distributed-strategy=ddp
   - FSDP: 配置 train.distributed-strategy=fsdp
   ↓ 两个实验各自完成
2. 生成对比报告:
   python -m evaluation.generate_report \
       --ddp-dir /app/results/ddp-experiment \
       --fsdp-dir /app/results/fsdp-experiment \
       --output-dir ./reports \
       --format both
   ↓
3. 查看:
   - reports/comparison_report.json (结构化数据)
   - reports/comparison_report.md (Markdown 对比表)
```

### 工作流 C: 独立 checkpoint 评估

```
1. 获取训练好的 checkpoint (adapter 或 full state)
   ↓
2. 运行独立评估:
   python -m evaluation.run_evaluation \
       --adapter-path /path/to/adapter \
       --base-model Qwen/Qwen3-14B \
       --output-dir ./eval_results \
       --with-baseline \
       --mmlu-dataset cais/mmlu \
       --compression-report
   ↓
3. 查看:
   - eval_results/summary.json (完整结果)
   - eval_results/predictions_federated_peft.jsonl (逐样本预测)
```

---

## 四、依赖说明

### 轻量模块（不需要 GPU / torch / transformers）

| 模块 | 用途 | 依赖 |
|---|---|---|
| `data_split.py` | 三方数据划分 | `hashlib` (标准库) |
| `dp_report.py` | DP 隐私预算 | `math` (标准库) |
| `compression_report.py` | 压缩报告接口 | `math` (标准库) |
| `training_performance.py` | 训练性能聚合 | `json` (标准库) |
| `resource_usage.py` | 资源使用聚合 | `json` (标准库) |
| `federated_timing.py` | 联邦时序聚合 | `json` (标准库) |
| `comparison_report.py` | DDP vs FSDP 对比 | `json` (标准库) |
| `generate_report.py` | 报告生成 CLI | `json` (标准库) |

### 重型模块（需要 GPU + torch + transformers）

| 模块 | 用途 | 额外依赖 |
|---|---|---|
| `validation_metrics.py` | Loss / PPL | `torch` |
| `generation_metrics.py` | ROUGE-L / BERTScore / Acc / F1 / EM | `torch`, `rouge_score`(可选), `bert_score`(可选) |
| `baseline_comparison.py` | 基线对比 | `torch`, `transformers`, `peft` |
| `mmlu_evaluator.py` | MMLU / MMLU-Pro | `torch`, `datasets` |
| `final_evaluator.py` | K8s 评估入口 | `torch`, `transformers`, `peft`, `omegaconf` |
| `run_evaluation.py` | 独立评估 CLI | `torch`, `transformers`, `peft`, `datasets` |

### 可选依赖

| 库 | 用途 | 缺失时的行为 |
|---|---|---|
| `rouge_score` | ROUGE-L 计算 | 回退到内置 LCS 实现 |
| `bert_score` | BERTScore 计算 | 返回 `None`（不影响其他指标） |
| `psutil` | CPU 利用率/内存 | 训练侧资源监控不可用 |
| `pynvml` | GPU 利用率 | 回退到 `nvidia-smi` 子进程 |

---

## 五、输出文件格式

### 5.1 `summary.json` (run_evaluation.py 输出)

```json
{
  "evaluated_at": "2025-09-11T12:00:00Z",
  "metadata": {
    "base_model": "Qwen/Qwen3-14B",
    "adapter_path": "/path/to/adapter",
    "dataset": "vicgalle/alpaca-gpt4",
    "max_length": 512,
    "max_new_tokens": 128,
    "do_sample": false,
    "num_beams": 1
  },
  "results": {
    "federated": {
      "validation": { "val_loss": 1.23, "perplexity": 3.42, ... },
      "generation": { "rouge_l": {"f1": 0.45}, "bertscore": {"f1": 0.87}, ... },
      "evaluation_seconds": 120.5
    },
    "base": { ... },
    "delta": { "val_loss": -0.15, "rouge_l_f1": 0.08, ... },
    "mmlu": { "accuracy": 0.52, "per_subject": { ... } },
    "compression": [ { "config_name": "int8_only", "compression_ratio": 3.8, ... } ],
    "dp": { "epsilon": 4.2, "alpha_optimal": 8, ... }
  }
}
```

### 5.2 `comparison_report.md` (generate_report.py 输出)

```markdown
# DDP vs FSDP Comparison Report

## 1. Training Performance

| Metric | DDP | FSDP |
|---|---|---|
| Avg step (ms) | 850 | 920 |
| Avg All-Reduce (ms) | 120 | 0 |
| Avg All-Gather (ms) | 0 | 85 |
| Throughput (tokens/s) | 2400 | 2200 |

## 2. Resource Usage

| Metric | DDP | FSDP |
|---|---|---|
| GPU mem peak (MB) | 18000 | 8000 |
| GPU util avg (%) | 92 | 88 |

## 3. Federated Update Timing

| Metric | DDP | FSDP |
|---|---|---|
| Avg WAN transfer (s) | 15.2 | 15.2 |
| Avg FedAvg aggregation (s) | 0.8 | 0.8 |

## 4. Model Accuracy

| Metric | DDP | FSDP | Delta |
|---|---|---|---|
| Val Loss ↓ | 1.23 | 1.22 | -0.01 |
| ROUGE-L F1 ↑ | 0.45 | 0.46 | 0.01 |
```

### 5.3 `evaluation.json` (final_evaluator.py 输出)

```json
{
  "status": "completed",
  "partition_id": 0,
  "split_info": { "total_samples": 50000, "test_samples": 2500 },
  "evaluation_seconds": 95.3,
  "federated": {
    "validation": { "val_loss": 1.23, "perplexity": 3.42 },
    "generation": { "rouge_l": {"f1": 0.45}, ... }
  },
  "base": { ... },
  "delta": { "val_loss": -0.15, ... },
  "mmlu": { "accuracy": 0.52 },
  "compression": [ ... ],
  "dp": { "epsilon": 4.2 }
}
```

---

## 六、常见问题

### Q: 轻量聚合模块报 `ModuleNotFoundError: No module named 'torch'`

**A**: 不应该出现。轻量模块 (`training_performance`, `resource_usage`, `federated_timing`, `comparison_report`, `dp_report`, `compression_report`, `data_split`) 已实现懒加载，不依赖 torch。如果报错，检查是否直接导入了重型模块（如 `from evaluation.validation_metrics import ...`），应改用 `from evaluation import compute_validation_metrics`（lazy stub）。

### Q: BERTScore 结果为 `None`

**A**: `bert_score` 库未安装。安装后可获取 BERTScore：`pip install bert-score`。未安装时不影响其他指标（ROUGE-L、Accuracy、F1、EM 正常输出）。

### Q: ROUGE-L 结果与 `rouge_score` 库不同

**A**: 如果 `rouge_score` 库未安装，会回退到内置 LCS 实现，结果可能有微小差异。安装后使用标准库：`pip install rouge-score`。

### Q: `generate_report` 报告中某些字段为 `None`

**A**: 对应的运行时数据未记录。常见原因：
- 训练 Job 未正常完成 → `metrics_detailed.json` 缺失
- 未启用最终评估 → `experiment_summary.json` 中无 `final_evaluation_metrics`
- 对象存储模式 → `evaluation_summary.json` 不可用

### Q: DDP vs FSDP 对比报告中只有一方的数据

**A**: 另一方的实验结果目录可能缺少产出文件。检查目录下是否存在 `metrics_detailed.json` 和 `experiment_summary.json`。

### Q: K8s 评估 Job 失败

**A**: 检查 `clientapp_error.json` 和 Pod 日志。常见原因：
- GPU 内存不足：减少 `NUM_EVAL_SAMPLES` 或 `EVAL_MAX_LENGTH`
- 模型加载失败：检查 `MODEL_STATE_PATH` 和 `MODEL_NAME` 是否匹配
- 数据集未缓存：确保镜像中已预下载或允许联网
