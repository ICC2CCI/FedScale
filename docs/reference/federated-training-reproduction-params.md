# 联邦微调实验复现参数 — Qwen2.5-0.5B + Medical Flashcards

本文档列出复现 S3R12v3 / SecAgg 实验所需的参数。

> **路径以当前双集群为准**（2026-09 起）：  
> - 代码入口：`experiments/run_s3r12v3_fsdp.py` + `scripts/start_s3r12v3_fsdp_run.sh`  
> - 配置：`configs/s3r12v3-fsdp-run.yaml` / `configs/s3r12v3-fsdp-secagg-verify.yaml`  
> - 模型：各端 `model/Qwen/Qwen2.5-0.5B`（见 `ICC1_PATHS.md` / `ICC2_PREP.md`）  
> - 数据：`data/medical_flashcards_*.json`，切分 `data/splits/`  
> - 联调流程：`docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md`  
> 下文部分表格仍保留早期单机 / LLaMA-Factory 路径作历史对照；**新实验请用上面路径**。

## 1. 基础模型

| 参数 | 值 |
|---|---|
| 模型名称 | Qwen/Qwen2.5-0.5B（base，非 instruct） |
| 下载来源 | ModelScope（`modelscope.snapshot_download`） |
| **当前双集群存放** | 各端仓库内 `model/Qwen/Qwen2.5-0.5B` |
| 历史单机路径 | `/data/models/Qwen/Qwen2.5-0.5B`（若仍使用） |
| 训练常用 dtype | fp16（FSDP + `transfer_dtype: fp16`）；加载可为 fp32/bf16 再进 Accelerate |
| 参数量 | ~630M，24 层 transformer |

## 2. 数据集

| 参数 | 值 |
|---|---|
| 数据集名称 | medalpaca/medical_meadow_medical_flashcards |
| 来源 | HuggingFace `medalpaca/medical_meadow_medical_flashcards` |
| 训练集 | 30176 samples |
| 验证集 | 3352 samples |
| 数据格式 | JSON 数组，每条 3 字段：`instruction` / `input` / `output` |
| 训练集路径（当前） | `data/medical_flashcards_train.json` |
| 验证集路径（当前） | `data/medical_flashcards_eval.json` |
| ICC1 切分 | `data/splits/icc1_client0_train.json` |
| ICC2 切分 | `data/splits/client1_train.json`（或节点文档约定路径） |
| 历史 LLaMA-Factory 路径 | `flower-llm/llamafactory-local/assets/datasets/...`（勿作当前入口） |

### 数据切分

| 参数 | 值 |
|---|---|
| 切分方式 | IID |
| 客户端数 | 2 |
| 切分代码 | `random.seed(SEED); random.shuffle(rows); client_train = [rows[i::n] for i in range(n)]` |
| 每客户端样本数 | 15088 |

### 数据样本示例

```json
{
  "instruction": "Answer this question truthfully",
  "input": "What is the name of the portion in eukaryotic hnRNA that contains noncoding segments of DNA that interrupt coding sequences?",
  "output": "The introns are the portion of eukaryotic hnRNA that contain intervening noncoding segments of DNA."
}
```

### Chat 模板格式

使用 Qwen 原生 `apply_chat_template` 转为 3 条消息：

```
<|im_start|>system
Answer this question truthfully<|im_end|>
<|im_start|>user
What is the name of the portion in eukaryotic hnRNA...?<|im_end|>
<|im_start|>assistant
The introns are the portion of eukaryotic hnRNA...<|im_end|>
```

| 参数 | 值 |
|---|---|
| tokenizer | Qwen2TokenizerFast |
| padding_side | right |
| legacy | False |
| pad_token | eos_token (`<|im_end|>`) |
| response_template | `<|im_start|>assistant\n`（token ids: [151644, 77091, 198]） |
| data_collator | DataCollatorForCompletionOnlyLM（只在 assistant 部分算 loss） |
| max_seq_length | 512 |
| packing | False |

## 3. 联邦训练配置

| 参数 | 值 |
|---|---|
| 客户端数 | 2（ICC1=0，ICC2=1） |
| 联邦轮数 | 常用 20（yaml / `NUM_ROUNDS_ENV`） |
| 每轮本地步数 | 30 |
| 聚合方式 | FedAvg（按客户端数据量加权）；SecAgg 时先 unmask 再平均 |
| 客户端选择 | 全选（每轮 2 个客户端都训练） |
| 覆盖比例 | 验证配置常 `coverage_h=10`（约 10% blocks） |
| 传输 | `transfer_dtype: fp16`；SecAgg 为 INT16 masked raw |
| seed | 20260831 |
| 入口 | `experiments/run_s3r12v3_fsdp.py` + `configs/s3r12v3-fsdp-*.yaml` |

### 聚合公式

```
global_state[key] += Σ_c (weight_c × delta_c[key]) / Σ_c weight_c

其中 weight_c = len(client_train[c])（客户端数据量）
```

## 4. 训练超参

本文档涉及两类训练，参数有差异，分别列出：

### 4.1 联邦训练（S2 / S3 / S3R11 / S3R12v3 等）

联邦训练用 SFTTrainer 直接调用，每轮每客户端本地训练 30 步。

| 参数 | 值 | 说明 |
|---|---|---|
| optimizer | AdamW（`adamw_torch`，transformers 默认） | |
| learning_rate | 1e-5 | |
| lr_scheduler_type | cosine | |
| warmup_ratio | 0.1 | 30 步内前 3 步 warmup |
| weight_decay | 0（默认，未显式设置） | |
| max_grad_norm | 1.0（默认） | |
| per_device_train_batch_size | 8 | |
| gradient_accumulation_steps | 2 | |
| 有效 batch size | 16 | 8 × 2 |
| max_steps | 30 | 每轮 30 步，不按 epoch 训练 |
| bf16 | True | |
| gradient_checkpointing | True | |
| dataloader_num_workers | 2 | |
| save_strategy | no | |
| report_to | none | |
| seed | 20260831 | |
| 每轮训练 seed | `SEED + rnd * 100 + cid` | 轮间有变化但可复现 |
| 每轮覆盖样本数 | 30 × 16 = 480 samples/客户端 | |

TrainingArguments 代码（联邦每轮本地训练）：

```python
TrainingArguments(
    output_dir=str(OUTPUT_DIR / f"r{rnd}-c{cid}"),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    max_steps=30,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    gradient_checkpointing=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=SEED + rnd * 100 + cid,
    dataloader_num_workers=2,
)
```

> **注意**：联邦训练每轮用 `max_steps=30` 而非 `num_train_epochs`，lr scheduler 在 30 步内完成 warmup + cosine 衰减到 0。每轮独立调度，不跨轮累积。

### 4.2 全量训练（历史基线 S1-D，LLaMA-Factory CLI）

> **历史单机路径**，不是当前双集群入口。当前联邦训练用 `experiments/run_s3r12v3_fsdp.py` + Accelerate FSDP。

全量训练曾使用 LLaMA-Factory CLI（`llamafactory-cli train`），参数通过 YAML 配置文件指定。

配置文件路径（历史）：`flower-llm/llamafactory-local/configs/qwen25-0.5b-medical-flashcards-full.yaml`

| 参数 | 值 | 说明 |
|---|---|---|
| stage | sft | 监督微调 |
| finetuning_type | full | 全参数微调（非 LoRA） |
| model_name_or_path | `/data/models/Qwen/Qwen2.5-0.5B` | 基础模型 |
| flash_attn | auto | 自动选择 attention 实现 |
| per_device_train_batch_size | 8 | 同联邦 |
| per_device_eval_batch_size | 8 | 同联邦 |
| gradient_accumulation_steps | 2 | 同联邦 |
| gradient_checkpointing | true | 同联邦 |
| learning_rate | 1.0e-5 | 同联邦 |
| num_train_epochs | 2.0 | 2 个 epoch（全量按 epoch 训） |
| lr_scheduler_type | cosine | 同联邦 |
| warmup_ratio | 0.03 | ⚠️ 与联邦不同（联邦是 0.1） |
| weight_decay | 0.01 | ⚠️ 联邦未显式设置（默认 0） |
| max_grad_norm | 1.0 | 同联邦 |
| optim | adamw_torch | 同联邦 |
| fp16 | true | ⚠️ 全量用 fp16 |
| bf16 | false | ⚠️ 联邦用 bf16 |
| seed | 20260831 | 同联邦 |
| cutoff_len | 512 | 同联邦 seq_len |
| preprocessing_num_workers | 4 | |
| dataloader_num_workers | 4 | ⚠️ 联邦是 2 |
| logging_steps | 10 | 同联邦 |
| save_strategy | "no" | 同联邦 |
| eval_strategy | steps | 每 100 步评估 |
| eval_steps | 100 | |
| val_size | 0.1 | 10% 训练数据作验证 |
| dataset | medical_flashcards | LLaMA-Factory dataset_info 注册名 |
| dataset_dir | `flower-llm/llamafactory-local/assets/datasets` | |
| template | qwen | LLaMA-Factory 内置 Qwen 模板 |
| report_to | none | |
| plot_loss | true | 训练结束后画 loss 曲线 |

LLaMA-Factory YAML 配置：

```yaml
### model
model_name_or_path: /data/models/Qwen/Qwen2.5-0.5B
flash_attn: auto

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: medical_flashcards
dataset_dir: /data/home/qiaoyanchen/liuchao/fedscale/flower-llm/llamafactory-local/assets/datasets
template: qwen
cutoff_len: 512
preprocessing_num_workers: 4
dataloader_num_workers: 4

### output
output_dir: /data/home/qiaoyanchen/liuchao/fedscale/output/s1d-qwen25-0.5b-medical-flashcards
logging_steps: 10
save_strategy: "no"
plot_loss: true
overwrite_output_dir: true
report_to: none
skip_memory_metrics: false

### train
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 2
gradient_checkpointing: true
learning_rate: 1.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0
optim: adamw_torch
fp16: true
bf16: false
seed: 20260831
ddp_timeout: 180000000
eval_strategy: steps
eval_steps: 100
val_size: 0.1
```

运行命令：

```bash
llamafactory-cli train \
  flower-llm/llamafactory-local/configs/qwen25-0.5b-medical-flashcards-full.yaml
```

### 4.3 全量训练 vs 联邦训练差异对照

| 参数 | S1-D 全量训练 | S2/S3R12v3 联邦训练 | 说明 |
|---|---|---|---|
| 训练方式 | `num_train_epochs=2.0` | `max_steps=30`/轮 | 全量按 epoch，联邦按 step |
| warmup_ratio | 0.03 | 0.1 | 联邦每轮 30 步需要更快 warmup |
| weight_decay | 0.01 | 0（默认） | 全量显式设置 |
| dtype | fp16 | bf16 | 不同精度策略 |
| dataloader_num_workers | 4 | 2 | |
| eval 方式 | 每 100 步，val_size=0.1 | 每轮结束，独立 3352 eval | 全量用 train 切 10%，联邦用独立 eval 集 |
| 训练框架 | LLaMA-Factory CLI | SFTTrainer 直接调用 | |
| lr scheduler | 跨整个训练（2 epoch） | 每轮独立（30 步内 warmup→cosine→0） | |
| 总训练步数 | ~3770 步（30176×2/16） | 20 轮×30 步×2 客户端 = 1200 步 | 全量训练更多 |
| 预期 eval loss | ~0.86（S1-D 终点） | ~1.01（S2 终点） | 全量有更多数据/更久训练 |

## 5. S3R12v3 分片参数

### 5.1 模型分组

| 参数 | 值 |
|---|---|
| 分组方式 | 25 个独立组 |
| layer 组 | 24 个（layer_0 ~ layer_23），每组 12 个 key，14.9M elems |
| non_layer 组 | 1 个，3 个 key（embed_tokens + lm_head + norm），272.3M elems |
| 分组依据 | key name 中含 `.layers.{N}.` 归入 layer_N 组，其余归入 non_layer 组 |

### 5.2 Block 切分

| 参数 | 值 |
|---|---|
| BLOCK_SIZE | 524288 elements（= 1 MB bf16，2 bytes/element） |
| 切分方式 | 每个 key 的 tensor 展平为 1D，按 BLOCK_SIZE 切分为连续 block |
| 总 block 数 | 1433 |
| 每层 block 数 | 38 |
| non_layer block 数 | 521 |
| block 表示 | `(key_name, start_elem, end_elem)` |

### 5.3 排列与轮转

| 参数 | 值 |
|---|---|
| 覆盖周期 H | 5（每 5 个成功轮次 = 1 个 Mask Epoch） |
| 排列算法 | Fisher-Yates shuffle（每组独立） |
| Epoch 数 | 4（20 轮 ÷ H=5） |
| 选择规则 | 每组选排列中 `position % H == slot` 的 block |
| slot | `(round - 1) % H` |
| 覆盖保证 | H=5 轮内每个 block 恰好被选一次（100% 覆盖，无饥饿） |

### 5.4 种子派生

```python
epoch_seed = SHA256("FedScale-BlockMask-v1" | SEED | epoch)
group_seed = SHA256(epoch_seed | "\x00" | "FedScale-GroupMask-v1" | group_id)
permutation = FisherYates(n_blocks, PRG=Random(int.from_bytes(group_seed, "big")))
```

| 参数 | 值 |
|---|---|
| epoch_seed | 32 bytes，每个 Mask Epoch 开始时生成一次 |
| group_seed | 32 bytes，由 epoch_seed + group_id 派生 |
| 新 Epoch | 重新 shuffle 所有 25 个组的 block 排列 |

### 5.5 Memory 机制

| 参数 | 值 |
|---|---|
| memory | 有（error feedback，累积未上传的 delta） |
| memory_decay | 0.9 |
| memory 更新方式 | block 级 |
| 已上传 block | memory 清零（`flat[start:end] = 0`） |
| 未上传 block | memory 保留 delta + 旧 memory |
| 整体 | `memory = (delta + old_memory) × decay`，已上传部分先清零再 decay |

### 5.6 Memory 更新代码

```python
def update_block_memory(to_send, selected_by_key, decay):
    new_memory = {}
    for key_name in to_send:
        flat = to_send[key_name].view(-1).to(torch.float32).clone()
        if key_name in selected_by_key:
            for s, e in selected_by_key[key_name]:
                flat[s:e] = 0.0  # 已上传 block 清零
        new_memory[key_name] = (flat * decay).to(original_dtype).view(original_shape)
    return new_memory
```

### 5.7 每轮上传量

| 参数 | 值 |
|---|---|
| 每轮上传比例 | ~20%（1/H = 1/5） |
| 每轮上传 block 数 | ~286 个 |
| 每轮上传 elems | ~126M |
| 每轮上传大小 | ~252 MB (bf16) |
| 上传波动 | 18.5%~21.5%（range 3%） |
| 20 轮总上传 | 5.0 GB |

## 6. 评估配置

| 参数 | 值 |
|---|---|
| 评估频率 | 每轮结束后 |
| 评估数据 | 全部 3352 eval samples |
| per_device_eval_batch_size | 8 |
| bf16 | True |
| 评估指标 | eval_loss（cross-entropy mean） |
| 评估代码 | `SFTTrainer.evaluate()` |

## 7. 上传协议（Block 级）

### 7.1 上传

```python
# 客户端训练后：
delta = trained_state - global_state
to_send = delta + client_memory

# 只上传 selected blocks 的切片：
for key_name, slices in selected_by_key.items():
    flat = to_send[key_name].view(-1)
    for start, end in slices:
        upload flat[start:end]  # 1D tensor 切片
```

### 7.2 聚合

```python
# 服务端 FedAvg：
for key_name, slices in selected_by_key.items():
    gflat = global_state[key_name].view(-1)
    for start, end in slices:
        acc = weighted_avg(client_deltas[key_name][start:end])
        gflat[start:end] += acc
    global_state[key_name] = gflat.view(original_shape)
```

### 7.3 Block 级操作的关键

由于 weight tensor 是多维的（如 q_proj 是 [896, 896]），所有 block 操作在 **flatten 后的 1D view** 上进行：

```python
flat = tensor.contiguous().view(-1)  # 多维 → 1D
slice = flat[start:end]              # 1D 切片
result = flat.view(original_shape)   # 1D → 多维（写回）
```

## 8. 完整每轮流程

```
每轮（round r）：

  1. 确定选择：
     slot = (r-1) % 5
     如新 Epoch（slot=0），重新 shuffle 所有 25 个组的 block 列表
     selected_by_key = 每组 position % 5 == slot 的 block 集合

  2. 每个客户端（2 个）：
     a. 加载全局模型 global_state
     b. 本地训练 30 步（全模型训练）
     c. delta = trained_state - global_state
     d. to_send = delta + client_memory
     e. 上传 selected blocks 的 1D 切片
     f. 更新 memory：已上传 block 清零，整体 ×0.9

  3. 服务端聚合：
     global_state[selected_blocks] += weighted_avg(client_deltas[selected_blocks])

  4. 评估：加载更新后的 global_state，在 3352 eval samples 上算 loss
```

## 9. 运行环境

### 9.1 当前双集群（优先）

| 组件 | 值 |
|---|---|
| 拓扑 | Central（CPU + MinIO）+ ICC1 / ICC2（各 8×V100 32GB） |
| Python | 3.10 |
| Client conda | `flwr-ft` |
| Server conda | `fedscale-server`（聚合进程） |
| PyTorch | 2.8.0+cu128（实测；V100 driver 可向下兼容） |
| Transformers / Accelerate | 见 `deployment/environment-setup.md` |
| 模型 | 各端 `model/Qwen/Qwen2.5-0.5B` |
| 数据 | `data/medical_flashcards_*.json`，`data/splits/` |
| 配置 | `configs/s3r12v3-fsdp-run.yaml` / `configs/s3r12v3-fsdp-secagg-verify.yaml` |
| 启动 | `bash scripts/start_s3r12v3_fsdp_run.sh` |
| 结果 | `results/YYYYMMDDHHMM/` |

路径约定：`ICC1_PATHS.md`、`ICC2_PREP.md`、`deployment/dual-cluster-nodes.md`。

### 9.2 历史单机消融（勿作当前入口）

| 组件 | 版本 |
|---|---|
| GPU | 曾用 1× H100 80GB |
| Conda | `flwr-ft`（单机路径） |
| 脚本 | `scripts/run_s3r12v3_block_uniform.py` 等 |
| 模型/数据 | `/data/models/...`、`flower-llm/llamafactory-local/...` |

## 10. 脚本与输出

### 10.1 当前双集群

| 文件 | 说明 |
|---|---|
| `scripts/start_s3r12v3_fsdp_run.sh` | 一键启动 Server + 两端 |
| `experiments/run_s3r12v3_fsdp.py` | FSDP 客户端 |
| `experiments/server/aggregation_server.py` | 聚合服务 |
| `scripts/plot_s3r12v3_fsdp_run.py` | 画 `figures/` |
| `results/<run_id>/round_log.json` | 主日志 |
| `results/<run_id>/figures/` | loss / time / transfer 图 |

### 10.2 历史单机消融

| 文件 | 说明 |
|---|---|
| `scripts/run_s3r12v3_block_uniform.py` | 单机 S3R12v3（早期） |
| `scripts/run_s3r12v3_ratio.py --ratio 0.X` | ratio 消融 |
| `output/s3r12v3-*/round_log.json` | 单机结果 |

## 11. 复现步骤

### 11.1 当前双集群（推荐）

```bash
# 各端：conda activate flwr-ft；模型与数据路径见 ICC1_PATHS.md / ICC2_PREP.md
# Central：MinIO + aggregation_server（脚本会拉起）

# 非 SecAgg
bash scripts/start_s3r12v3_fsdp_run.sh

# SecAgg 20 轮（常用 8081）
RUN_CONFIG=$PWD/configs/s3r12v3-fsdp-secagg-verify.yaml \
AGGREGATION_PORT_OVERRIDE=8081 \
NUM_ROUNDS_ENV=20 \
bash scripts/start_s3r12v3_fsdp_run.sh

# 正式对照保持 --detailed-train-metrics 关闭（默认关）
```

参考正式跑次：`results/202609180941/`（SecAgg Hadamard，R20 eval=0.985）。

### 11.2 历史单机消融

```bash
conda activate flwr-ft
ls /data/models/Qwen/Qwen2.5-0.5B/   # 或当前仓库 model/...
python scripts/run_s3r12v3_block_uniform.py
python scripts/run_s3r12v3_ratio.py --ratio 0.10
```

## 12. 预期结果

### 12.1 当前双集群（实测）

| 路径 | 参考目录 | R20 eval |
|---|---|---|
| 非 SecAgg fp16 | `results/20260914-final-clean` | ≈0.987 |
| SecAgg Hadamard（正式） | `results/202609180941` | **0.985** |
| 旧 SecAgg per-window | `results/202609162025` | 1.328 |

### 12.2 历史单机 ratio 消融（20 轮）

| ratio | H | 终点 eval loss | 总上传 | gap vs S2 |
|---|---|---|---|---|
| 5% | 20 | 1.1298 | 1.3 GB | +0.121 |
| 10% | 10 | 1.0979 | 2.5 GB | +0.089 |
| **20%** | 5 | **1.0514** | 5.0 GB | +0.042 |
| 30% | 3 | 1.0326 | 8.4 GB | +0.024 |
| 40% | 5, 2slots | 1.0316 | 10.1 GB | +0.023 |
| 50% | 2 | 1.0196 | 12.6 GB | +0.011 |
| 100% (S2) | — | 1.0091 | 25.2 GB | — |
