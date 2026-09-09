# 双集群 FSDP 部署 S3R12v3 规划

- **状态**：planned
- **负责人 / Agent**：opencode
- **创建**：2026-09-09
- **更新**：2026-09-09

## 目标

在两台 8×V100 服务器（ICC1、ICC2）+ 1 台 Central Server 上部署 S3R12v3 联邦微调，使用 FSDP 进行多卡训练。

## 背景

- 当前 S3R12v3 实验在单卡 H100 上完成（单进程模拟 2 客户端）
- 新部署需要真正跨节点：2 个集群各 8×V100 32GB，1 台 Central Server 聚合
- V100 不支持 bf16，不支持 flash_attn 2，需要适配
- S3R12v3 核心算法（block 选择、permutation、memory）不变，改的是外围 I/O 层

## 硬件

| 节点 | 角色 | 硬件 | 职责 |
|---|---|---|---|
| Central Server | Flower ServerApp | CPU 即可 | 维护 global_state、调度 block mask、FedAvg 聚合 |
| ICC1 | Flower SuperNode (Client 0) | 8× V100 32GB | FSDP 本地训练 + block 选择 + 上传 |
| ICC2 | Flower SuperNode (Client 1) | 8× V100 32GB | 同 ICC1 |

## 部署架构

```
                    ┌─────────────────────────┐
                    │     Central Server       │
                    │  (Flower ServerApp)      │
                    │                          │
                    │  - 聚合 (FedAvg)          │
                    │  - block mask 调度        │
                    │  - global_state 维护     │
                    │  - 对象存储 (可选)        │
                    └────────┬────────────────┘
                             │ gRPC / HTTPS
                    ┌────────┴────────┐
                    │                 │
             ┌──────┴──────┐   ┌──────┴──────┐
             │    ICC1     │   │    ICC2     │
             │  Client 0   │   │  Client 1   │
             │             │   │             │
             │ 8× V100     │   │ 8× V100     │
             │ FSDP (8卡)  │   │ FSDP (8卡)  │
             │             │   │             │
             │ SuperNode   │   │ SuperNode   │
             └─────────────┘   └─────────────┘
```

### 通信链路

- ICC 内部：8 GPU 之间 FSDP all-gather/reduce-scatter（NVLink 或 PCIe）
- ICC ↔ Server：只 rank 0 跨网络通信（上传 block 切片、下载 global_state）
- FSDP 的多卡通信只在 ICC 内部，不跨网络

### 通信方式

| 方式 | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|
| Flower gRPC（默认） | 简单，框架内置 | 大模型传输慢 | 模型较小或带宽充足 |
| 对象存储（MinIO） | 支持断点续传、大文件 | 需额外部署 MinIO | **推荐**，模型 1GB+ |

仓库已有 `object_store_strategy.py` 和 MinIO 部署配置，建议复用。

### 部署方式

Central Server：K8s 部署 SuperLink + ServerApp（现有 `configs/superlink-deployment.yaml` 已支持）

ICC1/ICC2：每台机器部署 SuperNode，连接到 Central Server

```bash
# ICC1 上
flower-supernode --insecure --superlink <server-ip>:9092

# ICC2 上
flower-supernode --insecure --superlink <server-ip>:9092
```

## S3R12v3 适配 FSDP 的改动

### 当前 S3R12v3 流程（单 GPU）

```
1. 加载 global_state（全量）
2. 本地训练 30 步（SFTTrainer，单卡）
3. delta = local_state - global_state
4. block 级选择（permutation + rotation）
5. 上传 selected blocks
6. server 聚合
7. memory 机制（error feedback）
```

### FSDP 下的变化

FSDP 将模型参数/梯度/优化器状态分片到 8 张卡上，每张卡只持有 1/8。block 选择需要完整 state dict，这是主要适配点。

### 改动 1：模型加载 → FSDP 包装

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

dist.init_process_group(backend="nccl")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
model.to(rank)

fsdp_config = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=fsdp_config,
    device_id=rank,
    use_orig_params=True,
)
```

### 改动 2：提取完整 state dict（block 选择需要）

```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
    full_state = model.state_dict()  # 只有 rank 0 拿到完整 state dict
```

关键：`offload_to_cpu=True` 避免 rank 0 显存爆炸，`rank0_only=True` 只在 rank 0 收集。

### 改动 3：block 选择只在 rank 0 执行

```python
if rank == 0:
    delta = {k: full_state[k] - global_state[k] for k in full_state}
    selected_blocks = block_selection(permutation, round_idx, H)
    to_send = extract_blocks(delta, selected_blocks)
    upload_to_server(to_send)
```

### 改动 4：下载 global_state 后广播

```python
if rank == 0:
    global_state = download_from_server()

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
    if rank == 0:
        model.load_state_dict(global_state)
    # FSDP 内部会自动 broadcast 并重新分片
```

### 改动 5：V100 特殊处理

| 项目 | H100（当前） | V100（新） | 改动 |
|---|---|---|---|
| dtype | bf16 | **fp16** | V100 不支持 bf16 |
| flash_attn | auto | **disabled** | V100 不支持 flash_attn 2 |
| gradient_checkpointing | True | True（需 `use_cache=False`） | FSDP + grad_ckpt 需 `use_orig_params=True` |
| 显存 | 80GB | 32GB | 需更激进的分片 + grad_ckpt |

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
```

### S3R12v3 算法本身是否需要改？

核心算法不需要改。block 选择、permutation、memory 机制都在 rank 0 的 CPU 上操作完整 state dict，与 FSDP 无关。

需要改的是外围的 I/O 层：

| 层 | 改动 | 算法 |
|---|---|---|
| 模型加载 | FSDP 包装 | 不改 |
| 训练循环 | SFTTrainer → FSDP-aware Trainer | 不改 |
| state dict 提取 | `FSDP.state_dict_type` 上下文 | 不改 |
| block 选择 | 只在 rank 0 | **不改** |
| 上传/下载 | 只 rank 0 | **不改** |
| 聚合 | server 端无变化 | **不改** |
| memory | rank 0 CPU 上 | **不改** |

### 需要新增/修改的代码文件

| 文件 | 改动 |
|---|---|
| `experiments/run_s3r12v3_block_uniform.py` | 新增 FSDP 分支：`--fsdp` flag |
| `flowertune-llm/flowertune_llm/distributed_trainer.py` | 已有 FSDP 支持，需适配 block 选择 |
| `flowertune-llm/flowertune_llm/fedscale_state.py` | state dict 提取改为 FSDP-aware |
| 新增 `experiments/run_s3r12v3_fsdp.py` | FSDP 版本（推荐独立文件，避免单卡/多卡逻辑混杂） |

## V100 显存估算

Qwen2.5-0.5B 在 8× V100 32GB 上 FSDP：

| 项目 | 显存（每卡） |
|---|---|
| 模型参数（FSDP 分片 1/8） | ~0.6 GB / 8 = 0.075 GB |
| 梯度（FSDP 分片） | 0.075 GB |
| 优化器状态（AdamW，分片） | 0.15 GB |
| 激活值（batch=8, seq=512, grad_ckpt） | ~8 GB |
| **总计** | **~8.3 GB / 卡** |

结论：8× V100 32GB 完全够用，甚至可以增大 batch size。

## 潜在风险

| 风险 | 影响 | 对策 |
|---|---|---|
| V100 fp16 数值不稳定 | 训练 loss 异常 | 用 `GradScaler` 或改用 fp32 累加 |
| FSDP + gradient_checkpointing 兼容 | 报错 | `use_orig_params=True` |
| `summon_full_params` 显存峰值 | rank 0 OOM | `offload_to_cpu=True` |
| 跨节点 FSDP（误用） | 通信爆炸 | 确保只在 ICC 内部 FSDP，不跨节点 |
| block 选择在 rank 0 阻塞 | 其他 rank 空等 | 可接受（block 选择 < 1s） |

## 步骤

- [ ] 1. 先在单台 8×V100 上验证 FSDP 训练（不接联邦，确认 FSDP + fp16 + grad_ckpt 能跑通）
- [ ] 2. 再接入 S3R12v3 block 选择（rank 0 提取 state dict + 选择 + 模拟上传）
- [ ] 3. 最后接入 Central Server（Flower SuperNode 跨节点联邦）
- [ ] 4. 端到端验证：ICC1 + ICC2 + Server 三节点联邦 20 轮

## 验证

- FSDP 训练 loss 正常下降（对比单卡 baseline）
- S3R12v3 block 选择在 FSDP 下 upload_ratio 稳定在 18.5%~21.5%
- 端到端 20 轮 eval loss ≈ 1.05（对比单卡 H100 的 1.0514）
- V100 fp16 无数值异常

## 风险与回滚

- 如 FSDP 适配困难，可回退到单卡模式（每台 V100 服务器用 1 张卡跑，不 FSDP）
- 如 fp16 不稳定，可尝试 fp32（显存够用）
- 如跨节点通信不稳定，可先用对象存储异步传输

## 日志

- 2026-09-09: 讨论双集群 FSDP 部署架构，记录规划文档
