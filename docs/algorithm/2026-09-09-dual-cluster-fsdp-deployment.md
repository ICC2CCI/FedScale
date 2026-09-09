# 双集群 FSDP 部署 S3R12v3 规划

- **状态**：planned
- **负责人 / Agent**：opencode
- **创建**：2026-09-09
- **更新**：2026-09-09

## 目标

在两台 8×V100 服务器（ICC1、ICC2）+ 1 台 Central Server 上部署 S3R12v3 联邦微调，使用 FSDP 进行多卡训练。去掉 Flower 框架，用自写轻量通信层 + MinIO 对象存储，为未来 7B/13B 扩展打基础。

## 最终架构决策

| 决策项 | 选定方案 | 理由 |
|---|---|---|
| 框架 | **去掉 Flower**，自写轻量 HTTP+MinIO 通信层 | S3R12v3 通信简单，去掉 Flower 后 Accelerate FSDP 多进程启动不受约束，减少框架耦合 |
| 训练框架 | **Accelerate + FSDP** | HF 官方支持，稳定，未来可切换 DeepSpeed/ZeRO |
| 通信方式 | **MinIO 对象存储** | 支持 7B/13B 大模型断点续传，天然适配 block 切片，仓库已有 `object_store_strategy.py` |
| Central Server | **单独服务器**，K8s 部署 | 模拟真实场景，无需 GPU，需 32GB+ 内存 |
| MinIO 部署 | **放 Central Server 上**，ICC1/ICC2 通过 HTTP 访问 | 统一存储，简化部署 |
| 数据划分 | **50/50 均分**，后续可调整 | 代码留 `--data-split` 参数 |
| 监控 | **复用 `experiment-dashboard/`** | 读 `round_log.json` 展示 |

## 硬件

| 节点 | 角色 | 硬件 | 职责 |
|---|---|---|---|
| Central Server | 聚合节点 | CPU 4C+，内存 32GB+，无需 GPU | 维护 global_state、调度 block mask、FedAvg 聚合、MinIO 存储 |
| ICC1 | Client 0 | 8× V100 32GB，CUDA 13.0 driver | FSDP 本地训练 + block 选择 + 上传到 MinIO |
| ICC2 | Client 1 | 8× V100 32GB，CUDA 13.0 driver | 同 ICC1 |

### V100 环境说明

- `nvidia-smi` 显示 CUDA 13.0（driver 支持的最高版本），向下兼容
- `nvcc` 未安装，不需要单独安装 CUDA toolkit（PyTorch 自带 CUDA runtime）
- PyTorch 2.8.0+cu128 可直接使用

## 部署架构

```
                    ┌──────────────────────────────────┐
                    │         Central Server            │
                    │  (无 GPU, CPU 4C+, 内存 32GB+)     │
                    │                                   │
                    │  ┌─────────────────────────────┐ │
                    │  │  Aggregation Server (Python) │ │
                    │  │  - FedAvg 聚合                │ │
                    │  │  - block mask 调度            │ │
                    │  │  - global_state 维护          │ │
                    │  │  - round_log 记录             │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────┴──────────────────┐ │
                    │  │  MinIO 对象存储               │ │
                    │  │  - global_state/             │ │
                    │  │  - uploads/round-N/client-C/ │ │
                    │  │  - (block 切片)              │ │
                    │  └─────────────────────────────┘ │
                    └────────┬──────────────┬──────────┘
                             │ HTTP/S3      │ HTTP/S3
                    ┌────────┴──────┐ ┌──────┴────────┐
                    │    ICC1       │ │    ICC2       │
                    │  Client 0     │ │  Client 1     │
                    │               │ │               │
                    │ 8× V100 32GB  │ │ 8× V100 32GB  │
                    │ FSDP (8卡)    │ │ FSDP (8卡)    │
                    │ Accelerate    │ │ Accelerate    │
                    │               │ │               │
                    │ rank0:        │ │ rank0:        │
                    │  下载global   │ │  下载global   │
                    │  上传blocks   │ │  上传blocks   │
                    │ rank1-7:      │ │ rank1-7:      │
                    │  只做训练     │ │  只做训练     │
                    └───────────────┘ └───────────────┘
```

### 通信流程（每轮）

```
1. Server 生成 RoundPlan (block mask permutation)
   → 写到 MinIO: global_state/round-N/plan.json

2. ICC1/ICC2 rank0 从 MinIO 下载 global_state + RoundPlan

3. ICC1/ICC2 广播 global_state 给 8 卡 → FSDP 重新分片

4. ICC1/ICC2 各自 FSDP 本地训练 30 步

5. ICC1/ICC2 rank0 提取完整 state dict (offload_to_cpu)
   → 计算 delta = local_state - global_state
   → block 选择 (rank0 CPU)
   → 上传 selected blocks 到 MinIO: uploads/round-N/client-C/blocks.pt

6. Server 从 MinIO 拉取两个 client 的 blocks
   → FedAvg 聚合 → 更新 global_state
   → 写新 global_state 到 MinIO: global_state/round-(N+1)/state.pt

7. ICC1/ICC2 rank0 下载新 global_state → 进入下一轮
```

### 通信量估算

| 模型大小 | 全量上传 | S3R12v3 20% | 10% | 7B 20% | 13B 20% |
|---|---|---|---|---|---|
| 每轮每客户端 | 1.0 GB | 250 MB | 125 MB | 2.8 GB | 5.2 GB |
| 每轮总计（2客户端） | 2.0 GB | 500 MB | 250 MB | 5.6 GB | 10.4 GB |
| 20 轮总计 | 40 GB | 10 GB | 5 GB | 112 GB | 208 GB |

MinIO 断点续传确保大文件传输稳定。

## 通信层设计（去掉 Flower）

### 协议：HTTP REST + MinIO S3 API

不使用 Flower 的 gRPC 协议，改用简单的 HTTP REST + MinIO：

**Server 端 REST API**：

```
GET  /api/round/current          → 返回当前轮次号
GET  /api/round/{N}/plan         → 返回 RoundPlan (block mask)
POST /api/round/{N}/client/{C}/upload-complete  → 通知上传完成
GET  /api/round/{N}/result       → 返回聚合完成标志 + eval loss
```

**MinIO 对象路径**：

```
fedscale-bucket/
├── global_state/
│   ├── round-0/state.pt         → 初始模型
│   ├── round-1/state.pt         → 第1轮聚合后
│   └── ...
├── uploads/
│   ├── round-1/
│   │   ├── client-0/blocks.pt   → ICC1 上传的 block 切片
│   │   └── client-1/blocks.pt   → ICC2 上传的 block 切片
│   └── ...
└── plans/
    └── round-1/plan.json        → block mask 调度计划
```

### 代码结构

```
experiments/
├── run_s3r12v3_fsdp.py          ← 客户端主脚本 (accelerate launch)
├── server/
│   ├── aggregation_server.py    ← Central Server (REST API + 聚合)
│   ├── minio_client.py          ← MinIO 读写封装
│   └── block_scheduler.py       ← block mask 调度 (从 S3R12v3 提取)
└── shared/
    ├── block_selection.py       ← block 选择算法 (client/server 共用)
    ├── state_dict_utils.py      ← FSDP state dict 提取/加载
    └── protocol.py              ← 通信协议定义
```

## FSDP 适配

### 改动 1：模型加载 → FSDP 包装

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

dist.init_process_group(backend="nccl")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # V100 不支持 bf16
    attn_implementation="eager",  # V100 不支持 flash_attn 2
)
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
    use_orig_params=True,  # 兼容 gradient_checkpointing
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
    # 上传到 MinIO
    minio_client.put_object(f"uploads/round-{N}/client-{C}/blocks.pt", to_send)
    # 通知 Server
    requests.post(f"{server_url}/api/round/{N}/client/{C}/upload-complete")
```

### 改动 4：下载 global_state 后广播

```python
if rank == 0:
    # 从 MinIO 下载
    global_state = minio_client.get_object(f"global_state/round-{N}/state.pt")

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
    if rank == 0:
        model.load_state_dict(global_state)
    # FSDP 内部会自动 broadcast 并重新分片
```

### 改动 5：V100 特殊处理

| 项目 | H100（当前） | V100（新） | 改动 |
|---|---|---|---|
| dtype | bf16 | **fp16** | V100 不支持 bf16 |
| flash_attn | auto | **disabled (eager)** | V100 不支持 flash_attn 2 |
| gradient_checkpointing | True | True（需 `use_cache=False`） | FSDP + grad_ckpt 需 `use_orig_params=True` |
| 显存 | 80GB | 32GB | 需更激进的分片 + grad_ckpt |

### S3R12v3 算法本身不需要改

核心算法不需要改。block 选择、permutation、memory 机制都在 rank 0 的 CPU 上操作完整 state dict，与 FSDP 无关。

| 层 | 改动 | 算法 |
|---|---|---|
| 模型加载 | FSDP 包装 | 不改 |
| 训练循环 | Accelerate + FSDP | 不改 |
| state dict 提取 | `FSDP.state_dict_type` 上下文 | 不改 |
| block 选择 | 只在 rank 0 | **不改** |
| 上传/下载 | MinIO 替代 gRPC | **不改** |
| 聚合 | server 端无变化 | **不改** |
| memory | rank 0 CPU 上 | **不改** |

## Accelerate FSDP 配置

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_mixed_precision: fp16
num_processes: 8
gpu_ids: 0,1,2,3,4,5,6,7
```

启动命令：

```bash
accelerate launch --config_file accelerate_config.yaml \
  experiments/run_s3r12v3_fsdp.py \
  --client-id 0 \
  --server-url http://<central-server-ip>:8080 \
  --minio-endpoint http://<central-server-ip>:9000 \
  --model-path /data/models/Qwen/Qwen2.5-0.5B \
  --data-path data/medical_flashcards_train.json
```

## V100 显存估算

Qwen2.5-0.5B 在 8× V100 32GB 上 FSDP：

| 项目 | 显存（每卡） |
|---|---|
| 模型参数（FSDP 分片 1/8） | ~0.6 GB / 8 = 0.075 GB |
| 梯度（FSDP 分片） | 0.075 GB |
| 优化器状态（AdamW，分片） | 0.15 GB |
| 激活值（batch=8, seq=512, grad_ckpt） | ~8 GB |
| **总计** | **~8.3 GB / 卡** |

7B 模型估算：

| 项目 | 显存（每卡） |
|---|---|
| 模型参数（FSDP 分片 1/8） | ~14 GB / 8 = 1.75 GB |
| 梯度（FSDP 分片） | 1.75 GB |
| 优化器状态（AdamW，分片） | 3.5 GB |
| 激活值（batch=4, seq=512, grad_ckpt） | ~12 GB |
| **总计** | **~19 GB / 卡** |

结论：0.5B 和 7B 在 V100 32GB 上都够用。

## Central Server 部署

### 组件

```
Central Server
├── aggregation_server.py    ← REST API + 聚合逻辑
├── MinIO                    ← 对象存储
└── experiment-dashboard/    ← 监控看板（可选）
```

### 部署步骤

```bash
# 1. 安装 MinIO
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -v /data/minio:/data \
  -e MINIO_ROOT_USER=fedscale \
  -e MINIO_ROOT_PASSWORD=<password> \
  minio/minio server /data --console-address ":9001"

# 2. 启动聚合服务器
python experiments/server/aggregation_server.py \
  --port 8080 \
  --minio-endpoint http://localhost:9000 \
  --num-clients 2 \
  --num-rounds 20 \
  --ratio 0.2

# 3. 启动监控看板（可选）
cd experiment-dashboard && python server.py --port 3000
```

## 潜在风险

| 风险 | 影响 | 对策 |
|---|---|---|
| V100 fp16 数值不稳定 | 训练 loss 异常 | 用 `GradScaler` 或改用 fp32 累加 |
| FSDP + gradient_checkpointing 兼容 | 报错 | `use_orig_params=True` |
| `summon_full_params` 显存峰值 | rank 0 OOM | `offload_to_cpu=True` |
| 跨节点 FSDP（误用） | 通信爆炸 | 确保只在 ICC 内部 FSDP，不跨节点 |
| block 选择在 rank 0 阻塞 | 其他 rank 空等 | 可接受（block 选择 < 1s） |
| MinIO 网络中断 | 上传/下载失败 | 重试机制 + 断点续传 |
| Central Server 内存不足 | 聚合 OOM | 7B 需要 32GB+，13B 需要 64GB+ |

## 步骤

- [ ] 1. Central Server 部署：MinIO + aggregation_server.py
- [ ] 2. ICC1/ICC2 环境搭建：Accelerate + FSDP + PyTorch cu128
- [ ] 3. 单台 V100 验证 FSDP 训练（不接联邦，确认 FSDP + fp16 + grad_ckpt 能跑通）
- [ ] 4. 接入 S3R12v3 block 选择（rank 0 提取 state dict + 选择 + 上传 MinIO）
- [ ] 5. 端到端验证：ICC1 + ICC2 + Server 三节点联邦 20 轮
- [ ] 6. 对比单卡 H100 结果（eval loss ≈ 1.0514），验证 FSDP 无精度损失

## 验证

- FSDP 训练 loss 正常下降（对比单卡 baseline）
- S3R12v3 block 选择在 FSDP 下 upload_ratio 稳定在 18.5%~21.5%
- 端到端 20 轮 eval loss ≈ 1.05（对比单卡 H100 的 1.0514）
- V100 fp16 无数值异常
- MinIO 大文件传输稳定（断点续传验证）

## 风险与回滚

- 如 FSDP 适配困难，可回退到单卡模式（每台 V100 服务器用 1 张卡跑，不 FSDP）
- 如 fp16 不稳定，可尝试 fp32（显存够用）
- 如 MinIO 通信不稳定，可回退到直接 HTTP 传输
- 如 Accelerate FSDP 有问题，可回退到手动 `torch.distributed.fsdp`

## 日志

- 2026-09-09: 讨论双集群 FSDP 部署架构，记录规划文档
- 2026-09-09: 确认最终架构决策：去掉 Flower、用 Accelerate+FSDP、MinIO 通信、Central Server 独立部署。V100 CUDA 13.0 driver 向下兼容 cu128 PyTorch。
- 2026-09-09: Central Server 侧 MinIO 已用 Docker Compose 部署（运维细节见 `deployment/central-server-prep.md`，不改动本规划 checklist 正文以免与上游并行编辑冲突）。
