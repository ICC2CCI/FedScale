# FlowRTune-LLM 联邦训练 + 分布式训练 设计说明

## 1. 项目概述

FlowRTune-LLM 是基于 [Flower](https://flower.ai/) 框架的大模型联邦微调系统，结合了**联邦学习（Federated Learning）**和**分布式数据并行训练（DDP）**两种能力，在 Kubernetes 集群上实现对 LLM 的隐私保护微调。

### 1.1 核心能力

| 能力 | 说明 |
|---|---|
| **联邦训练** | 多个客户端（SuperNode）各自持有本地数据，通过 FedAvg 聚合模型参数，数据不出域 |
| **分布式训练** | 单个客户端内部通过 PyTorch DDP 在多节点多 GPU 上并行训练，加速单轮训练 |
| **多集群部署** | 同一套代码可部署到不同 K8s 集群（A/B/C...），通过环境变量适配各集群差异 |

### 1.2 技术栈

| 层级 | 技术 |
|---|---|
| 联邦框架 | Flower 1.28.0 (ServerApp + ClientApp + SuperNode + SuperLink) |
| 训练框架 | PyTorch 2.8.0 + DDP |
| LLM 微调 | TRL 0.8.1 (SFTTrainer) + PEFT (LoRA) + Transformers 4.53.0 |
| 量化 | BitsAndBytes 4-bit (QLoRA) |
| 数据集 | Flower Datasets + IID Partitioner |
| 部署 | Kubernetes (Deployment + Indexed Job + PVC/CFS) |
| 容器 | Docker 多阶段构建 |

---

## 2. 系统架构

### 2.1 整体架构图

```
                    ┌─────────────────────────────────┐
                    │         SuperLink (协调器)        │
                    │  ┌───────────┐ ┌──────────────┐  │
                    │  │ Fleet API │ │ ServerApp     │  │
                    │  │  :9092    │ │ (FedAvg策略)  │  │
                    │  └─────┬─────┘ └──────────────┘  │
                    └────────┼────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
   ┌──────────▼──┐   ┌──────▼──────┐   ┌───▼───────────┐
   │ SuperNode A │   │ SuperNode B │   │ SuperNode ... │
   │ (客户端 A)   │   │ (客户端 B)   │   │ (客户端 ...)   │
   │             │   │             │   │               │
   │ ┌─────────┐ │   │ ┌─────────┐ │   │               │
   │ │SuperExec│ │   │ │SuperExec│ │   │               │
   │ │ClientApp│ │   │ │ClientApp│ │   │               │
   │ └────┬────┘ │   │ └────┬────┘ │   │               │
   └──────┼──────┘   └──────┼──────┘   └───────────────┘
          │                 │
          ▼                 ▼
   ┌──────────────┐  ┌──────────────┐
   │ K8s Indexed  │  │ K8s Indexed  │
   │ Job (N Pods) │  │ Job (N Pods) │
   │ 训练Pod 0..N │  │ 训练Pod 0..N │
   └──────────────┘  └──────────────┘
```

### 2.2 组件说明

| 组件 | 镜像 | 职责 |
|---|---|---|
| **SuperLink** | `flwr/superlink:1.28.0` | 联邦训练协调器，运行 ServerApp，执行 FedAvg 聚合策略 |
| **SuperNode** | `flwr/supernode:1.28.0` | 联邦客户端节点，连接 SuperLink，桥接 ClientApp |
| **SuperExec** | `flwr_superexec:<版本>` | 运行 ClientApp 逻辑，调度 K8s 训练 Job |
| **训练 Pod** | `flwr_client_train:<版本>` | 实际执行 DDP 分布式训练的 GPU Pod |

---

## 3. 联邦训练流程

### 3.1 训练时序图

```
SuperLink(ServerApp)        SuperNode-A(ClientApp)        SuperNode-B(ClientApp)
        │                           │                             │
        │  Round 1: 分发全局模型      │                             │
        ├───── model weights ──────►│                             │
        ├───── model weights ───────────────────────────────────►│
        │                           │                             │
        │                    本地训练(DDP)                    本地训练(DDP)
        │                    返回更新后模型                    返回更新后模型
        │                           │                             │
        │◄── updated weights ──────┤                             │
        │◄── updated weights ────────────────────────────────────┤
        │                           │                             │
        │  FedAvg 聚合              │                             │
        │  ──────────              │                             │
        │                           │                             │
        │  Round 2: ...            │                             │
        │                           │                             │
```

### 3.2 FedAvg 聚合策略

- **聚合算法**: Federated Averaging (FedAvg)
- **参与比例**: `fraction_train` 控制每轮参与训练的客户端比例
- **学习率调度**: Cosine Annealing（随轮次衰减）
- **模型保存**: 每 `save_every_round` 轮保存全局模型 checkpoint

### 3.3 ServerApp (`server_app.py`)

```python
# 核心流程
1. 初始化全局模型 → 提取 LoRA PEFT 权重作为初始参数
2. 配置 FedAvg 策略（参与比例、评估函数）
3. 启动 N 轮联邦训练
4. 每轮聚合客户端返回的 LoRA 权重
5. 定期保存全局模型 checkpoint
```

### 3.4 ClientApp (`client_app.py`)

```python
# 核心流程（每轮训练）
1. 接收服务端下发的全局 LoRA 权重
2. 加载基座模型 + 设置 PEFT 权重
3. 保存初始权重到 PVC 共享存储
4. 创建 K8s Indexed Job（N 个训练 Pod）
5. 等待 Job 完成
6. 从 PVC 读取训练后的 LoRA 权重
7. 返回更新后的权重给服务端
8. 清理 K8s Job 和 Service
```

---

## 4. 分布式训练设计

### 4.1 DDP 架构

```
              K8s Indexed Job (parallelism=N, completions=N)
              ┌──────────────────────────────────────────────┐
              │                                              │
     Pod-0 (Node 0)     Pod-1 (Node 1)     Pod-2 (Node 2)
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │ GPU 0       │    │ GPU 0       │    │ GPU 0       │
     │ rank=0      │    │ rank=1      │    │ rank=2      │
     │ 主进程       │    │ 工作进程     │    │ 工作进程     │
     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
            │                  │                  │
            └──────── NCCL 通信（AllReduce）───────┘
                      Master: Pod-0
```

### 4.2 分布式通信

| 参数 | 来源 | 说明 |
|---|---|---|
| `MASTER_ADDR` | Headless Service DNS | `{job}-0.{job}.{namespace}.svc.cluster.local` |
| `MASTER_PORT` | 固定值 | `29500` |
| `WORLD_SIZE` | `NUM_NODES × GPUS_PER_NODE` | 总 GPU 数 |
| `node_rank` | `BATCH_JOB_COMPLETION_INDEX` / Pod 名称解析 | 节点编号 |
| `local_rank` | `LOCAL_RANK` 环境变量 | 节点内 GPU 编号 |

### 4.3 DNS 解析机制

```
K8s Indexed Job + Headless Service:

Pod DNS: {job_name}-{index}.{job_name}.{namespace}.svc.cluster.local
示例:    train-round-1-0-12345-0.train-round-1-0-12345.flower-supernode-a.svc.cluster.local

所有训练 Pod 以 Pod-0 为 Master 建立 NCCL 进程组
```

### 4.4 训练 Pipeline (`distributed_trainer.py`)

```
1. setup_distributed()
   ├── 解析 MASTER_ADDR, WORLD_SIZE, rank 等
   ├── DNS 解析重试（最多 60 秒）
   └── dist.init_process_group(backend="nccl")

2. 加载模型和数据
   ├── get_tokenizer_and_data_collator() → tokenizer, DataCollator
   ├── load_data(partition_id) → 联邦数据分片
   └── get_model() → 4-bit 量化 + LoRA

3. 加载初始权重
   └── set_peft_model_state_dict(model, initial_weights)

4. DDP 包装
   ├── gradient_checkpointing_enable() ← 必须在 DDP 之前
   ├── model.to(cuda)
   └── DistributedDataParallel(model)

5. 数据预处理（TRL 0.8.1 兼容方案）
   ├── remove_columns(非 text 列)
   ├── StrippingDataCollator 包装
   └── SFTTrainer(dataset_text_field="text")

6. 训练
   └── trainer.train() → 10 steps, 3 epochs

7. 保存结果（仅 rank 0）
   ├── get_peft_model_state_dict() → LoRA 权重
   ├── torch.save → PVC
   └── metrics.json
```

### 4.5 LoRA 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| LoRA rank (r) | 32 | 低秩矩阵维度 |
| LoRA alpha | 64 | 缩放因子 |
| LoRA dropout | 0.075 | 正则化 |
| 量化 | 4-bit (NF4) | QLoRA 节省显存 |
| Gradient Checkpointing | True | 用计算换显存 |
| 学习率 | Cosine Annealing (5e-5 → 1e-6) | 随联邦轮次衰减 |
| Batch Size | 4 per GPU | 单 GPU batch |
| Sequence Length | 512 | 最大 token 数 |
| 训练步数 | 10 steps / round | 每轮快速微调 |

---

## 5. 数据处理

### 5.1 数据集格式（Alpaca）

```json
{
  "instruction": "描述任务的指令",
  "input": "可选的上下文输入",
  "output": "GPT-4 生成的回答",
  "text": "预格式化的完整文本"
}
```

`text` 字段格式:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response: {output}
```

### 5.2 联邦数据分片

```python
# IID 均匀分片
partitioner = IidPartitioner(num_partitions=N)
FDS = FederatedDataset(
    dataset="vicgalle/alpaca-gpt4",
    partitioners={"train": partitioner},
)
# 每个 SuperNode 获取 1/N 的数据
```

### 5.3 训练数据处理（TRL 0.8.1 兼容方案）

```
原始数据集: [instruction, input, response, text]
    ↓ remove_columns → [text]
    ↓ SFTTrainer._prepare_dataset(tokenize) → [text, input_ids, attention_mask]
    ↓ StrippingDataCollator(去除 text 字符串)
    ↓ DataCollatorForCompletionOnlyLM(padding + labels masking)
    ↓ [input_ids, attention_mask, labels] → 模型训练
```

**关键设计**：
- **DataCollatorForCompletionOnlyLM**: 查找 `### Response:` token，将其之前的 instruction 部分 mask 为 `-100`，只在 Response 部分计算 loss
- **StrippingDataCollator**: TRL 0.8.1 tokenize 后不移除原始 `text` 列，需要在 batch 级别去除字符串字段
- **remove_unused_columns=False**: DDP 包装的模型 forward 签名不包含 `input_ids`，需防止 Trainer 误删

---

## 6. 存储架构

### 6.1 PVC 设计

```
                    CFS (共享网络存储)
                    ┌──────────────────────────┐
                    │                          │
   flowertune-cache-pvc (50Gi, RWX)           │
   ├── huggingface/hub/                       │
   │   ├── models--openlm-research--.../      │
   │   │   ├── blobs/     (实际文件)           │
   │   │   └── snapshots/ (版本软链接)          │
   │   └── ...                                │
   └── (模型缓存，所有 Pod 共享)                │
                                              │
   flowertune-output-pvc (50Gi, RWX)          │
   └── {job_name}/                            │
       ├── initial_weights.pt  (SuperExec 写入)│
       ├── model_weights.pt    (训练 Pod 写入)  │
       └── metrics.json        (训练 Pod 写入)  │
                    └──────────────────────────┘
```

### 6.2 存储挂载

| 路径 | 类型 | 来源 | 用途 |
|---|---|---|---|
| `/app/.cache` | PVC | flowertune-cache-pvc | HuggingFace 模型/数据缓存 |
| `/app/outputs` | PVC | flowertune-output-pvc | 训练输入/输出 |
| `/app/.flwr` | emptyDir | - | Flower 运行时状态 |
| `/app/.config` | emptyDir | - | 配置临时目录 |

### 6.3 模型传递机制

```
SuperExec (ClientApp)                训练 Pod 0..N
      │                                    │
      │  1. 加载基座模型 + 全局 LoRA 权重     │
      │  2. 保存 initial_weights.pt → PVC   │
      │  3. sleep(5s) 等待 CFS 同步          │
      │                                    │
      │  ───── K8s Job 启动 ─────           │
      │                                    │
      │                    4. 从 PVC 读取 initial_weights.pt
      │                    5. 加载基座模型 + 设置 LoRA 权重
      │                    6. DDP 训练
      │                    7. rank 0 保存 model_weights.pt → PVC
      │                                    │
      │  ◄── Job 完成 ────                  │
      │                                    │
      │  8. 从 PVC 读取 model_weights.pt    │
      │  9. 返回更新后的权重给 ServerApp      │
```

---

## 7. Kubernetes 部署架构

### 7.1 命名空间规划

```
flower-supernode-a (集群 A)          flower-supernode-b (集群 B)
├── superlink (可选)                 ├── supernode-b
├── supernode-a                      ├── superexec-clientapp-b
├── superexec-clientapp-a            ├── flower-client-sa (RBAC)
├── flower-client-sa (RBAC)          ├── flowertune-cache-pvc (CFS)
├── flowertune-cache-pvc (CFS)       └── flowertune-output-pvc (CFS)
└── flowertune-output-pvc (CFS)
```

### 7.2 RBAC 权限

SuperExec Pod 通过 ServiceAccount 调用 K8s API 创建训练 Job：

| 资源 | 权限 | 用途 |
|---|---|---|
| `batch/jobs` | create, get, list, watch, delete | 创建/监控训练 Job |
| `services` | create, get, list, watch, delete | 创建 Headless Service (DNS) |
| `pods` | get, list, watch | 监控 Pod 状态 |
| `persistentvolumeclaims` | get, list | 读取 PVC 信息 |

### 7.3 部署依赖顺序

```
1. RBAC (ServiceAccount + Role + RoleBinding)
       ↓
2. CFS PVC (cache-pvc + output-pvc)
       ↓
3. SuperLink (联邦协调器)
       ↓
4. SuperNode (客户端节点)
       ↓
5. SuperExec Deployment (ClientApp)
```

### 7.4 训练 Job 生命周期

```
SuperExec 创建 Indexed Job
    │
    ├── 创建 Headless Service (DNS 解析)
    ├── 创建 N 个训练 Pod (GPU)
    │       │
    │       ├── DNS 解析重试 (最多 60s)
    │       ├── NCCL 进程组初始化
    │       ├── DDP 分布式训练
    │       └── Rank 0 保存结果到 PVC
    │
    ├── SuperExec 轮询 Job 状态 (每 30s)
    │
    └── Job 完成
        ├── 收集训练结果 (model_weights.pt)
        ├── 删除 Job
        └── 删除 Headless Service
```

---

## 8. 多集群兼容设计

### 8.1 环境变量驱动

同一份 `client_app.py` 代码通过环境变量适配不同集群：

| 环境变量 | 说明 | 来源 |
|---|---|---|
| `POD_NAMESPACE` | 当前命名空间 | K8s Downward API 自动注入 |
| `TRAIN_IMAGE` | 训练 Pod 镜像 | SuperExec Deployment 配置 |
| `CACHE_PVC` | 缓存 PVC 名称 | SuperExec Deployment 配置 |
| `OUTPUT_PVC` | 输出 PVC 名称 | SuperExec Deployment 配置 |
| `DIST_NUM_NODES` | 训练节点数 | SuperExec Deployment 配置 |
| `DIST_GPUS_PER_NODE` | 每节点 GPU 数 | SuperExec Deployment 配置 |

### 8.2 集群差异化配置示例

```yaml
# 集群 A (3×V100)
- name: TRAIN_IMAGE
  value: "ccr.ccs.tencentyun.com/flwr_pcl/flwr_client_train:0.0.19"
- name: DIST_NUM_NODES
  value: "3"
- name: DIST_GPUS_PER_NODE
  value: "1"

# 集群 B (2×A100)
- name: TRAIN_IMAGE
  value: "ccr.ccs.tencentyun.com/flwr_pcl/flwr_client_train:0.0.19"
- name: DIST_NUM_NODES
  value: "2"
- name: DIST_GPUS_PER_NODE
  value: "2"
```

---

## 9. 模型加载策略

### 9.1 SuperExec（有网络）

```
models.py: get_model()
├── 优先从本地缓存加载 (local_files_only=True)
├── 缓存损坏 → 自动清理并重试
├── 缓存不存在 → 多源下载 (hf-mirror.com → huggingface.co)
└── 使用 snapshot_download 控制下载流程
```

### 9.2 训练 Pod（无网络）

```
train_models.py: get_model()
├── 仅从本地缓存加载 (local_files_only=True)
├── 使用绝对缓存路径（绕过 Transformers 4.53.0 + CFS 兼容性问题）
└── 加载失败直接报错（不尝试下载）
```

### 9.3 CFS 存储兼容性处理

Transformers 4.53.0 + HuggingFace Hub 0.36.2 在 CFS 等网络文件系统上存在缓存识别问题：即使缓存文件完整且软链接有效，通过模型 ID 加载时仍会报错。

**解决方案**: 训练 Pod 直接使用缓存目录的绝对路径加载：
```
/app/.cache/huggingface/hub/models--openlm-research--open_llama_3b_v2/snapshots/{commit_hash}/
```

---

## 10. 关键文件清单

### 10.1 核心代码

| 文件 | 用途 | 运行环境 |
|---|---|---|
| `server_app.py` | ServerApp：FedAvg 聚合策略 | SuperLink |
| `client_app.py` | ClientApp：调度 K8s 训练 Job | SuperExec |
| `distributed_trainer.py` | DDP 分布式训练入口 | 训练 Pod |
| `models.py` | 模型加载（有网络，多源下载） | SuperExec |
| `train_models.py` | 模型加载（无网络，仅缓存） | 训练 Pod |
| `train_dataset.py` | Tokenizer + DataCollator + 数据加载 | 训练 Pod |
| `dataset.py` | Tokenizer + DataCollator + 数据加载 | SuperExec |

### 10.2 部署配置

| 文件 | 用途 |
|---|---|
| `superlink-deployment.yaml` | SuperLink 部署 |
| `supernode-{a,b}-deployment.yaml` | SuperNode 部署 |
| `superexec-{a,b}-deployment.yaml` | SuperExec 部署（含集群特定配置） |
| `flower-client-rbac{-b}.yaml` | RBAC 权限配置 |
| `flowertune-cfs-pvc{-b}.yaml` | CFS 持久化存储 |
| `supernode-{a,b}-svc.yaml` | SuperNode Service |

### 10.3 构建文件

| 文件 | 用途 |
|---|---|
| `superexec.Dockerfile` | SuperExec 镜像（含 ClientApp 代码） |
| `Dockerfile.multinode` | 训练 Pod 镜像（含训练代码 + PyTorch + GPU 依赖） |
| `pyproject.toml` | Python 依赖 + Flower App 配置 |

---

## 11. 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|---|---|---|
| TRL 0.8.1 tokenize 后不移除原始列 | `_prepare_dataset` 不自动 `remove_columns` | 预处理移除 + StrippingDataCollator |
| DDP forward 签名不匹配 | DDP 包装后 `forward()` 返回通用签名 | `remove_unused_columns=False` |
| Gradient Checkpointing 报错 | DDP 包装后无法调用 | 在 DDP 包装前启用 |
| CFS 缓存识别失败 | Transformers 4.53.0 软链接检测问题 | 使用绝对缓存路径加载 |
| DNS 解析超时 | K8s Service/Pod 创建延迟 | 重试机制（最多 60 秒） |
| `packing=False` 校验失败 | TRL 0.8.1 强制要求 `dataset_text_field` | 传入 `dataset_text_field="text"` |
