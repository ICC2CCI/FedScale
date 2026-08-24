# flwrtune-llm

Flower框架下大模型联邦训练 + **多机多卡分布式训练**

当前集群的实验预检、启动、恢复与结果说明见
[联邦实验运行与运维指南](docs/联邦实验运行与运维指南.md)。

## 🚀 新特性：跨节点多机多卡分布式训练

本项目现已支持**真正的跨Kubernetes节点多机多卡分布式训练**！

### 架构特点

- ✅ Flower Client作为K8s Job调度器
- ✅ 跨多个物理节点的PyTorch DDP训练
- ✅ 保持Flower联邦学习架构不变
- ✅ 灵活的GPU资源扩展（理论上无上限）
- ✅ 生产级的容错和监控

### 快速开始

```bash
# 1. 构建Docker镜像
cd flowertune-llm
docker build -f Dockerfile.multinode -t <your-image> .
docker push <your-image>

# 2. 配置RBAC权限
kubectl apply -f configs/flower-client-rbac.yaml

# 3. 部署多机训练Client
kubectl apply -f configs/clientapp-multinode-deployment.yaml

# 4. 查看训练状态
kubectl logs -f deployment/clientapp-multinode -n flower-supernode-a
```

- 📖 **完整文档**: [联邦实验运行与运维指南](docs/联邦实验运行与运维指南.md)
- 🏗️ **实现说明**: [DESIGN.md](DESIGN.md)

---

## 项目概述

**flwrtune-llm** 是基于 Flower 框架的大模型联邦训练系统，支持：

- 🤝 大模型联邦学习训练
- 🌐 跨节点多机多卡分布式训练
- ⚡ PEFT/LoRA 高效微调
- 🔢 4-bit/8-bit 量化训练
- ☸️ Kubernetes 原生部署

## 两种训练模式

### 模式1: 单Pod多GPU (client_app.py)

**适用场景**: 单节点内多GPU训练（≤8卡）

```bash
kubectl apply -f configs/clientapp-a-deployment.yaml
```

**特点**:
- 简单快速部署
- 无需额外权限
- 适合中小模型

### 模式2: 多节点多GPU (client_app_multinode.py) ⭐推荐

**适用场景**: 跨节点大规模训练（>8卡）

```bash
kubectl apply -f configs/flower-client-rbac.yaml
kubectl apply -f configs/clientapp-multinode-deployment.yaml
```

**特点**:
- 跨节点扩展
- 充分利用集群资源
- 适合大规模模型

## 项目结构

```
flowertune-llm/
├── flowertune_llm/
│   ├── client_app.py              # 单Pod多GPU训练
│   ├── client_app_multinode.py    # 多节点多GPU训练 (NEW)
│   ├── distributed_trainer.py     # 分布式训练执行器 (NEW)
│   ├── server_app.py              # Server聚合逻辑
│   ├── models.py                  # 模型定义
│   └── dataset.py                 # 数据处理
├── configs/
│   ├── flower-client-rbac.yaml           # RBAC权限 (NEW)
│   ├── clientapp-multinode-deployment.yaml # 多节点部署 (NEW)
│   └── ...                        # 其他配置
├── Dockerfile.multinode           # 多节点训练镜像 (NEW)
├── pyproject.toml                 # 项目依赖
├── MULTINODE_DEPLOYMENT_GUIDE.md  # 部署指南 (NEW)
└── MULTINODE_IMPLEMENTATION.md    # 实现说明 (NEW)
```

## 技术栈

- **Flower**: 联邦学习框架
- **PyTorch**: 深度学习框架
- **Kubernetes**: 容器编排和调度
- **PEFT/LoRA**: 参数高效微调
- **TRL**: Transformer强化学习库
- **BitsAndBytes**: 量化训练
- **HuggingFace Transformers**: 模型库

## 依赖

主要依赖见 `pyproject.toml`，包括：
- flwr >= 1.28.0
- torch == 2.8.0
- kubernetes >= 28.1.0 (新增)
- transformers == 4.53.0
- peft == 0.6.2
- trl == 0.8.1

## License

Apache-2.0
