# ICC2 节点准备状态（Client 1）

- **主机**: dgx1-30
- **角色**: ICC2 / Client 1（8× V100 32GB）
- **状态**: 已联调；SecAgg / 非 SecAgg 双集群实验可直接跑

## 已完成

| 项 | 路径 / 说明 |
|---|---|
| ICC2 训练数据 | `data/splits/client1_train.json`（15088 条，官方 50/50 划分） |
| 共享评估集 | `data/medical_flashcards_eval.json`（3352 条） |
| 划分元信息 | `data/splits/manifest.json` |
| 旧数据备份 | `data_/`（此前本机切分结果，可忽略） |
| 基础模型 | `model/Qwen/Qwen2.5-0.5B`（Qwen2.5-0.5B） |
| Conda 环境 | `flwr-ft`（Python 3.10，torch 2.8.0+cu128） |
| FSDP 配置 | `configs/accelerate/accelerate_config.yaml`（或仓库根 `accelerate_config.yaml`） |
| GPU 验证 | 8× Tesla V100-SXM2-32GB-LS；模型 fp16 加载成功 |

## 激活环境

```bash
source /home/pcl/miniconda3/bin/activate flwr-ft
cd /home/pcl/liuchao/fedscale-icc-2
```

推荐用 Central 侧一键脚本拉起（见根 `README.md`），或本机：

```text
--client-id 1
--model-path model/Qwen/Qwen2.5-0.5B
--data-path data/splits/client1_train.json
--eval-path data/medical_flashcards_eval.json
```