# ICC2 节点准备状态（Client 1）

- **主机**: dgx1-30
- **角色**: ICC2 / Client 1（8× V100 32GB）
- **状态**: 本机数据/模型/环境已就绪；等待 Central Server + 训练脚本联调

## 已完成

| 项 | 路径 / 说明 |
|---|---|
| ICC2 训练数据 | `data/splits/client1_train.json`（15088 条，官方 50/50 划分） |
| 共享评估集 | `data/medical_flashcards_eval.json`（3352 条） |
| 划分元信息 | `data/splits/manifest.json` |
| 旧数据备份 | `data_/`（此前本机切分结果，可忽略） |
| 基础模型 | `model/Qwen/Qwen2.5-0.5B`（Qwen2.5-0.5B，~943MB） |
| Conda 环境 | `flwr-ft`（Python 3.10，torch 2.8.0+cu128） |
| FSDP 配置 | `configs/accelerate/accelerate_config.yaml`（8 卡 fp16） |
| GPU 验证 | 8× Tesla V100-SXM2-32GB-LS，CUDA driver 13.0，模型 fp16 加载成功 |

## 激活环境

```bash
source /home/pcl/miniconda3/bin/activate flwr-ft
cd /home/pcl/liuchao/fedscale-icc-2
```

## 推荐路径参数（ICC2）

```text
--client-id 1
--model-path model/Qwen/Qwen2.5-0.5B
--data-path data/splits/client1_train.json
--eval-path data/medical_flashcards_eval.json
```
