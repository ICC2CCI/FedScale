# ICC1 本地路径约定

| 资源 | 路径 |
|---|---|
| 仓库根目录 | `/home/pcl/liuchao/fedscale-icc-1` |
| 模型 | `model/Qwen/Qwen2.5-0.5B` |
| 完整训练数据 | `data/medical_flashcards_train.json` |
| 评估数据 | `data/medical_flashcards_eval.json` |
| ICC1 训练划分 (client-0) | `data/splits/icc1_client0_train.json` |
| 节点角色 | ICC1 = Client 0 |

启动示例（FSDP 脚本就绪后）：

```bash
accelerate launch --config_file accelerate_config.yaml \
  experiments/run_s3r12v3_fsdp.py \
  --client-id 0 \
  --model-path model/Qwen/Qwen2.5-0.5B \
  --data-path data/splits/icc1_client0_train.json
```
