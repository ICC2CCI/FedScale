# 联邦数据划分（双集群）

## 医学闪卡（当前联调用）

| 文件 | 说明 |
|---|---|
| `icc1_client0_train.json` | ICC1 / Client 0（15088 条） |
| `client1_train.json` | ICC2 / Client 1（15088 条） |
| `icc1_manifest.json` | ICC1 划分元信息 |
| `icc2_manifest.json` | ICC2 划分元信息 |

划分参数：`seed=20260831`，训练集 50/50（与双集群部署文档一致）。

完整训练集 / 评估集仍在上级目录（若本机有）：

- `../medical_flashcards_train.json`
- `../medical_flashcards_eval.json`

## Dolly-15k（DATA-D2，审稿人预期 Non-IID）

源：`../dolly_15k_train.json`（13510 条）。共享评估：`../dolly_15k_eval.json`（1501 条，seed=20260831）。  
按真正的 `category` 做 Dirichlet，不是闪卡文本前缀。复现：

```bash
python scripts/split_federated_data.py \
  --train data/dolly_15k_train.json --eval data/dolly_15k_eval.json \
  --out-dir data/splits/dolly-iid --method uniform --seed 20260831 --label-field category

python scripts/split_federated_data.py \
  --train data/dolly_15k_train.json --eval data/dolly_15k_eval.json \
  --out-dir data/splits/dolly-dirichlet-a0.5 --method dirichlet --alpha 0.5 \
  --seed 20260831 --label-field category
```

| 目录 | 切法 | client0 | client1 |
|---|---|---|---|
| `dolly-iid/` | uniform 50/50 | 6755 | 6755 |
| `dolly-dirichlet-a0.5/` | Dirichlet α=0.5 by category | 6314 | 7196 |

两端 **eval 都用** `data/dolly_15k_eval.json`。类别直方图见各目录 `federated_split_manifest.json`。
