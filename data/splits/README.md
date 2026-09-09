# 联邦数据划分（双集群）

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
