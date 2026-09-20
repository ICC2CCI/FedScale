# 测试数据

本目录包含联邦微调实验使用的数据集。

## 文件列表

| 文件 | 大小 | 说明 |
|---|---|---|
| `medical_flashcards_train.json` | 16 MB | 医学闪卡训练集（30176 条） |
| `medical_flashcards_eval.json` | 1.7 MB | 医学闪卡评估集（3352 条） |
| `dolly_15k.json` | ~13 MB | Dolly-15k 全量（15011 条，含 `category`；gitignore） |
| `dolly_15k_train.json` / `dolly_15k_eval.json` | | 90/10，seed=20260831（DATA-D1） |
| `splits/dolly-iid/` | | 2 client IID 50/50（DATA-D2） |
| `splits/dolly-dirichlet-a0.5/` | | 2 client、按 category Dirichlet α=0.5（DATA-D2） |
| `dataset_info.json` | 634 B | LLaMA-Factory 数据集注册信息 |

## 数据来源

- 医学：HuggingFace `medalpaca/medical_meadow_medical_flashcards`
- Dolly：HuggingFace `databricks/databricks-dolly-15k`（脚本：`python scripts/prepare_dolly_15k.py`）

## 数据格式

```json
{
  "instruction": "问题文本",
  "input": "",
  "output": "答案文本"
}
```

详见 [deployment/data-preparation.md](../deployment/data-preparation.md)。
