# 测试数据

本目录包含联邦微调实验使用的医疗闪卡数据集。

## 文件列表

| 文件 | 大小 | 说明 |
|---|---|---|
| `medical_flashcards_train.json` | 16 MB | 训练集（30176 条） |
| `medical_flashcards_eval.json` | 1.7 MB | 评估集（3352 条） |
| `dataset_info.json` | 634 B | LLaMA-Factory 数据集注册信息 |

## 数据来源

- 原始数据集：`medalpaca/medical_meadow_medical_flashcards`（HuggingFace）
- 仅 HuggingFace 有，ModelScope 无此数据集
- 已预处理为 alpaca 格式 JSON 文件

## 数据格式

```json
{
  "instruction": "问题文本",
  "input": "",
  "output": "答案文本"
}
```

详见 [deployment/data-preparation.md](../deployment/data-preparation.md)。
