# 数据准备

本实验使用 `medalpaca/medical_meadow_medical_flashcards` 数据集（医疗闪卡问答）。

## 1. 数据已包含在仓库中

数据文件已随仓库提交，无需额外下载：

```
data/
├── medical_flashcards_train.json   # 30176 条训练样本 (16 MB)
├── medical_flashcards_eval.json    # 3352 条评估样本 (1.7 MB)
└── dataset_info.json               # LLaMA-Factory 数据集注册信息
```

## 2. 数据格式

每条数据格式如下：

```json
{
  "instruction": "问题文本",
  "input": "",
  "output": "答案文本"
}
```

示例：

```json
{
  "instruction": "What is the mechanism of action of beta-blockers?",
  "input": "",
  "output": "Beta-blockers competitively block beta-adrenergic receptors, decreasing heart rate, myocardial contractility, and blood pressure."
}
```

## 3. 数据统计

| 统计项 | 训练集 | 评估集 |
|---|---|---|
| 样本数 | 30176 | 3352 |
| 文件大小 | 16 MB | 1.7 MB |
| 格式 | JSON（alpaca 格式） | JSON（alpaca 格式） |
| 平均长度 | ~200 tokens | ~200 tokens |

## 4. 从 HuggingFace 重新下载（可选）

如需从原始来源重新下载：

```python
from datasets import load_dataset

ds = load_dataset("medalpaca/medical_meadow_medical_flashcards")
# 转为 alpaca 格式并保存
```

## 5. 联邦数据划分

联邦训练中，训练集被划分为 2 个客户端：

```python
# 在实验脚本中自动完成
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/medical_flashcards_train.json")
split = dataset["train"].train_test_split(test_size=0.5, seed=20260831)
client_0_data = split["train"]  # 15088 条
client_1_data = split["test"]   # 15088 条
```

| 客户端 | 样本数 | 占比 |
|---|---|---|
| client 0 | 15088 | 50% |
| client 1 | 15088 | 50% |

## 6. 用于 LLaMA-Factory（S1-D 全量训练）

如需运行 S1-D 全量训练基线，需要将数据注册到 LLaMA-Factory：

```bash
# data/dataset_info.json 已包含注册信息
# 确保 LLaMA-Factory 的 dataset_dir 指向 data/ 目录
```

`dataset_info.json` 内容：

```json
{
  "medical_flashcards": {
    "file_name": "medical_flashcards_train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```
