# 模型下载

本实验使用 Qwen2.5-0.5B 作为基础模型，通过 ModelScope 下载。

## 1. 下载脚本

```python
from modelscope import snapshot_download

model_path = snapshot_download(
    'Qwen/Qwen2.5-0.5B',
    cache_dir='/data/models'
)
print(f"Model downloaded to: {model_path}")
```

## 2. 运行下载

```bash
python -c "
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir='/data/models')
print(f'Downloaded to: {path}')
"
```

## 3. 验证模型

```bash
# 检查模型文件
ls -lh /data/models/Qwen/Qwen2.5-0.5B/

# 应包含：
# config.json
# model.safetensors (约 1GB)
# tokenizer.json
# tokenizer_config.json
# ...
```

```python
# 验证可加载
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/models/Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Model: {model.config.model_type}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Layers: {model.config.num_hidden_layers}")
```

预期输出：

```
Model: qwen2
Parameters: 494.0M
Layers: 24
```

## 4. 模型信息

| 属性 | 值 |
|---|---|
| 模型名 | Qwen2.5-0.5B |
| 参数量 | 494M（float），630M（含 embedding） |
| 层数 | 24 transformer layers |
| 隐藏维度 | 896 |
| dtype | bf16（联邦训练）/ fp16（全量训练） |
| 下载来源 | ModelScope |
| 目标路径 | `/data/models/Qwen/Qwen2.5-0.5B` |

## 5. 其他可用模型

如需更换模型，同样从 ModelScope 下载：

```python
# Qwen3 系列
snapshot_download('Qwen/Qwen3-8B', cache_dir='/data/models')
```

## 6. 注意事项

- **必须用 ModelScope**（不要用 HuggingFace），国内下载更稳定
- 模型路径需与实验脚本中的 `MODEL_PATH` 一致
- 如路径不同，修改脚本中的 `MODEL_PATH` 变量
- bf16 精度下模型约 1GB，确保磁盘空间充足
