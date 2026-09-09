# 故障排查

## 1. CUDA out of memory

**现象**：

```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**解决**：

```python
# 减小 batch size
per_device_train_batch_size=4,
gradient_accumulation_steps=4,

# 启用 gradient checkpointing
gradient_checkpointing=True,

# 确保使用 bf16
bf16=True,
```

## 2. gradient_checkpointing 报错

**现象**：

```
RuntimeError: gradient_checkpointing requires use_cache=False
```

或 FedRolex 中：

```
RuntimeError: Parameter at index ... does not require grad
```

**解决**：

```python
# 方案 1：关闭 gradient_checkpointing
gradient_checkpointing=False,

# 方案 2：显式设置 use_cache=False
model.config.use_cache = False
```

FedRolex 实验必须用 `gradient_checkpointing=False`。

## 3. 模型加载失败

**现象**：

```
OSError: Can't load the model for '/data/models/Qwen/Qwen2.5-0.5B'.
```

**解决**：

```bash
# 检查路径
ls -la /data/models/Qwen/Qwen2.5-0.5B/

# 应有 config.json 和 model.safetensors
# 如缺失，重新下载
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir='/data/models')"
```

## 4. 数据加载失败

**现象**：

```
FileNotFoundError: medical_flashcards_train.json not found
```

**解决**：

```bash
# 确认数据文件存在
ls data/medical_flashcards_train.json data/medical_flashcards_eval.json

# 如缺失，从仓库重新拉取
git pull origin main
```

## 5. Flower 版本不兼容

**现象**：

```
ImportError: cannot import name 'ServerApp' from 'flwr'
```

**解决**：

```bash
pip install flwr==1.18.0 flwr-sim==1.18.0
```

## 6. TRL 版本不兼容

**现象**：

```
ImportError: cannot import name 'SFTTrainer' from 'trl'
```

**解决**：

```bash
pip install trl==0.8.6
```

## 7. 收敛异常

**现象**：eval loss 不下降或上升

**排查**：

1. 确认 learning_rate = 1e-5（太大不收敛，太小不动）
2. 确认 bf16 = True（fp32 不必要，fp16 可能数值不稳定）
3. 确认 response template 正确：`<|im_start|>assistant\n`
4. 确认 DataCollatorForCompletionOnlyLM 正确 mask 了 input 部分

## 8. 上传比例异常

**现象**：S3R12v3 上传比例不在 18.5%~21.5% 范围内

**排查**：

1. 确认 BLOCK_SIZE = 524288
2. 确认 H = 5（20% ratio）
3. 确认 non_layer 组参与了轮转（不是 always_on）
4. 查看 round_log.json 中的 `upload_ratio` 字段
