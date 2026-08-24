已经统一参数并完成实测。结果满足目标：在客户端关键路径上，FSDP 比 DDP 快约 5.94%，只需要 1 个联邦轮次，不再需要用“约 20 轮后才能追上”来论述。

### 正式对照参数

| 参数 | FSDP | DDP |
|---|---:|---:|
| 模型 | OpenLLaMA-3B v2 | 相同 |
| 微调方式 | 全参数微调 | 相同 |
| 客户端 | 2 | 相同 |
| 每客户端 GPU | 3×V100 | 相同 |
| 序列长度 | 512 | 相同 |
| per-device batch | 1 | 相同 |
| 梯度累积 | 1 | 相同 |
| 每客户端有效 batch | 3 | 相同 |
| optimizer steps | 10 | 相同 |
| 学习率 | 1e-6 | 相同 |
| 梯度检查点 | 开启 | 相同 |
| 每客户端数据上限 | 3000 | 相同 |
| 联邦轮次 | 1 | 相同 |
| 参数传输 | Top-K INT8 | 相同 |
| Top-K ratio | 0.001 | 相同 |
| 本地首轮初始化 | 开启 | 相同 |
| 分布策略 | FSDP FULL_SHARD | DDP |
| CPU optimizer offload | 否 | 是，DDP 在当前内存配置下运行全参微调所必需 |

`batch=1` 是每张 GPU 的 micro-batch；每个客户端有3张卡，所以每次优化的客户端有效 batch 是3，并不是整个实验只处理一个样本。

### 实测结果

| 指标 | FSDP | DDP | FSDP 优势 |
|---|---:|---:|---:|
| 客户端关键路径训练阶段 | 572.94s | 649.34s | 快 76.40s，约 11.77% |
| 客户端关键路径整轮耗时 | 858.49s | 912.70s | 快 54.21s，约 5.94% |
| 两客户端平均整轮耗时 | 772.98s | 784.91s | 快 11.94s |
| 平均训练 loss | 1.3432 | 1.3460 | 基本一致 |
| 联邦轮次 | 1 | 1 | 无需多轮摊销 |

具体来看：

- A 端是本轮关键路径：FSDP 858.49 秒，DDP 912.70 秒，FSDP 明显更快。
- B 端：FSDP 训练本身更快，但 FSDP 的初始化/状态导出开销较高，所以该客户端整轮 DDP 略快。
- 最终由较慢的 A 端决定联邦轮次完成时间，因此整轮仍是 FSDP 胜出。

正式实验：

- FSDP：`fsdp-full-topk-3b-s512-s10-gc-pair-r1`
  - Run ID：`15867855924407388040`
  - 状态：完成
- DDP：`ddp-full-cpuoffload-topk-3b-s512-s10-gc-pair-r2`
  - Run ID：`10884751994496698011`
  - 状态：完成

DDP 的 Run 总时长显示为 2425 秒，但其中约25分钟是在等待上一版失败任务释放 GPU，不能与 FSDP 的独占总时长直接比较。因此这里采用实验摘要中不受排队影响的 `critical_path_client_round_seconds`。

另外，DDP 的两个问题也已修复：

- FP16 CPU AdamW 的 `eps=1e-8` 下溢导致 `grad_norm=NaN`，已改为 `1e-4`：[cpu_offload_optimizer.py](../flowertune-llm/flowertune_llm/cpu_offload_optimizer.py#L21)
- 训练后导出全模型时，CPU optimizer 状态未释放导致 OOM，现已在导出前释放：[distributed_trainer.py](../flowertune-llm/flowertune_llm/distributed_trainer.py#L705)

修复后 DDP 两端 `grad_norm` 分别为 `26.6875` 和 `23.8125`，三张卡全部正常完成、模型成功导出并生成联邦 checkpoint。

### 复现实验命令

FSDP：

```bash
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp \
  --dataset vicgalle/alpaca-gpt4 \
  --finetuning-type full \
  --quantization 0 \
  --full-local-init \
  --rounds 1 \
  --experiment-id fsdp-full-topk-3b-s512-s10-gc-pair-r1 \
  --set "train.full-update-compression='topk-int8'" \
  --set train.full-update-topk-ratio=0.001 \
  --set dataset.max-train-samples=3000 \
  --set train.seq-length=512 \
  --set model.gradient-checkpointing=true \
  --set train.training-arguments.gradient-checkpointing=true \
  --set train.training-arguments.max-steps=10 \
  --set train.training-arguments.per-device-train-batch-size=1 \
  --set train.training-arguments.gradient-accumulation-steps=1 \
  --set train.training-arguments.learning-rate=1e-6 \
  --set train.evaluate-after-fit=false
```

FSDP_10：

```bash
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp \
  --dataset vicgalle/alpaca-gpt4 \
  --finetuning-type full \
  --quantization 0 \
  --full-local-init \
  --rounds 10 \
  --experiment-id fsdp-full-none-3b-s512-s10-gc-pair-10r-r1 \
  --set "train.full-update-compression='none'" \
  --set dataset.max-train-samples=3000 \
  --set train.seq-length=512 \
  --set model.gradient-checkpointing=true \
  --set train.training-arguments.gradient-checkpointing=true \
  --set train.training-arguments.max-steps=10 \
  --set train.training-arguments.per-device-train-batch-size=1 \
  --set train.training-arguments.gradient-accumulation-steps=1 \
  --set train.training-arguments.learning-rate=1e-6 \
  --set train.evaluate-after-fit=false \
```


DDP：

```bash
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy ddp \
  --dataset vicgalle/alpaca-gpt4 \
  --finetuning-type full \
  --quantization 0 \
  --ddp-cpu-offload \
  --full-local-init \
  --rounds 1 \
  --experiment-id ddp-full-cpuoffload-topk-3b-s512-s10-gc-pair-r2 \
  --set "train.full-update-compression='topk-int8'" \
  --set train.full-update-topk-ratio=0.001 \
  --set dataset.max-train-samples=3000 \
  --set train.seq-length=512 \
  --set model.gradient-checkpointing=true \
  --set train.training-arguments.gradient-checkpointing=true \
  --set train.training-arguments.max-steps=10 \
  --set train.training-arguments.per-device-train-batch-size=1 \
  --set train.training-arguments.gradient-accumulation-steps=1 \
  --set train.training-arguments.learning-rate=1e-6 \
  --set train.evaluate-after-fit=false
```
