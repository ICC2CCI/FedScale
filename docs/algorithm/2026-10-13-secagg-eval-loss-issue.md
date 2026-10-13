# SecAgg eval_loss 计算问题与修复方案

> 日期：2026-10-13
> 项目：SecAgg / S3R12v3（现有方案）
> 状态：**已确认问题，待修复**

## 问题描述

### 当前做法

`experiments/server/aggregation_server.py:805-844`：

```python
eval_losses = [float(client_meta[c]["eval_loss"]) for c in participated
               if client_meta[c].get("eval_loss") is not None]
avg_eval = (sum(eval_losses) / len(eval_losses)) if eval_losses else None
```

server 端取各 client 上报的 `eval_loss` 的**算术平均**作为该轮的 eval_loss。

### client 端 eval 时机

`experiments/run_s3r12v3_fsdp.py:1365-1380`：

```python
# 在线 eval（本地训练后），全 ranks 参与 FSDP 前向
eval_loss_val = eval_local_batches(model, eval_loader, ...)
```

eval 在**本地训练后、上传前**执行，此时模型是**本地训练后的模型**，不是聚合后的全局模型。

### 问题

1. **eval 的不是全局模型**——评估的是各 client 本地训练后的模型，不反映联邦聚合效果
2. **取均值无明确语义**——虽然两个 client 都用同一评估集 `medical_flashcards_eval.json`，但各自评估的是不同的本地模型，均值不代表任何单一模型的性能
3. **画图判断收敛有偏差**——eval_loss 曲线反映的是"各 client 本地模型平均 loss"而非"全局模型 loss"，可能误判收敛趋势

## 修复方案

### 方案：把 eval 移到下一轮开始时（下载全局模型后、训练前）

**当前流程**（每轮）：
```
下载全局模型 → 本地训练 → eval（本地模型） → 上传 delta
```

**修复后流程**（每轮）：
```
下载全局模型 → eval（全局模型） → 本地训练 → 上传 delta
```

这样 Round N 的 eval_loss = Round N-1 聚合后全局模型在评估集上的 loss，真实反映联邦聚合效果。

### 具体改动

**文件**：`experiments/run_s3r12v3_fsdp.py`

1. 把 eval 代码块（约 1365-1380 行）从「训练后」移到「下载全局模型后、训练前」
2. Round 0 没 eval（初始模型 baseline 可单独跑，或在 Round 0 下载初始模型后 eval 一次）
3. eval 结果上报到 server 的字段保持 `eval_loss`，但语义变为「上一轮聚合后全局模型的 eval_loss」

**文件**：`experiments/server/aggregation_server.py`

4. 第 840 行 `avg_eval` 的计算保持不变（取均值），但因为两个 client 下载的全局模型相同、评估集相同，所以两个 client 的 eval_loss **应该完全一致**，均值 = 单值
5. 可选：改为只取 client 0 的 eval_loss（避免重复计算），但 2-client 下无实际差异

### 注意事项

- eval 会在每轮增加约 10-30s（取决于 eval_max_batches 设置）
- 现有 `--eval-every-n-rounds` 参数仍可控制 eval 频率
- Round 0 的 eval_loss 为 None（无上一轮全局模型），画图时从 Round 1 开始
- SecAgg 和非 SecAgg 路径都适用（eval 时机与聚合方式无关）

## TODO

- [ ] 修改 `experiments/run_s3r12v3_fsdp.py`：把 eval 从训练后移到下载全局模型后
- [ ] 验证 Round 0 处理（无全局模型时跳过 eval）
- [ ] 跑一轮 SecAgg smoke test 确认 eval_loss 语义正确
- [ ] 更新相关文档（`docs/algorithm/2026-09-*-secagg-*.md`）说明 eval_loss 语义变更
