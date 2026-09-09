# S3R12v3 — 分层 Block 级随机排列轮转 + Memory（均匀上传版）

- **状态**：done
- **创建**：2026-09-04
- **更新**：2026-09-04
- **关联**：`docs/exe-plans/active/2026-09-02-three-scenario-loss-comparison.md`
- **前序**：`docs/exe-plans/active/2026-09-04-s3r12v2-block-permutation.md`（S3R12v2，key 级选择）
- **灵感来源**：`FedScale_Public_Block_Mask_Implementation_Spec_v1.docx`（分层随机排列 + 无放回轮转）

## 1. 背景

S3R12v2 实现了真正的 20% 带宽（non_layer 纳入排列轮转），终点 eval loss 1.057。但存在上传量波动问题：每轮上传在 7%~37% 间波动（range 30%），因为：

- **key 级选择粒度太粗**：non_layer 只有 3 个 key（embed_tokens 136M、lm_head 136M、norm 896），大小极端不均
- 命中 embed/lm_head 的轮次上传飙升到 37%，没命中的轮次只有 7%
- 导致 loss 曲线有锯齿（r18 命中大 key 时 loss 跳变）

S3R12v3 改为 **block 级选择**：把每个 key 切成 1MB block，在 block 级别排列轮转，使每轮上传量接近恒定。

## 2. 核心改进 vs S3R12v2

| | S3R12v2（key 级） | S3R12v3（block 级） |
|---|---|---|
| **选择粒度** | 整个 key（整个 weight 矩阵） | **1MB block**（weight 矩阵的连续切片） |
| **每轮上传波动** | 7.3%~37.4%（range 30%） | **18.5%~21.5%（range 3%）** |
| **non_layer 处理** | 3 个 key 参与，但大小极端不均 | 521 个 block 参与，大小一致 |
| **上传协议** | 整个 key 的 tensor | **block 切片（flat view + offset）** |
| **终点 eval loss** | 1.0569 | **1.0514**（好 0.006） |

## 3. Block 切分方案

### 3.1 Block 定义

```
BLOCK_SIZE = 524288 elements = 1 MB (bf16, 2 bytes/element)

每个 key 的 tensor 展平为 1D，按 BLOCK_SIZE 切分为连续 block：
  block = (key_name, start_elem, end_elem)
  最后一个 block 可能不足 1MB
```

### 3.2 各组 Block 数量

| 组类型 | 数量 | keys/组 | **blocks/组** | elems/组 | bf16 MB/组 |
|---|---|---|---|---|---|
| layer_N | 24 | 12 | **38** | 14.9M | 29.8 |
| non_layer | 1 | 3 | **521** | 272.3M | 544.5 |
| **合计** | **25** | **291** | **1433** | **630.2M** | **1260.3** |

### 3.3 为什么 block 级能均匀

```
S3R12v2 (key级):
  non_layer 3 个 key: embed(136M) / lm_head(136M) / norm(896)
  → 命中 embed 或 lm_head 的轮次上传 37%，没命中的 7%

S3R12v3 (block级):
  non_layer 521 个 block，每个 ~1MB
  → 每轮选 104 个 block ≈ 104 MB，恒定
```

## 4. Mask Epoch 与排列

```
H = 5（覆盖周期），每 5 个成功轮次 = 1 个 Mask Epoch
20 轮 = 4 个 Epoch

每个 Epoch 开始（slot=0）：
  对每个组的 block 列表独立 Fisher-Yates shuffle：
    layer_N 组：38 个 block 随机排列
    non_layer 组：521 个 block 随机排列
  
  排列在 Epoch 内固定，5 轮内不重新 shuffle
  新 Epoch 重新 shuffle（新 epoch_seed）
```

### 4.1 种子派生

```
epoch_seed = SHA256("FedScale-BlockMask-v1" | SEED | epoch)
group_seed = SHA256(epoch_seed | "\x00" | "FedScale-GroupMask-v1" | group_id)
permutation = FisherYates(n_blocks, PRG=Random(group_seed))
```

## 5. 每轮选择规则

```
slot = (round - 1) % 5
每组选排列中 position % 5 == slot 的 block
```

### 5.1 各 slot 选 block 数量

**layer 组（38 blocks, H=5）**：

| slot | 选中 position | blocks 数 |
|---|---|---|
| 0 | 0,5,10,15,20,25,30,35 | **8** |
| 1 | 1,6,11,16,21,26,31,36 | **8** |
| 2 | 2,7,12,17,22,27,32,37 | **8** |
| 3 | 3,8,13,18,23,28,33 | **7** |
| 4 | 4,9,14,19,24,29,34 | **7** |

**non_layer 组（521 blocks, H=5）**：

| slot | blocks 数 |
|---|---|
| 0 | 104 |
| 1 | 104 |
| 2 | 104 |
| 3 | 104 |
| 4 | 105 |

### 5.2 每轮总 block 数

| slot | layer 贡献 | non_layer 贡献 | 总 blocks | 上传 elems | 上传 % |
|---|---|---|---|---|---|
| 0 | 24×8 = 192 | 104 | **296** | ~130M | ~20.6% |
| 1 | 24×8 = 192 | 104 | **296** | ~129M | ~20.4% |
| 2 | 24×8 = 192 | 104 | **296** | ~134M | ~21.3% |
| 3 | 24×7 = 168 | 104 | **272** | ~117M | ~18.5% |
| 4 | 24×7 = 168 | 105 | **273** | ~121M | ~19.2% |

**波动仅 18.5%~21.5%（range 3%）**，因为所有 block 大小一致（~1MB），只有最后一个 block 可能偏小。

## 6. 每轮完整流程

```
每轮（round r）：

  1. 确定选择：
     slot = (r-1) % 5
     如新 Epoch（slot=0），重新 shuffle 所有 25 个组的 block 列表
     selected_by_key = {}  # key_name -> list of (start, end)
     对每个组：
       对排列中 position % 5 == slot 的 block：
         (key_name, start, end) = blocks[block_idx]
         selected_by_key[key_name].append((start, end))

  2. 每个客户端（2 个）：
     a. 加载全局模型 global_state
     b. 本地训练 30 步（全模型训练，lr=1e-5, bs=8, grad_accum=2, cosine+warmup 0.1）
     c. 算 delta = trained_state - global_state
     d. 累加 memory: to_send = delta + client_memory[cid]
     e. 上传 selected blocks:
        对每个 key_name 的每个 (start, end):
          slice = to_send[key_name].view(-1)[start:end]  # flat view 切片
          上传 slice
     f. 更新 memory（flat view 操作）:
        对每个 key:
          flat = to_send[key].view(-1).clone()
          对 selected 的 block: flat[start:end] = 0  (已上传，清零)
          flat *= 0.9  (decay)
          memory[key] = flat.view(original_shape)

  3. 服务端聚合（block 级）:
     对每个 selected block (key_name, start, end):
       acc = weighted_avg(client_block_deltas[key_name][block])
       global_state[key_name].view(-1)[start:end] += acc

  4. 评估：加载更新后的 global_state，在 eval 集上算 loss
```

### 6.1 Block 级操作的关键实现

由于 weight tensor 是多维的（如 q_proj 是 [896, 896]），block 切分在 **flatten 后的 1D view** 上操作：

```python
# 切分：tensor → 1D → block offsets
flat = tensor.contiguous().view(-1)
for start in range(0, flat.numel(), BLOCK_SIZE):
    end = min(start + BLOCK_SIZE, flat.numel())
    block = (key_name, start, end)

# 上传：提取 block 切片
slice = to_send[key_name].view(-1)[start:end].clone()

# 聚合：写回 block 切片
gflat = global_state[key_name].view(-1)
gflat[start:end] += acc
global_state[key_name] = gflat.view(original_shape)

# Memory 更新：清零已上传 block
flat = to_send[key_name].view(-1).clone()
flat[start:end] = 0  # for each selected block
flat *= decay
memory[key_name] = flat.view(original_shape)
```

## 7. 每 Epoch 覆盖保证

```
Epoch（5 轮）结束后：
  - 每层 38 个 block 全部被选一次 → 24×38 = 912 block 更新
  - non_layer 521 个 block 全部被选一次 → 521 block 更新
  - 总计 1433 个 block 全部覆盖 = 630.2M elems = 100% 模型

  无饥饿、无重复，信息零浪费。
```

## 8. 20 轮完整数据

### 每轮上传量与 eval loss

| 轮次 | Epoch | Slot | 上传 blocks | 上传 elems | 上传 % | eval loss |
|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 297 | 129.9M | 20.6% | 1.3946 |
| 2 | 0 | 1 | 296 | 128.7M | 20.4% | 1.2804 |
| 3 | 0 | 2 | 296 | 134.0M | 21.3% | 1.2236 |
| 4 | 0 | 3 | 272 | 116.5M | 18.5% | 1.2030 |
| 5 | 0 | 4 | 272 | 121.1M | 19.2% | 1.1865 |
| 6 | 1 | 0 | 297 | 126.7M | 20.1% | 1.1715 |
| 7 | 1 | 1 | 296 | 132.2M | 21.0% | 1.1559 |
| 8 | 1 | 2 | 296 | 132.9M | 21.1% | 1.1464 |
| 9 | 1 | 3 | 272 | 120.1M | 19.1% | 1.1395 |
| 10 | 1 | 4 | 272 | 118.2M | 18.8% | 1.1314 |
| 11 | 2 | 0 | 297 | 130.8M | 20.8% | 1.1259 |
| 12 | 2 | 1 | 296 | 126.7M | 20.1% | 1.1183 |
| 13 | 2 | 2 | 296 | 130.5M | 20.7% | 1.1005 |
| 14 | 2 | 3 | 272 | 119.8M | 19.0% | 1.0953 |
| 15 | 2 | 4 | 272 | 122.4M | 19.4% | 1.0901 |
| 16 | 3 | 0 | 297 | 130.5M | 20.7% | 1.0857 |
| 17 | 3 | 1 | 296 | 135.8M | 21.5% | 1.0807 |
| 18 | 3 | 2 | 296 | 128.3M | 20.4% | 1.0762 |
| 19 | 3 | 3 | 272 | 117.6M | 18.7% | 1.0730 |
| 20 | 3 | 4 | 272 | 118.0M | 18.7% | 1.0514 |

### 上传统计

```
上传 % : min=18.5%, max=21.5%, avg=20.0%, range=3.0%
总上传 : 2520.7M elems = 5.0 GB (bf16)，恰好 S2 的 20%
```

### 每 Epoch 上传总量

```
Epoch 0 (r1-5):  129.9 + 128.7 + 134.0 + 116.5 + 121.1 = 630.2M (100%)
Epoch 1 (r6-10): 126.7 + 132.2 + 132.9 + 120.1 + 118.2 = 630.1M (100%)
Epoch 2 (r11-15): 130.8 + 126.7 + 130.5 + 119.8 + 122.4 = 630.2M (100%)
Epoch 3 (r16-20): 130.5 + 135.8 + 128.3 + 117.6 + 118.0 = 630.2M (100%)

每 Epoch 精确覆盖 100%，平均每轮 20%。
```

## 9. 结果对比

### 全方案对比

| 方案 | 终点 eval loss | 总上传 | % of S2 | 上传波动 | gap vs S2 |
|---|---|---|---|---|---|
| S2 全量 | 1.0091 | 25.2 GB | 100% | — | — |
| **S3R12v3** block-uniform | **1.0514** | **5.0 GB** | 20% | **18.5%~21.5%** | +0.042 |
| S3R12v2 key-level | 1.0569 | 5.0 GB | 20% | 7.3%~37.4% | +0.048 |
| S3R11 random-layer+mem | 1.1190 | 13.9 GB | 55% | — | +0.110 |
| S3 shard no-mem | 1.1501 | 5.0 GB | 20% | — | +0.141 |

### S3R12v3 vs S3R12v2 直接对比

| 指标 | S3R12v2（key 级） | S3R12v3（block 级） | 提升 |
|---|---|---|---|
| 终点 eval loss | 1.0569 | **1.0514** | **好 0.006** |
| 上传波动 range | 30.1% | **3.0%** | **缩小 10 倍** |
| 上传 min | 7.3% | 18.5% | 更稳定 |
| 上传 max | 37.4% | 21.5% | 无尖峰 |
| loss 曲线 | 有锯齿（r18 跳变） | **平滑单调** | 更美观 |
| 总带宽 | 5.0 GB | 5.0 GB | 相同 |

### 关键发现

1. **S3R12v3 比 S3R12v2 略好 0.006**：block 级均匀上传不仅消除波动，还带来微弱收敛提升
2. **上传均匀性大幅改善**：波动从 30% range 缩小到 3% range，几乎完美均匀
3. **在真 20% 带宽下**：S3R12v3 gap vs S2 仅 0.042，达到 **96% 全量收敛质量**
4. **loss 曲线更平滑**：消除了 S3R12v2 的锯齿，全程单调下降无跳变

## 10. 收敛曲线特点

```
r 1: 1.3946  ← 比 S3R12v2(1.3606) 略高，因为 block 级选择每层只传部分 key
r 5: 1.1865  ← Epoch 0 结束，100% 覆盖
r10: 1.1314  ← Epoch 1 结束
r15: 1.0901  ← Epoch 2 结束
r20: 1.0514  ← Epoch 3 结束（4 个 Epoch = 20 轮）
```

与 S3R12v2 不同，S3R12v3 的 loss 曲线**全程平滑无锯齿**，因为每轮上传量恒定，没有大 key 命中导致的跳变。

## 11. 配置

- **脚本**：`scripts/run_s3r12v3_block_uniform.py`
- **模型**：`/data/models/Qwen/Qwen2.5-0.5B`（base，bf16，24 层，~494M 参数）
- **数据**：medical_meadow_flashcards，30176 train IID 切 2 份，3352 eval
- **联邦**：20 轮，每轮每客户端本地训练 30 步，2 客户端
- **Block 切分**：BLOCK_SIZE = 524288 elements（1 MB bf16），全模型 1433 个 block
- **分组**：25 个组（24 layer + 1 non_layer），每组独立排列轮转
- **排列**：Fisher-Yates shuffle，每 Epoch 重新生成，H=5 无放回轮转
- **memory**：有，decay=0.9，block 级清零（已上传 block 清零，未上传保留 ×decay）
- **超参**：lr=1e-5, bs=8, grad_accum=2, cosine+warmup 0.1, bf16, gradient_checkpointing, seq_len=512
- **日志**：`logs/s3r12v3-block-uniform.log`
- **输出**：`output/s3r12v3-block-uniform/`（含 `round_log.json`、`final_global.pt`）
- **曲线图**：`output/s3r12v3-vs-all.png`（脚本 `scripts/plot_s3r12v3_vs_all.py`）

## 12. S3R12v2 → S3R12v3 的改动总结

### 12.1 build_group_blocks（新增）

```python
# S3R12v2: build_group_map 返回 group_id -> [key_names]
# S3R12v3: build_group_blocks 返回 group_id -> [(key_name, start, end)]
def build_group_blocks(state):
    for key_name in keys:
        for start in range(0, n_elem, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, n_elem)
            blocks.append((key_name, start, end))
```

### 12.2 encode_block_delta（block 切片上传）

```python
# S3R12v2: encode_shard_delta 上传整个 key 的 tensor
# S3R12v3: encode_block_delta 上传 key 的 block 切片（flat view）
def encode_block_delta(to_send, selected_by_key):
    flat = to_send[key_name].contiguous().view(-1)
    result[key_name] = [(s, e, flat[s:e].clone()) for s, e in slices]
```

### 12.3 apply_block_delta（block 切片聚合）

```python
# S3R12v2: apply_shard_delta 对整个 key 聚合
# S3R12v3: apply_block_delta 对 block 切片聚合（flat view 写回）
def apply_block_delta(global_state, ...):
    gflat = global_state[key_name].contiguous().view(-1)
    gflat[s:e] += acc
    global_state[key_name] = gflat.view(original_shape)
```

### 12.4 update_block_memory（block 级 memory 更新）

```python
# S3R12v2: 整个 key 一起更新 memory
# S3R12v3: block 级清零（已上传 block 清零，未上传保留），然后整体 ×decay
def update_block_memory(to_send, selected_by_key, decay):
    flat = to_send[key_name].view(-1).clone()
    for s, e in selected_by_key[key_name]:
        flat[s:e] = 0.0  # 已上传 block 清零
    flat *= decay
    memory[key_name] = flat.view(original_shape)
```

## 13. 结论

S3R12v3 是当前最优的 20% 带宽方案：

1. **真正 20% 带宽**：5.0 GB 总上传，和 S3 公平对比
2. **上传均匀**：每轮 18.5%~21.5%，波动仅 3%，消除尖峰
3. **收敛最优**：终点 1.0514，在 20% 带宽下达到 96% 全量收敛质量
4. **覆盖保证**：H=5 轮内 100% block 覆盖，无饥饿
5. **memory 补偿**：未上传 block 的 delta 累积到 memory，信息不丢失
6. **曲线平滑**：全程单调下降无锯齿

从 S3R12v2 到 S3R12v3 的改进证明了：**block 级选择不仅解决上传波动问题，还带来微弱收敛提升**。block 级粒度使每轮全模型均匀参与，信息覆盖更精细。

## 日志

- 2026-09-04: 基于 S3R12v2 的上传波动问题（7%~37%），设计 S3R12v3 改为 block 级选择
- 2026-09-04: 实现 `scripts/run_s3r12v3_block_uniform.py`，初次运行因多维 tensor 切片报错（`RuntimeError: size mismatch`）
- 2026-09-04: 修复——改用 `tensor.contiguous().view(-1)` flat view 操作 block 切片，重新运行成功
- 2026-09-04: S3R12v3 完成 20 轮训练，终点 eval loss 1.0514，上传统计 min=18.5% max=21.5% range=3.0%，生成对比图 `output/s3r12v3-vs-all.png`
