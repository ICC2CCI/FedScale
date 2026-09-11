# 联邦分片上传的安全防护方案

- **状态**：planned
- **创建**：2026-09-08
- **更新**：2026-09-08
- **关联**：`docs/exec-plans/completed/2026-09-04-s3r12v3-block-uniform.md`；未实现项见 `docs/exec-plans/active/2026-09-11-dual-cluster-to-production.md`（SEC-1~3）

## 1. 背景与威胁模型

S3R12v3 的 block 级上传中，攻击者（截获上传数据的人）可能通过以下信息推断模型结构：

- **key_name 泄露**：当前上传格式 `(key_name, start, end, slice)` 直接暴露了 block 对应的模型参数名（如 `model.layers.0.self_attn.q_proj.weight`）
- **block 大小差异**：不同 key 的 tensor 大小不同，截获者可从大小区分
- **delta 值分布**：不同层的 delta 有不同的统计特征（如 embed_tokens vs layernorm）

**威胁场景**：
- 网络中间人截获客户端→服务端的上传包
- 恶意客户端看到其他客户端的上传（如 P2P 聚合）
- 服务端被入侵，查看历史上传记录

**目标**：在**不影响训练效果**的前提下，让截获者无法从上传数据推断出对应模型的哪个部分。

## 2. 分层防护方案

### 第一层：统一 block 大小（已实现 ✓）

**现状**：S3R12v3 已将所有 block 切分为 1MB（524288 elements），截获者无法从 tensor 大小区分不同 key。

| 防护点 | 状态 |
|---|---|
| 所有 block 大小一致（1MB） | ✓ 已实现 |
| 最后一个 block 可能偏小 | ⚠️ 可填充到 1MB |

### 第二层：上传时不传 key_name（待实现）

**现状**：当前上传格式包含 `key_name`，直接暴露模型结构。

**改进**：用 `global_block_id`（0~1432 的整数）替代 `key_name`，映射表只存在服务端。

```
当前格式:  (key_name, start, end, slice)
           ("model.layers.0.self_attn.q_proj.weight", 0, 524288, tensor[524288])

改为:      (global_block_id, slice)
           (37, tensor[524288])  ← 只有服务端知道 37 → q_proj.weight 的映射
```

**效果**：截获者只看到一个 1D tensor 切片 + 一个数字，不知道对应哪个 key。

**实现要点**：
- 服务端在 Mask Epoch 开始时构建 `block_id → (key_name, start, end)` 映射表
- 客户端上传时只发 `global_block_id`，不发 key_name
- 映射表不下发给客户端（客户端只需知道自己要上传哪些 block 的切片，由服务端通过 RoundPlan 指定 block_id）

### 第三层：block_id 加密（待实现，推荐）

**现状**：即使不传 key_name，`global_block_id` 仍暴露了 block 的位置信息（如 block_id=0~37 是 layer_0）。

**改进**：用 per-round 对称密钥加密 `global_block_id`：

```python
# 客户端和服务端共享 per-round key（通过 DH 密钥协商或预共享密钥派生）
round_key = HKDF(epoch_seed, context=f"round-{rnd}")
encrypted_block_id = AES_encrypt(round_key, block_id.to_bytes(4, 'big'))

# 上传: (encrypted_block_id, slice)
# 截获者看到: (0x3a7f..., tensor) ← 无法解出 block_id
```

**服务端解密聚合**：

```python
round_key = HKDF(epoch_seed, context=f"round-{rnd}")
for encrypted_id, slice in received:
    block_idx = int.from_bytes(AES_decrypt(round_key, encrypted_id), 'big')
    key_name, start, end = block_layout[block_idx]
    aggregate(key_name, start, end, slice)
```

**密钥管理**：
- `epoch_seed` 由服务端在每个 Mask Epoch 开始时生成并广播
- `round_key` 由 `epoch_seed + round_number` 派生，每轮不同
- 客户端和服务端都能独立派生 `round_key`（不需要密钥分发）
- 密钥协商可用 Diffie-Hellman 或预共享密钥

**效果**：截获者连 block_id 都看不到，只看到一串密文 + tensor。即使积累多轮数据，因每轮密钥不同，也无法关联。

### 第四层：SecAgg 安全聚合（可选，高安全）

**现状**：服务端能看到单个客户端的 delta 值。

**改进**：客户端之间通过 secret sharing 添加随机 mask，聚合后 mask 抵消：

```
客户端 0: delta_0 + mask_0 → 上传
客户端 1: delta_1 + mask_1 → 上传
服务端: (delta_0 + mask_0) + (delta_1 + mask_1) = delta_0 + delta_1  ← mask 抵消
```

**效果**：服务端只能看到聚合后的 sum，看不到单个客户端的 delta。截获者也只能看到加了 mask 的密文。

**代价**：
- 实现复杂（需要客户端间密钥协商、掉线处理）
- 额外通信开销（客户端间交换 mask seed）
- 参考实现：FedScale 的 SecAgg 模块

### 第五层：DP 差分隐私噪声（可选，最高安全但影响效果）

**改进**：给 delta 加少量高斯噪声：

```python
delta += torch.normal(0, sigma, delta.shape)
```

**效果**：即使解密，截获者也无法精确还原 delta。数学上保证不可区分性。

**代价**：
- **会影响收敛**：噪声累积到模型中，需要更大的学习率或更多轮次
- 需要调 `sigma`：太小无保护，太大不收敛
- 与 memory 机制交互复杂：memory 也会累积噪声

## 3. 方案对比

| 方案 | 安全性 | 效果影响 | 实现复杂度 | 通信开销 | 推荐 |
|---|---|---|---|---|---|
| 统一 block 大小 | 低 | 无 | 已实现 | 无 | ✓ 必做 |
| 不传 key_name | 中 | 无 | 低 | 略减（少了 key_name 字符串） | ✓ 必做 |
| block_id 加密 | 中高 | 无 | 中 | 略增（密文比明文长） | ✓ 推荐 |
| SecAgg | 高 | 无 | 高 | 中（mask 交换） | 可选 |
| DP 噪声 | 最高 | **有**（需调参） | 中 | 无额外 | 谨慎 |

## 4. 推荐方案（不影响效果 + 中高安全）

采用**第一层 + 第二层 + 第三层**组合：

```
客户端上传格式:
  (AES_encrypt(round_key, block_id), tensor_slice)

服务端处理:
  1. 解密 block_id → 查映射表 → 得到 (key_name, start, end)
  2. 聚合 slice 到 global_state[key_name][start:end]
```

**安全保证**：
- 截获者只看到 `(密文, 1D tensor[524288])`
- 无法从密文解出 block_id（每轮密钥不同）
- 无法从 tensor 大小区分 key（统一 1MB）
- 无法从 tensor 值分布推断层类型（所有层都切成 1D slice，统计特征混合）

**训练效果**：完全不受影响（只是传输格式变化 + 加解密开销）。

## 5. 实现计划

### 5.1 阶段一：不传 key_name（最小改动）

- [ ] 服务端构建 `block_id → (key_name, start, end)` 全局映射表
- [ ] RoundPlan 中用 `block_id` 替代 `key_name`
- [ ] 客户端上传格式改为 `(block_id, slice)`
- [ ] 服务端聚合时查映射表还原 `(key_name, start, end)`

### 5.2 阶段二：block_id 加密

- [ ] 实现 `HKDF(epoch_seed, context=f"round-{rnd}")` 派生 per-round key
- [ ] 客户端上传前 `AES_encrypt(round_key, block_id)`
- [ ] 服务端收到后 `AES_decrypt(round_key, encrypted_id)`
- [ ] 确保密钥派生跨客户端一致（所有客户端用相同 epoch_seed）

### 5.3 阶段三（可选）：SecAgg

- 参考 FedScale 的 SecAgg 模块实现
- 需要客户端间 DH 密钥协商
- 处理掉线客户端的 mask 残留问题

## 6. 注意事项

1. **mask 选择本身是 public 的**：spec 规定 mask 不依赖私有 delta，截获者知道"哪些 block 被选"不是安全问题——真正需要保护的是 delta 的值和 block 到 key 的关联。

2. **填充最后一个 block**：如果某 key 的 tensor 不能被 BLOCK_SIZE 整除，最后一个 block 偏小，截获者可从大小推断。解决：填充到 1MB（用 0 填充），服务端解填充。

3. **流量分析**：即使加密 block_id，截获者可统计"某轮上传了 N 个 block"来推断 mask_ratio。这不影响安全（mask_ratio 是 public 的），但如需隐藏可固定每轮上传包数量。

4. **重放攻击**：截获者重放旧上传包。解决：RoundPlan 中包含 `round_number` 和 `epoch_seed`，服务端验证轮次一致性。

## 7. 参考文献

- FedScale Public Block Mask Implementation Spec v1（`FedScale_Public_Block_Mask_Implementation_Spec_v1.docx`）
- Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning", CCS 2017（SecAgg）
- Diffie-Hellman key exchange（密钥协商）
- HKDF-SHA256（密钥派生，RFC 5869）

## 日志

- 2026-09-08: 讨论安全防护需求，设计五层防护方案，推荐前三层组合（不影响效果 + 中高安全）
