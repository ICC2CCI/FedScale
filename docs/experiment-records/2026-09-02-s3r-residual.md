# 阶段 D — S3+残差累积结果与反常分析

- **状态**：done
- **创建**：2026-09-02
- **更新**：2026-09-02
- **关联**：`docs/exe-plans/active/2026-09-02-three-scenario-loss-comparison.md`

## 配置

- **脚本**：`scripts/run_s3r_federated_shard20_residual.py`
- **基于 S3 改动**：每客户端维护 `residual_state`（初始 0）
  - 每轮起点 `initial = global + residual`（包含上轮未上传的累积更新）
  - 训练后 `delta = trained - initial`
  - 上传 selected block 的 delta
  - `residual[selected] = 0`（已上传块清零）
  - `residual[other] += delta[other]`（未上传块累积）
- 其余同 S3：20 轮，每轮 30 步，2 客户端，block_size=1MB，mask_ratio=0.2
- **日志**：`logs/s3r-federated-shard20-residual.log`
- **输出**：`output/s3r-federated-shard20-residual/`
- **曲线图**：`output/s3r-federated-shard20-residual/all4-loss-curve.png`（四场景对比）

## 结果（四场景完整对比）

| 指标 | S1-D 基线 | S2 全量上传 | S3 分片20%(无残差) | **S3R 分片20%+残差** |
|---|---|---|---|---|
| eval loss 起点 | 1.10 | 1.24 | 1.52 | 1.52 |
| **eval loss 终点** | **0.86** | **1.01** | **1.15** | **1.33** |
| 下降幅度 | 0.24 | 0.23 | 0.37 | 0.19 |
| gap vs 基线 | — | +0.15 | +0.29 | **+0.47** |
| 过拟合 | 无 | 无 | 无 | 无 |
| 每轮上传量 | — | ~988MB | ~50MB | ~50MB |

## 关键反常发现：S3R 比 S3 更差

**预期**：残差累积保留未上传更新，应改善收敛（接近 S2）。
**实际**：S3R 终点 1.33 比 S3 的 1.15 还差 0.18，且前 10 轮几乎停滞（1.52→1.42）。

## 原因分析

残差机制在本设置下反而有害，根因是 **residual 基于"过期 global"的 delta，加到"新 global"上方向不一致**：

1. **stale residual 问题**：round N 的 residual 累积的是 `(trained_N - global_N)` 的未上传部分。但 round N+1 的 global 已被 server 聚合更新（global_{N+1} ≠ global_N），把基于 global_N 的 delta 加到 global_{N+1} 上，相当于用过期梯度方向扰动新参数，干扰训练。

2. **residual 越积越大**：未上传块每轮都 `+= delta`，但 delta 基于不断变化的起点，residual 可能累积出与当前 global 优化方向相反的量，导致前 10 轮停滞。

3. **block 轮转周期 5 轮**：残差要等 5 轮才被某 block 上传清零，期间该 block 的 residual 一直累积旧 delta，越来越 stale。

4. **对比 S3 无残差**：S3 每轮从干净 global 开始，虽然丢弃 80% 更新，但至少没引入 stale 扰动——"干净但浪费"比"保留但干扰"收敛更好。

## eval loss 完整轨迹

```
round  1 → 1.5164  (同 S3，residual=0)
round  2 → 1.5161
round  3 → 1.4987  (S3 同期 1.32，S3R 反而高)
round  4 → 1.4777
round  5 → 1.4514
round  6 → 1.4493  (前 10 轮几乎停滞)
round  7 → 1.4507
round  8 → 1.4518
round  9 → 1.4313
round 10 → 1.4155
round 11 → 1.4105
round 12 → 1.4083
round 13 → 1.4132
round 14 → 1.3784  (开始加速下降)
round 15 → 1.3633
round 16 → 1.3561
round 17 → 1.3561
round 18 → 1.3631
round 19 → 1.3311
round 20 → 1.3265  ← 终点
```

## 结论

**朴素残差累积（直接把未上传 delta 累加到 residual）在轮转 block mask 机制下反而有害**，因为 residual 携带的是基于过期 global 的 stale delta，加到新 global 上引入方向噪声。

### 残差机制要 work，需要改进方向
1. **residual 衰减**：每轮 `residual *= 0.9`，避免 stale delta 无限累积。
2. **基于固定 base 的 delta**：residual 累积 `(trained - base)` 而非 `(trained - global_N)`，base 不变则 delta 方向稳定。
3. **residual 上传后重置全局**：每 5 轮（完整覆盖一次）后把 residual 清零，避免跨周期 stale。
4. **server 端维护 residual**：让 server 跟踪未聚合块的状态，而非客户端各自累积（避免各客户端 residual 方向不一致）。

### 对实验目标的结论
本实验的增值点已验证：**朴素残差累积不能直接改善分片上传的收敛**，反而因 stale 问题恶化。这本身是有价值的发现——说明残差机制需要更精细的设计（衰减/基于固定 base/重置周期）才能 work。

## 四场景最终排序

```
eval loss 终点（越低越好）：
  S1-D 基线      0.86  ████████████████████  (天花板)
  S2 全量上传    1.01  ████████████████████████  (+0.15 联邦代价)
  S3 分片20%     1.15  ████████████████████████████  (+0.29 联邦+分片代价)
  S3R 分片+残差  1.33  █████████████████████████████████  (+0.47 残差反害)
```

## 日志

- 2026-09-02: 完成 S3R 训练。残差累积反常比 S3 差 0.18，根因是 stale residual（基于过期 global 的 delta 加到新 global 上）。记录改进方向（衰减/固定 base/重置周期）。四场景对比完成，进入阶段 E 汇总。
