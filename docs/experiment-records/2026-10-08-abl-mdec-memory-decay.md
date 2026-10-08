# ABL-MDEC：`memory_decay` 同栈扫描（0.5B 医学 SecAgg）

- **状态**：done
- **创建 / 结项**：2026-10-08
- **关联**：[执行计划](../exec-plans/active/2026-09-18-non-iid-and-larger-models.md) ABL-MDEC
- **图**：`results/abl-mdec-compare/memory_decay_eval_train_samestack.png`（同栈四条）
- **决策**：默认 **`memory_decay: 1.0`**（标准 error-feedback）；与 `quant_residual_decay: 1.0` 区分开

## 1. 设定

只改未选中 block 的 `block_memory` 衰减；其余对齐 10% 公开 mask + Windowed SecAgg（Hadamard + INT16 + Issue #1）。

| 项 | 值 |
|---|---|
| 模型 / 数据 | Qwen2.5-0.5B，医学闪卡 IID，双 ICC |
| 协议 | `compressor=public_random`，`coverage_h=10`，SecAgg on |
| 轮数 | 20；eval 每 5 轮（R1/5/10/15/20） |
| 固定 | `quant_residual_decay=1.0`、seed=20260831、batch=8、local_steps=30 |
| 配置族 | `configs/s3r12v3-fsdp-medical-0.5b-mdec-d*.yaml` |

dense 上该系数不起作用；扫描必须走稀疏 mask。

扫过的取值：`{0, 0.5, 0.9, 1.0}`。跳过 0.2 / 0.7（趋势已清楚）。旧跑 `202609181406`（decay=0.9，eval 每轮）**不作中间轮对照**（R1 fingerprint 不同）。

## 2. 跑次与 R20

| decay | 含义 | 跑次 | R20 train | **R20 eval** |
|---|---|---|---|---|
| 0.0 | 未传更新直接丢掉 | `202610081002` | 1.172 | **1.133** |
| 0.5 | 残差较快忘掉 | `202610081052` | 1.025 | **1.041** |
| 0.9 | 旧默认（同栈重跑） | `202610081442` | 0.880 | **0.962** |
| **1.0** | 残差原样保留到被选中 | `202610081136` | 0.844 | **0.953** |

同栈 R1 train 均为 **2.021724**、R1 eval 均为 **1.465684**（四枪起点一致）。

## 3. 同栈 eval 轨迹

| Round | decay=0 | 0.5 | 0.9 | 1.0 |
|---|---|---|---|---|
| 1 | 1.466 | 1.466 | 1.466 | 1.466 |
| 5 | 1.339 | 1.326 | 1.311 | 1.307 |
| 10 | 1.310 | 1.268 | **1.217** | **1.217** |
| 15 | 1.229 | 1.120 | 1.001 | 0.983 |
| 20 | 1.133 | 1.041 | 0.962 | **0.953** |

- R10：同栈下 0.9 与 1.0 几乎相同；此前误以为「0.9 在 R10 更好」是拿旧 `202609181406` 比新栈造成的。
- R11 起（`coverage_h=10` 进入 epoch 1）高 decay 的 train 掉得更陡：残差攒得更满，下一轮选中时一次送出更多。
- 终点排序：**1.0 < 0.9 < 0.5 < 0.0**（越小越好）。1.0 相对 0.9 约好 **0.009**。

历史参考（不同栈，勿与上表混比）：`202609181406` decay=0.9，R20 eval≈**0.985**。

## 4. 和「eval 上升」的关系

早期 SecAgg 从 R2 起 eval 上升（1.50→2.36）是 **错误 scale / amax**，不是 `memory_decay=1.0`。Issue #1 要求 **`quant_residual_decay=1.0`**（量化残差不衰减）。本次四条同栈曲线全程下降，**没有**「block_memory decay=1 会发散」的现象。

## 5. 落地

- `experiments/shared/protocol.py`：`DEFAULT_MEMORY_DECAY = 1.0`
- 生产 / 联调 yaml（`s3r12v3-fsdp-run.yaml`、医学 3B/7B、Dolly、secagg-verify 等）默认改为 `memory_decay: 1.0`
- 消融 yaml `mdec-d0/d02/d05/d07/d09` 仍保留各自取值，便于复现
