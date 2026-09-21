# Qwen2.5-7B 医学闪卡 IID + Windowed SecAgg

- **状态**：done（5 轮冒烟 + 20 轮）
- **创建**：2026-09-21
- **5 轮跑次**：`results/202609210952/`
- **20 轮跑次**：`results/202609211137/`（gitignore；图用 `python scripts/plot_s3r12v3_fsdp_run.py` 生成）
- **20 轮配置**：`configs/s3r12v3-fsdp-medical-7b-secagg-20r.yaml`
- **关联**：
  - [执行计划](../exec-plans/active/2026-09-18-non-iid-and-larger-models.md) MODEL-7B
  - [V100 7B QK fp32](../algorithm/2026-09-21-v100-7b-qk-fp32.md)
  - [FSDP scatter-load](../algorithm/2026-09-20-fsdp-scatter-load.md)
  - 3B SecAgg：`202609201530` R20 eval=0.880（**不要和 7B eval 直接比**）

## 1. 设定

只换模型，数据仍是医学闪卡 IID。完整方案：Hadamard + INT16 + Issue #1。

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-7B base，8×V100 FSDP，`Qwen2DecoderLayer` |
| 参数 / window | 7.62B 浮点；`coverage_h=10` → **1609** window |
| 训练精度 | fp16 权重 + AMP；**QK matmul 单独 fp32**（见算法笔记） |
| batch / seq | **1** / 512，`grad_accum=2`（batch=2 在 QK fp32 后 ~28 GiB/卡 OOM） |
| 端口 | `AGGREGATION_PORT_OVERRIDE=8081` |
| round-0 | `bootstrap_initial_state.sh model/Qwen/Qwen2.5-7B` |

## 2. 5 轮冒烟结果（`202609210952`）

| 轮 | eval | 平均 train |
|---|---|---|
| R1 | 1.351 | 2.576 |
| R2 | — | 1.451 |
| R3 | — | 1.365 |
| R4 | — | 1.223 |
| **R5** | **1.014** | 1.196 |

- 上行约 **1457–1460 MiB**/端（INT16 × 10%）
- 稳态整轮约 **18–20 min**（训练 ~7.5 min，encode ~2.7 min，上传 ~1.5 min，unmask ~42 s；有 eval 再 +110 s）
- 墙钟：09:52 → 11:27，约 **95 min / 5 轮**
- 图：`results/202609210952/figures/{eval,train}_loss.png` 等
- 无 OOM、无 nan。说明 7B + 公开 10% + SecAgg 在当前集群可训。

不要把 R5 eval=1.014 和 3B R20=0.880 比：轮次、batch、模型都不同。

## 3. 途中故障（修 nan）

| 跑次 | 结果 |
|---|---|
| `202609201834` 5 轮 batch=2 | 系统面过，**train/eval=nan**（fp16 QK overflow） |
| 只改 CE→fp32 / RMSNorm.float() | 仍 nan，或 FSDP mixed-dtype crash |
| sdpa | ICC2 ~28 GiB OOM |
| fp32 权重 + AMP | 仍 nan（AMP 把 matmul 打回 fp16） |
| 全 fp32 `202609202021` | ~29 GiB/卡 OOM（embed/lm_head 未切分） |
| QK fp32 + batch=2 `202609202031` | step1 loss 有限，下一步 backward OOM |
| QK fp32 + **batch=1** `202609202039` 2 轮 | train/eval 有限；随后 5 轮 `202609210952` |

## 4. 20 轮结果（`202609211137`）

| 轮 | eval | 平均 train |
|---|---|---|
| R1 | 1.351 | 2.576 |
| R5 | 1.014 | 1.196 |
| R10 | 0.821 | 0.829 |
| R15 | 0.779 | 0.707 |
| **R20** | **0.766** | 0.680 |

- 上行约 **1457–1460 MiB**/端；window=1609（约 10%）
- 整轮约 **18–19 min**（`avg_round_wall_s≈1118`）；墙钟 11:37 → 17:53，约 **6.3 小时**
- 图：`results/202609211137/figures/{eval,train}_loss.png` 等
- 无 OOM、无 nan，20 轮全部聚合成功

这是 7B 自己的 SecAgg 尺子（batch=1），不是相对全量 FedAvg 的掉点。论文主对照仍是 BASE-S2-3B。
