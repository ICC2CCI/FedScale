# Qwen2.5-3B 医学闪卡 IID + Windowed SecAgg

- **状态**：done
- **创建**：2026-09-20
- **跑次**：`results/202609201530/`（gitignore；图用 `python scripts/plot_s3r12v3_fsdp_run.py` 生成）
- **配置**：`configs/s3r12v3-fsdp-medical-3b-secagg.yaml`
- **关联**：
  - [执行计划](../exec-plans/active/2026-09-18-non-iid-and-larger-models.md) MODEL-3B
  - [FSDP scatter-load](../algorithm/2026-09-20-fsdp-scatter-load.md)
  - 0.5B SecAgg 主数字：`202609181406` R20 eval≈0.985（**不要和 3B eval 比**）

## 1. 设定

只换模型，数据仍是医学闪卡 IID（`icc1_client0_train.json` / `client1_train.json`，共享 `medical_flashcards_eval.json`）。完整方案：Hadamard + INT16 + Issue #1，**不跑无 SecAgg 对照**。

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-3B base，8×V100 FSDP，`Qwen2DecoderLayer` |
| 参数 / window | 3.40B 浮点；`coverage_h=10` → **840** window（不是纸面 ×6 的 1.6k） |
| batch / seq | **4** / 512（batch=8 预计 OOM；实测约 21 GiB/卡） |
| 轮数 | 20；eval 每 5 轮 + 第 1 轮 |
| 端口 | `AGGREGATION_PORT_OVERRIDE=8081` |
| round-0 | 须先 `bootstrap_initial_state.sh model/Qwen/Qwen2.5-3B`；不能沿用 0.5B 的 MinIO `state.pt` |

## 2. 结果

| 轮 | eval | 平均 train |
|---|---|---|
| R1 | 1.213 | 1.641 |
| R5 | 1.123 | 1.166 |
| R10 | 0.994 | 1.031 |
| R15 | 0.944 | 0.963 |
| **R20** | **0.880** | 0.887 |

- 上行约 622–660 MiB/端（INT16 × 10%）
- 稳态整轮约 **385 s**（训练 ~87 s，encode ~70–80 s，上传 ~40 s，unmask ~19 s；有 eval 再 +40 s）
- 图：`results/202609201530/figures/{eval,train}_loss.png` 等

换模型前必须换 MinIO round-0。第一次 3B 启动曾误加载 0.5B 的 630M / 269 window，已停掉重 bootstrap。

## 3. 途中故障

第 1 枪（无 scatter-load）R1 成功后 R2 ICC1 `SIGKILL`：8 份整模 broadcast。修复见 [FSDP scatter-load](../algorithm/2026-09-20-fsdp-scatter-load.md)。修复后 20 轮全部聚合成功。
