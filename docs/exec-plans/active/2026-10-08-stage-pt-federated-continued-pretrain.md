# STAGE-PT：联邦续预训练（0.5B + FineWeb-Edu）

- **状态**：active（**阶段一协议已验证**；**阶段二完整 PT 进行中**）
- **创建**：2026-10-08
- **更新**：2026-10-09（阶段二：packing + local_steps=200 + 500M 语料；PT-CODE-2 done）
- **不改**：Hadamard + INT16 + Issue #1；S3R12v3 `public_random`；SecAgg 协议不按任务特化
- **关联**：
  - 母计划 [Non-IID + 更大模型 §3.5](2026-09-18-non-iid-and-larger-models.md)（STAGE-PT 从该处拆出）
  - [当前联调流程](../../algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md)
  - [冒烟](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-smoke.md) · [R20](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-r20.md)
  - SFT 主数字（闪卡 / Dolly）**不进**本表；本表数字也**不回写** SFT 主表

## 0. 目标与边界

**目标**：证明双 ICC 上可以用现有稀疏 SecAgg 协议做 **文档级因果续预训练（CPT）**，而不只是指令 SFT。

### 0.1 阶段一结论（已验证）

| 项 | 结果 |
|---|---|
| 协议 | **跑通**：`causal_pt` + 10% SecAgg，冒烟 `202610081654` / 正式 `202610081718` |
| R20 eval | 2.614 → **2.608**（略降、无 nan） |
| 预算 | **故意偏小**：`local_steps=30` ⇒ 每端每轮 ≈0.5M tokens；不是工业级 CPT |
| 主张边界 | 只主张「联邦因果 PT 通路可用」；**不主张**「已显著写入通用知识」 |

### 0.2 阶段二目标（完整 PT）

阶段一仅验证协议通路，eval 几乎不动（20 轮仅降 0.006）。阶段二目标是在合理预算内做一次 **完整 CPT**，使 eval CE **明显下降**，并能与 S2 dense 对照量化隐私聚合代价。

| 项 | 阶段一（协议验证） | 阶段二（完整 PT） | 说明 |
|---|---|---|---|
| packing | 无（每篇 pad 到 seq_len） | **有**（EOS 拼接 + 切块） | 短文档占多数时 padding 浪费 40-60% tokens |
| `local_steps` | 30 | **200** | 每端每轮 ~0.5M → ~3.3M tokens |
| train tokens | 200M（每端 100M） | **500M**（每端 250M） | 0.5B CPT 需 1B 量级信号 |
| eval tokens | 20M | **50M** | PPL 更稳 |
| `eval_every_n_rounds` | 5 | **2** | 更细曲线 |
| 每轮 token（packing 后） | ~0.5M | ~3.3M+ | packing 提升有效 token 密度 |
| 20 轮每端累计 | ~10M（看 10% 数据） | ~66M+（看 ~26% 数据） | 有意义的知识写入 |
| S2 对照 | 未跑 | **PT-BASE-S2（后置）** | SecAgg 跑通后再开 |

| 项 | 决策 |
|---|---|
| 模型 | **Qwen2.5-0.5B base**（续训，不从随机初始化） |
| 领域 | **通用**（不绑医学） |
| 语料 | **FineWeb-Edu** 抽样（阶段一 200M；阶段二 **500M** train + 50M eval） |
| 切法 | 随机 50/50 → ICC1 / ICC2（IID proxy；异质切法后置） |
| 协议 | 复用现有 10% `public_random` + Windowed SecAgg（`memory_decay=1.0`） |
| eval | hold-out **PPL / CE**；可选下游探针后置 |
| 与 SFT | **分开报**；禁止拿本表数字解释闪卡 0.88 / 0.95 |

**不做（本阶段）**：全量 FineWeb、医学 PubMed/PMC、MIMIC、从零预训练、和 DATA-C1 / QUANT 同周叠跑。

---

## 1. 为什么用 FineWeb-Edu

| 理由 | 说明 |
|---|---|
| 通用 | 教育向网页，不绑医学；符合「不限定领域」 |
| 质量 | 比原始 C4 干净，0.5B 更怕脏数据 |
| 体量可控 | 易按 token 抽样到 1–2B，双 ICC 能跑完 |
| 社区习惯 | 近年开源预训练 / CPT 常用 FineWeb 系当高质量 web 默认 |

备选（本阶段不换，仅记录）：要写「机构文体异质」时再用 The Pile 分源（Wiki+CC vs ArXiv+SE）；医学故事另开计划用 PubMed-A / PMC。

HuggingFace：`HuggingFaceFW/fineweb-edu`（用官方 streaming / 已发布 shard，**固定 revision + 抽样 seed**）。

---

## 2. 看板

| ID | 优先级 | 状态 | 任务 | 依赖 |
|---|---|---|---|---|
| **PT-DOC** | P0 | **done**（本文） | 写清数据 / 训练 / eval / 验收；与 SFT 划界 | — |
| **PT-DATA-1** | P0 | **done（冒烟规模）** | FineWeb-Edu `sample/10BT` 第 0 分片（parquet）；chars/4 抽 **~200M train + 20M eval**。正式 1–2B 可再跑脚本加大 | PT-DOC |
| **PT-DATA-2** | P0 | **done** | `data/splits/fineweb-edu-pt/` 已切：icc1/icc2 各 84278 文档；eval 16857；已同步 ICC1/ICC2/Central | PT-DATA-1 |
| **PT-CODE-1** | P0 | **done（无 packing）** | `objective=causal_pt`：jsonl `text`、无 chat template；packing 后置 | PT-DOC |
| **PT-CFG-1** | P0 | **done** | `configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml` + `deployment/nodes-pt-fineweb-0.5b.yaml` | PT-DATA-2, PT-CODE-1 |
| **PT-SMOKE** | P0 | **done** `202610081654` | 5 轮完成：R1 eval=2.614 → R5 eval=2.612；SecAgg 全程 OK。记录 [smoke](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-smoke.md) | PT-CFG-1 |
| **PT-RUN-20** | P1 | **done** `202610081718` | R20 eval=**2.608**（R1=2.614）；记录 [r20](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-r20.md) | PT-SMOKE |
| **PT-REC** | P1 | **done** | [smoke](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-smoke.md) + [r20](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-r20.md) | PT-RUN-20 |
| **PT-CODE-2** | P1 | **done** | `PackedDataset`：EOS 拼接 + 切块到 `seq_len`；`--packing` CLI + yaml 开关；`causal_pt` only，eval 不 pack | PT-CODE-1 |
| **PT-DATA-3** | P1 | **todo** | FineWeb-Edu 抽 **~500M train + 50M eval**（2-3 个 parquet 分片）；覆盖原 200M 数据；同步 ICC1/ICC2 | PT-CODE-2 |
| **PT-CFG-2** | P1 | **done** | yaml 升级：`packing: true`、`local_steps: 200`、`eval_every_n_rounds: 2` | PT-CODE-2 |
| **PT-FULL-20** | P1 | **done** `202610091639` | 完整 PT 20 轮 SecAgg；R20 eval=**2.610**（R1=2.618，降幅 0.008）；eval 降幅偏小，根因 lr=5e-6 过低。记录 [full-r20](../../experiment-records/2026-10-09-stage-pt-fineweb-0.5b-full-r20.md) | PT-DATA-3, PT-CFG-2 |
| **PT-FULL-REC** | P1 | **done** | 完整 PT 实验记录 [full-r20](../../experiment-records/2026-10-09-stage-pt-fineweb-0.5b-full-r20.md) | PT-FULL-20 |
| **PT-BASE-S2** | P2 | todo | 同数据 **无 SecAgg 全量 S2** 对照（看隐私聚合代价）；PT-FULL-20 跑通后开 | PT-FULL-20 |
| **PT-LR-SWEEP** | P2 | todo | 提高 lr（如 `1e-5` / `2e-5`）重跑，确认 eval 降幅小是 lr 问题而非协议问题 | PT-FULL-20 |
| **PT-SCALE-DATA** | P3 | todo | 可选：语料抽到 1B+ tokens（仅在 500M 不够时再开） | PT-FULL-20 |

阶段一（协议验证）已闭环。阶段二（完整 PT）进行中。

---

## 3. 数据规格

### 3.1 抽样

| 参数 | 阶段一 | 阶段二（完整 PT） | 说明 |
|---|---|---|---|
| 源 | `HuggingFaceFW/fineweb-edu` | 同 | 固定 `revision` 写入 README |
| seed | `20260831` | 同 | 可复现 |
| train tokens | ~200M（每端 100M） | **~500M**（每端 250M） | 0.5B CPT 需 1B 量级信号 |
| eval tokens | ~20M | **~50M** | PPL 更稳 |
| parquet 分片 | 1 个（~2GB） | **2-3 个**（~4-6GB） | sample/10BT |
| 切分 | 文档级随机 50/50 → ICC1 / ICC2 | 同 | 同文档不跨端 |
| 格式 | jsonl `{"text": "..."}` | 同 | 不再用 instruction/input/output |

落地目录（建议）：

```
data/splits/fineweb-edu-pt/
  README.md          # revision、seed、token 统计、下载命令
  icc1_train.jsonl
  icc2_train.jsonl
  eval.jsonl
```

脚本建议：`scripts/prepare_fineweb_edu_pt.py`（streaming 抽样 → 计 token → 切分 → 写 README）。不要手工拷无版本 shard。

### 3.2 与现有 SFT 数据的区别

| | SFT（闪卡 / Dolly） | 本计划 PT |
|---|---|---|
| 样本 | 指令对 | 连续文档 `text` |
| 模板 | chat template | **无** chat；原文 tokenize |
| 长度 | 常短 QA | 长文；`seq_len` 建议 **1024**（显存不够再 512） |
| packing | 当前多 `packing=False` | **packing=True**（`PackedDataset`：EOS 拼接 + 切块到 seq_len） | 阶段二已实现 |

---

## 4. 训练规格

### 4.1 代码改动要点（PT-CODE-1 / PT-CODE-2）

当前 `experiments/run_s3r12v3_fsdp.py` 的 `to_chat_texts` 把 `instruction/input/output` 打成 chat——**PT 不能走这条路径**。

PT-CODE-1（done）：

1. 数据加载：读 jsonl 的 `text` 字段。
2. Dataset：因果 LM；`labels = input_ids`，pad 位 `-100`。
3. loss：已有 `causal_lm_loss_fp32` 可复用。
4. 配置开关：`train.objective: sft_chat | causal_pt`，避免改坏医学 SFT 配置。

PT-CODE-2（done）：

5. `PackedDataset`：所有文档用 EOS 拼接后切成 `seq_len` 块，消除 per-doc padding 浪费。仅 `causal_pt` 使用；eval 仍用 `ChatDataset`（per-doc，报准确 PPL）。
6. CLI 开关：`--packing` / `--no-packing`（`BooleanOptionalAction`）；yaml 开关：`train.packing: true`。
7. `run_config.py` schema 已加 `packing`，server/client/启动脚本三端一致。

**聚合 / SecAgg / MinIO / FSDP**：沿用，不特化。

### 4.2 超参

对齐 0.5B 医学 SecAgg 栈，只改与 PT 相关的项：

| 项 | 阶段一 | 阶段二（完整 PT） | 备注 |
|---|---|---|---|
| `num_rounds` | 20 | 20 | 与现主表轮次可比 |
| `packing` | false | **true** | `PackedDataset`；消除 padding 浪费 |
| `local_steps` | 30 | **200** | 每端每轮 ~0.5M → ~3.3M tokens |
| `batch_size` × `grad_accum` | 8 × 2 | 8 × 2 | packing 后视显存调 |
| `lr` | `5e-6` | `5e-6` | CPT 常用更小 LR |
| `seq_len` | 1024 | 1024 | 长文收益 |
| `coverage_h` | 10 | 10 | 同现网 |
| `memory_decay` | 1.0 | 1.0 | ABL-MDEC 结论 |
| SecAgg | Hadamard + INT16 on | 同 | 同现网 |
| `eval_every_n_rounds` | 5 | **2** | 更细曲线 |
| train tokens | 200M | **500M** | 每端 250M |
| eval tokens | 20M | **50M** | PPL 更稳 |

yaml：`configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml`。  
启动仍用 `scripts/start_s3r12v3_fsdp_run.sh` + `AGGREGATION_PORT_OVERRIDE` 等现有运维约定。

### 4.3 对照（后置）

| 跑次 | 目的 |
|---|---|
| SecAgg 10%（主） | 隐私聚合下 CPT 能否降 PPL |
| 无 SecAgg 全量 S2（PT-BASE-S2） | 协议代价上界 |
| （可选）单机集中 CPT 同 token | 联邦 vs 集中差距；GPU 紧时可不做 |

---

## 5. 评估

| 指标 | 做法 | 成功标准 |
|---|---|---|
| eval CE / PPL | `eval.jsonl` 上因果 LM；`PPL = exp(CE)` | 阶段一：R20 相对 R0 下降（已验证，降幅小）；**阶段二：R20 相对 R0 明显下降（>0.02）** |
| train CE | 每轮本地/全局日志 | 总体下降、无持续 nan |
| 稳定性 | 聚合成功、delta 可下发 | 与现 0.5B SecAgg 运维同级 |

**不要**用医学闪卡 eval、也不要用 Dolly CE 解释本实验。  
下游探针（HellaSwag / ARC 等）标为可选后置，不阻塞 PT-RUN-20。

---

## 6. 验收与结项

**阶段一（协议）已验收**（2026-10-08）：

1. `data/splits/fineweb-edu-pt/README.md` + `manifest.json` 可复现抽样；`*.jsonl` 不进 git（见 `.gitignore`）。
2. `objective=causal_pt` 不破坏医学 SFT 默认路径。
3. 冒烟 `202610081654` + 正式 `202610081718` 已记入 experiment-records。
4. 母计划 STAGE-PT 标为「阶段一 done / 加码可选」。

**阶段二（完整 PT）验收标准**（待 PT-FULL-20）：

1. `PackedDataset` 在 `causal_pt` 下正确工作；`packing=true` 不破坏 SFT 路径。
2. 500M/50M 数据已生成并同步双 ICC。
3. R20 eval CE 相对 R1 **明显下降（>0.02）**；train CE 总体下降；无 nan。
4. SecAgg 全程成功；运维与阶段一同级。
5. 实验记录写入 `docs/experiment-records/`。
6. PT-BASE-S2 对照（后置）：同数据无 SecAgg 全量 FedAvg，量化隐私聚合代价。

整份计划（含 `PT-BASE-S2`）全部 done 后，再移到 `docs/exec-plans/completed/`。

---

## 7. 落地顺序与启动

```
阶段一：PT-DATA-1 → PT-DATA-2 → PT-CODE-1 → PT-CFG-1 → PT-SMOKE → PT-RUN-20 → PT-REC   ✅

阶段二：PT-CODE-2 → PT-CFG-2 → PT-DATA-3 → PT-FULL-20 → PT-FULL-REC
                                                  ↘ PT-BASE-S2（后置）
```

阶段二数据准备（500M train / 50M eval）：

```bash
python scripts/prepare_fineweb_edu_pt.py \
  --source parquet --hf-endpoint https://hf-mirror.com --num-parquets 3 \
  --train-tokens 500000000 --eval-tokens 50000000 \
  --out-dir data/splits/fineweb-edu-pt
# 然后同步 icc1_train.jsonl / icc2_train.jsonl / eval.jsonl 到 ICC1 / ICC2
```

完整 PT 20 轮（SecAgg）：

```bash
AGGREGATION_PORT_OVERRIDE=8081 \
NODES_FILE=deployment/nodes-pt-fineweb-0.5b.yaml \
RUN_CONFIG=configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml \
TAG=pt-fineweb-0.5b-full \
bash scripts/start_s3r12v3_fsdp_run.sh
```

冒烟验证 packing 通路（5 轮）：

```bash
NUM_ROUNDS_ENV=5 \
AGGREGATION_PORT_OVERRIDE=8081 \
NODES_FILE=deployment/nodes-pt-fineweb-0.5b.yaml \
RUN_CONFIG=configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml \
TAG=pt-fineweb-0.5b-full-smoke \
bash scripts/start_s3r12v3_fsdp_run.sh
```
