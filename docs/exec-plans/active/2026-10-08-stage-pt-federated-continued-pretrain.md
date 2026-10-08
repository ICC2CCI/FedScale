# STAGE-PT：联邦续预训练（0.5B + FineWeb-Edu）

- **状态**：active（**阶段一协议已验证**；加码 CPT 仍开放）
- **创建**：2026-10-08
- **更新**：2026-10-08（PT-SMOKE + PT-RUN-20 done）
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

加码（更大 `local_steps` / 语料 / packing）见看板 `PT-SCALE-*`（可选，未开）。

| 项 | 决策 |
|---|---|
| 模型 | **Qwen2.5-0.5B base**（续训，不从随机初始化） |
| 领域 | **通用**（不绑医学） |
| 语料 | **FineWeb-Edu** 抽样（落地 ~200M train + 20M eval；可再加大） |
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
| **PT-BASE-S2** | P2 | todo | 可选：同数据 **无 SecAgg 全量 S2** 对照（看隐私聚合代价） | PT-RUN-20 |
| **PT-REC** | P1 | **done** | [smoke](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-smoke.md) + [r20](../../experiment-records/2026-10-08-stage-pt-fineweb-0.5b-r20.md) | PT-RUN-20 |
| **PT-SCALE-STEPS** | P2 | todo | 可选加码：提高 `local_steps`（如 200–300）使每轮吃更多 token | PT-RUN-20 |
| **PT-SCALE-DATA** | P3 | todo | 可选：语料抽到 1B+ tokens（仅在 steps 加大后有意义） | PT-SCALE-STEPS |

阶段一（协议验证）已闭环。GPU 加码实验按需再开，不阻塞 SFT 主线。

---

## 3. 数据规格

### 3.1 抽样

| 参数 | 建议值 | 说明 |
|---|---|---|
| 源 | `HuggingFaceFW/fineweb-edu` | 固定 `revision` 写入 README |
| seed | `20260831`（与现有联邦切分一致） | 可复现 |
| train tokens | **~1.0B** 总量（两端合计）；吃紧可先 **~200–500M** 冒烟 | 0.5B CPT 信号够用 |
| eval tokens | **~50M**（从同一池 hold-out，与 train 不重叠） | 报 PPL |
| 切分 | 文档级随机 50/50 → ICC1 / ICC2 | 同文档不跨端 |
| 格式 | **jsonl**，每行至少 `{"text": "..."}` | 不再用 instruction/input/output |

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
| packing | 当前多 `packing=False` | **建议 packing=True**（提高 token/步效率） |

---

## 4. 训练规格

### 4.1 代码改动要点（PT-CODE-1）

当前 `experiments/run_s3r12v3_fsdp.py` 的 `to_chat_texts` 把 `instruction/input/output` 打成 chat——**PT 不能走这条路径**。

需要：

1. 数据加载：读 jsonl 的 `text` 字段。
2. Dataset：因果 LM；`labels = input_ids`，pad 位 `-100`。
3. 可选：concat + pack 到 `seq_len`（文档边界可用 eos 分隔）。
4. loss：已有 `causal_lm_loss_fp32` 可复用。
5. 配置开关：例如 `train.objective: sft_chat | causal_pt`，避免改坏医学 SFT 配置。

**聚合 / SecAgg / MinIO / FSDP**：沿用，不特化。

### 4.2 超参（第一枪，可在冒烟后微调）

对齐 0.5B 医学 SecAgg 栈，只改与 PT 相关的项：

| 项 | 建议 | 备注 |
|---|---|---|
| `num_rounds` | 20（冒烟 5） | 与现主表轮次可比 |
| `local_steps` | 30 | 先不动；token 预算不够再加 |
| `batch_size` × `grad_accum` | 8 × 2 | 与 mdec-d10 同；packing 后视显存调 |
| `lr` | `1e-5` 或略低于 SFT（如 `5e-6`） | CPT 常用更小 LR；冒烟看 loss 是否炸 |
| `seq_len` | **1024**（不行退 512） | 长文收益 |
| `coverage_h` | 10 | 同现网 |
| `memory_decay` | 1.0 | ABL-MDEC 结论 |
| SecAgg | Hadamard + INT16 on | 同现网 |
| `eval_every_n_rounds` | 5 | 报 eval CE → 换算 PPL |

yaml 名建议：`configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml`。  
启动仍用 `scripts/start_s3r12v3_fsdp_run.sh` + `AGGREGATION_PORT_OVERRIDE` 等现有运维约定。

### 4.3 对照（后置）

| 跑次 | 目的 |
|---|---|
| SecAgg 10%（主） | 隐私聚合下 CPT 能否降 PPL |
| 无 SecAgg 全量 S2（PT-BASE-S2） | 协议代价上界 |
| （可选）单机集中 CPT 同 token | 联邦 vs 集中差距；GPU 紧时可不做 |

---

## 5. 评估

| 指标 | 做法 | 成功标准（第一枪） |
|---|---|---|
| eval CE / PPL | `eval.jsonl` 上因果 LM；`PPL = exp(CE)` | R20 相对 R0 **明显下降**（不必追 SOTA） |
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

整份计划（含 `PT-SCALE-*`）全部不做时，再移到 `docs/exec-plans/completed/`。

---

## 7. 落地顺序与启动

```
PT-DATA → PT-CODE → PT-CFG → PT-SMOKE → PT-RUN-20 → PT-REC   ✅
                                      ↘ PT-SCALE-STEPS / PT-BASE-S2（可选）
```

复现抽样：

```bash
python scripts/prepare_fineweb_edu_pt.py \
  --source parquet --hf-endpoint https://hf-mirror.com --num-parquets 1 \
  --train-tokens 200000000 --eval-tokens 20000000 \
  --out-dir data/splits/fineweb-edu-pt
```

正式 20 轮：

```bash
cp deployment/nodes-pt-fineweb-0.5b.yaml.example deployment/nodes-pt-fineweb-0.5b.yaml
# 按机房改 ssh / repo / model_path
AGGREGATION_PORT_OVERRIDE=8081 \
NODES_FILE=deployment/nodes-pt-fineweb-0.5b.yaml \
RUN_CONFIG=configs/s3r12v3-fsdp-pt-fineweb-0.5b.yaml \
TAG=pt-fineweb-0.5b-r20 \
bash scripts/start_s3r12v3_fsdp_run.sh
```

加码每轮 token：提高 yaml 里 `train.local_steps`（优先），必要时再加大语料。
