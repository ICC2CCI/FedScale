# 联邦 Non-IID 相关工作调研（数据怎么造、用什么集）

- **日期**：2026-09-18
- **用途**：给跨域实验选数据和评估口径，避免把「Dirichlet 同域」和「两端不同领域」混成同一种 Non-IID
- **关联**：执行计划 [跨域 Non-IID + 更大模型](../exec-plans/active/2026-09-18-non-iid-and-larger-models.md)
- **当前仓库**：两端是同一份 `medical_flashcards` 随机 50/50，属下表 **A 类（同域 IID）**

文献里「Non-IID」不是一种实验。按 **怎么把数据分到 client** 可以分成四档；FedLLM 论文几乎都落在 A/B，真正「一端一个领域」是 C，相对少、但是有人做。

---

## 1. 四档怎么造

| 档 | 名字 | 怎么分 | 异质从哪来 | 经典出处 |
|---|---|---|---|---|
| **A** | 同域 IID / 近 IID | 一份数据随机切成均等 shard | 几乎只有采样噪声 | FedAvg 对照；FlowerTune 各赛道 |
| **B** | 同域统计 Non-IID | 仍是 **一份** 数据；Dirichlet / 按 label 或 category 切 | \(P(y)\) 或任务类别不同 | Hsu et al. 2019 Dirichlet；Kairouz 2019 综述；LEAF |
| **C** | 跨源 / 跨域 | **每个 client 一个数据集**（医学 / 通用 / 金融 / 代码） | \(P(x)\) 领域不同 | OpenFedLLM 第二种划分 |
| **D** | 自然用户划分 | 按写者、角色、机构主键，不人造切 | 真实生成过程 | LEAF FEMNIST / Shakespeare |

Kairouz et al., *Advances and Open Problems in Federated Learning* (arXiv:1912.04977) 把异质拆成：特征偏移（covariate shift）、标签偏移（label skew）、概念漂移、数量不均。图像联邦实验 **绝大多数只做标签 Dirichlet**（B）。LLM 指令微调没有干净的「10 类标签」，所以 B 往往改成 **按 Dolly 的 8 个 category 做 Dirichlet / 每 client 缺若干类**。

你们要测的「不同领域」是 **C**，不是 B。仓库里 `split_federated_data.py --method dirichlet` 是 B。

---

## 2. 经典 FL（非 LLM）用什么数据

| 工作 | 数据 | Non-IID 做法 |
|---|---|---|
| McMahan et al. FedAvg 2017 | MNIST / Shakespeare | 按用户或按数字病理划分 |
| Hsu, Qi, Brown 2019 | CIFAR 视觉分类 | 每 client 类别比例 \(\sim \mathrm{Dir}(\alpha p)\)，\(\alpha\) 越小越 Non-IID |
| Caldas et al. LEAF 2018 | FEMNIST、Shakespeare、Reddit 等 | **按写者 / 角色** 的自然划分，不是把 CIFAR 再切 |
| Li et al. 2020 综述；Zhu et al. 2021 Non-IID survey | CIFAR / FEMNIST 为主 | 标签不平衡、数量不平衡；领域偏移讨论少、实验更少 |
| NIID-Bench (Li et al.) | CIFAR、FEMNIST、tabular | `noniid-labeldir`、每 client 固定 k 类、quantity skew |

结论：传统论文说 Non-IID，默认是 **同一任务、标签分布不同**。没有「一边 ImageNet 一边 CIFAR」当标准设定。

---

## 3. FedLLM / 联邦指令微调：他们具体怎么做

### 3.1 FedIT / Shepherd（Zhang et al., 2023）

- 论文：*Towards Building the Federated GPT: Federated Instruction Tuning*（[arXiv:2305.05644](https://arxiv.org/abs/2305.05644)）
- **数据**：几乎只用 **`databricks/databricks-dolly-15k`**（约 15k，8 个 InstructGPT 风格类别）
- **划分**：切成 **10 个 client**；代码里 `alpha=0.5` 按 **category 做 Dirichlet**，或 2-shard 随机。图上可见有的 client 缺整类任务
- **评估**：看联邦后是否比「只在本 client 上训」更好（多样性 / 数量）
- 档：**B**（同域、类别不平衡）。这是 Dolly 在 FedLLM 里被用最多的原因之一

### 3.2 FederatedScope-LLM（Kuang et al., 2023）

- 论文：[arXiv:2309.00363](https://arxiv.org/abs/2309.00363)；内置 `dolly-15k@llm`、`alpaca@llm`、`rosetta_alpaca@llm`、`code_search_net@llm`
- **划分**：IID / **LDA（Dirichlet）** / **MetaSplitter（按元信息）**
- Fed-Dolly：按任务类别拆，常做成 **8 client × 每端一类任务**
- Fed-Code：按编程语言拆
- 档：**B**，用 category/语言当「标签」

### 3.3 OpenFedLLM（Ye et al., KDD 2024）

- 论文：[arXiv:2402.06954](https://arxiv.org/abs/2402.06954)
- **数据表**（指令微调）：Alpaca、Alpaca-GPT4（通用）、FinGPT（金融）、**MedAlpaca / medical flashcards（医学）**、Code-Alpaca（代码）、MathInstruct（数学）
- **明确写了两种划分**：
  1. 一份数据随机切给多 client → **同域、同源**（A/弱 B）
  2. **每个 client 分到不同数据集** → **不同源**（**C**）
- 主实验很多仍是：Alpaca-GPT4 或 MedAlpaca，**20 client 随机切，每轮 2 个参与**（偏 A）
- 评估：通用用 MMLU / MT-Bench 等；医学用 MedQA、PubMedQA、MedMCQA、MMLU 医学子集——**按训练域选下游榜**，不是把所有域平均成一条 loss
- 这是和「ICC1 医学、ICC2 通用」**同一类设定** 的主要文献锚点

### 3.4 FlowerTune LLM Leaderboard（Gao et al., 2025）

- 论文：[arXiv:2506.02961](https://arxiv.org/abs/2506.02961)
- **四个独立赛道**（不是同一场联邦里混四个域）：

| 赛道 | 训练数据 | client 切法 | 评估 |
|---|---|---|---|
| General NLP | `alpaca-gpt4` | 近似均等 shard（文中为机构模拟，**赛道内近 IID**） | MMLU |
| Finance | FinGPT sentiment | 同上 | FPB / FIQA / TFNS |
| Medical | **medical-flashcards**（和你们同一来源） | 20 client 均等 | PubMedQA、MedMCQA、MedQA、CareQA |
| Code | Code-Alpaca-20k | 10 client 均等 | HumanEval、MBPP、MultiPL-E |

注意：FlowerTune 的「跨域」是 **四个分开的联邦任务**，不是「同一轮里一半医院一半代码公司」。每条赛道内部更接近 **A**。医学评估用 **公开医学 QA 榜**，训练却是闪卡——和你们「训练闪卡、eval 也是闪卡 hold-out」不完全一样。

### 3.5 其它

| 工作 | 数据 | 划分 |
|---|---|---|
| FedAMoLE 等个性化 FedLLM | Dolly-15k、Natural Instructions、SNLI | 同集 Non-IID 切；强调 **每 client 自己的 in-domain test** |
| TuneInsight federated-llms（医学隐私） | Flashcards、PubMedQA、MedMCQA **三个医学集** | **3 个参与方各用一个医学集**（同领域、题型不同，弱 C） |
| npj Digital Medicine Fed-MedLoRA 等 | 多医院临床笔记 | 真实跨机构、仍是医学域 |

---

## 4. 常用数据集速查（指令微调）

| 数据集 | 领域 | 约条数 | 谁在用 | 当 Non-IID 的哪一档 |
|---|---|---|---|---|
| Dolly-15k | 通用指令（8 类） | 15k | FedIT、FS-LLM、Shepherd | B：按 category；很少单独当「通用侧」去配医学 |
| Alpaca / Alpaca-GPT4 | 通用合成指令 | 52k | OpenFedLLM、FlowerTune General | A 切 shard；0.5B 上你们已踩过容量墙 |
| medical_flashcards / MedAlpaca | 医学 QA | ~34k | OpenFedLLM 医学实验、FlowerTune Medical、**本仓库** | A 切 shard；C 里当「医学那一端」 |
| Code-Alpaca-20k | 代码 | 20k | OpenFedLLM、FlowerTune Code | C 的代码端；或 B 按语言 |
| FinGPT sentiment 等 | 金融 | 不等 | OpenFedLLM、FlowerTune Finance | C 的金融端 |
| GSM8K / MathInstruct | 数学 | 大 | FS-LLM、OpenFedLLM | 较少和医学配对 |
| Natural Instructions | 多任务 NLP | 大 | 个性化 FedLLM | B：按 task 分 client |

---

## 5. 评估他们怎么报（跨域时最容易抄错）

- **A/B（同域）**：可以报一份全局 eval（MMLU 或该域 QA）。FedIT 还报「联邦 vs 只训本地」。
- **C（跨域）**：OpenFedLLM / FlowerTune 都是 **按域选下游榜**（医学就 MedQA 族，代码就 HumanEval）。个性化工作明确 **每 client 一份 in-domain test**。
- **几乎没有人** 把「医学 eval + 通用 eval」平均成一条曲线当主指标。

你们若做 C：ICC1 医学 hold-out、ICC2 Dolly hold-out，分开记；可选再加公开医学 QA 作对照。不要沿用现在的单一 `medical_flashcards_eval` 平均。

---

## 6. 和本仓库怎么对齐

| 你们想证明的 | 应对齐的档 | 数据 | 最接近的论文 |
|---|---|---|---|
| 协议在「真实一点的异质」下仍能训 | **C** | ICC1 闪卡 + ICC2 Dolly（或 Code-Alpaca） | OpenFedLLM 第二种划分 |
| 只复现「论文里最常见的 Non-IID」 | **B** | 仍用闪卡 Dirichlet，或 **只把 Dolly 按 8 类切成两端** | FedIT、FS-LLM |
| 和现有 0.985 数字可比 | **A** | 维持现状 50/50 闪卡 | 你们 `202609181406` |

**建议（与执行计划一致）**：主实验走 **C：闪卡 + Dolly**。Dolly 在文献里主要是 B 的载体，拿来当 C 的「通用侧」是合理迁移（OpenFedLLM 通用侧常用 Alpaca；你们 0.5B 换 Dolly 是容量原因）。若审稿人问「标准 Non-IID」，用已有 Dirichlet 脚本补一条 **B 对照**，成本低。

不建议第一枪：Alpaca-GPT4（S1-C 容量墙）、中英混合、同一联邦里塞四个 FlowerTune 域（client 数和评估都会炸）。

---

## 7. 写论文时：通用做法 vs 建议用的数据

审稿人说的「Non-IID」默认是 **B：同一份公开指令集，按类别/Dirichlet 切开**。只做「一边医学一边 Dolly」会被认为是 **两个任务的联邦**，不是社区最熟的那张表。论文里两档都要有。

### 7.1 社区通用方式（按出现频率）

1. **默认主表（几乎每篇 FedLLM 都有）**  
   拿 **一份** 标准指令数据，切成 N 个 client。  
   - 数据首选 **Dolly-15k**（FedIT / Shepherd / FS-LLM：按 8 个 `category` Dirichlet，常见 \(\alpha=0.5\)）  
   - 或 **Alpaca-GPT4**（OpenFedLLM / FlowerTune General：随机切 20 client，每轮抽 2 个）  
   - 对照：**同一份数据的 IID 均分**  
   - 评估：该域的公开榜（通用 MMLU / MT-Bench；不要只报训练 hold-out loss）

2. **跨机构 / 跨域（想讲「医院 vs 别的行业」才加）**  
   OpenFedLLM **第二种划分**：每个 client 一个公开数据集。常用组合就是他们表 2 那几个名字：  
   **MedAlpaca/闪卡、Alpaca 或 Dolly、Code-Alpaca、FinGPT**。  
   FlowerTune 则是 **四个赛道分开跑**（医学闪卡 / alpaca-gpt4 / FinGPT / Code-Alpaca），赛道内仍是均分，**不是**同一轮里混四个域。  
   评估：**按域分开报**（医学 MedQA/PubMedQA/MedMCQA；代码 HumanEval；通用 MMLU）。

3. **不要当成「标准 Non-IID」的**：只把「一边一个完全不同的任务」当作唯一实验；或只报一条平均 LM loss。

### 7.2 若你们后面写论文，数据建议

设定是 **2 个 cross-silo 机构 + 公开 mask + SecAgg**，不要假装 OpenFedLLM 的 20 client。表格可以小，但 **协议要眼熟**。

| 表 | 目的 | 数据 | 切法 | 评估 |
|---|---|---|---|---|
| **主 Non-IID（审稿人预期）** | 对齐 FedIT | **Dolly-15k** | 2 client：IID 50/50 vs 按 `category` Dirichlet \(\alpha=0.5\) | 每端本类 hold-out；有余力再加轻量通用 QA |
| **跨域（你们的故事）** | 对齐 OpenFedLLM 第二种 | **ICC1：medical_flashcards**（FlowerTune Medical 同源）**ICC2：Dolly-15k** | 一端一个源 | 医学 hold-out **和** PubMedQA/MedMCQA 之一；Dolly hold-out **分开报** |
| **已有基线** | 协议没坏 | 闪卡 50/50 IID | 现状 | 现有 eval loss / 0.985 |

模型变大（3B/7B）之后，通用侧可把 Dolly 换成 **Alpaca-GPT4**（和 OpenFedLLM/FlowerTune 完全同名）；0.5B 全参先不要换。代码域（Code-Alpaca）作 **附录/第二跨域**，不要当第一张主表。

**一句话**：论文通用方式是 **Dolly（或 Alpaca）同集 Dirichlet + 同集 IID**；跨域是加一张 **闪卡 vs Dolly/Alpaca**。执行计划已改为 **先做 Dolly 的 B 档，跨机构后置**。
