# Flower 联邦训练运维脚本

所有脚本都会从仓库中定位三个 kubeconfig，并在调用 `kubectl` 时临时清除
本机代理变量。可从任意工作目录运行。

| 脚本 | 用途 |
| --- | --- |
| `port-forward.sh` | 在本机 9093 端口建立 Flower 控制连接；使用 Flower CLI 时保持运行。 |
| `preflight.sh MODEL` | 检查中心端、A/B 端 Deployment 就绪及指定模型的完整缓存。 |
| `warm-model-cache.sh MODEL [--target center\|a\|b\|all]` | 预热缺失模型缓存；`preflight.sh` 默认自动调用。 |
| `run-federated.sh` | 提交可配置的联邦微调实验；默认先执行预检，失败则拒绝提交。 |
| `run-resilient-federated.sh` | 从每轮持久化的 LoRA 或全参数 checkpoint 自动恢复失败实验，带锁、退避与重试上限。 |
| `status.sh` | 查看中心端、A 端、B 端的 Deployment、Job、Pod 和 Service。 |
| `list-runs.sh` | 列出 Flower Run。 |
| `logs.sh {server\|a\|b\|all} [--follow]` | 查看 ServerApp 或 ClientApp 日志。 |
| `stop-run.sh RUN_ID` | 停止指定 Flower Run。 |
| `delete-training-job.sh {a\|b} JOB --yes` | 删除一个明确指定的临时 `train-round-*` Job。 |
| `restart-clientapps.sh --yes` | 滚动重启 A/B 两端 ClientApp。 |
| `list-results.sh` | 汇总状态、策略、模型、耗时、LoRA/全参数 checkpoint 与各轮指标数量。 |
| `export-experiment-report.sh ID [DIR]` | 导出中心 PVC 的实验元数据并在本地生成中文 `REPORT.md`，不复制模型权重。 |

## 启动实验

先在一个终端建立控制连接：

```bash
./scripts/port-forward.sh
```

再在另一个终端提交实验。例如，启动与既有 OpenLLaMA-3B DDP 结果对照的
FSDP + LoRA 实验：

```bash
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp \
  --dataset vicgalle/alpaca-gpt4 \
  --rounds 50
```

DDP 对照实验只需将 `--strategy fsdp` 换为 `--strategy ddp`。提交前可增加
`--dry-run` 查看最终 `run-config`；更多参数（例如局部步数）通过可重复的
`--set KEY=VALUE` 传入：

```bash
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp --rounds 50 \
  --set train.training-arguments.max-steps=50
```

全参数 FP16 FSDP 与 DDP CPU-offload 的短对照使用：

```bash
# FSDP：参数、梯度、优化器状态在 GPU 间 FULL_SHARD
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp --finetuning-type full --quantization 0 \
  --full-local-init \
  --rounds 1 --experiment-id openllama-full-fsdp \
  --set model.gradient-checkpointing=false \
  --set train.training-arguments.per-device-train-batch-size=1 \
  --set train.training-arguments.max-steps=2 \
  --set train.evaluate-after-fit=false

# DDP：完整 FP32 主模型/梯度在 GPU，FP16 AdamW 镜像和状态在 CPU
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy ddp --finetuning-type full --quantization 0 \
  --ddp-cpu-offload --full-local-init \
  --rounds 1 --experiment-id openllama-full-ddp-cpu \
  --set model.gradient-checkpointing=false \
  --set train.training-arguments.per-device-train-batch-size=1 \
  --set train.training-arguments.max-steps=2 \
  --set train.evaluate-after-fit=false
```

全参训练使用 FP32 主参数、FP16 AMP 计算；联邦回传和中心 checkpoint 使用 FP16
完整 state（每客户端约 6.5GB）。`--full-local-init` 仅用于三端 snapshot ID 经预检
完全一致的全新第 1 轮，用本地基座替代初始模型下发，不能用于恢复轮次。首次运行必须
先用 1 轮、2 steps 做内存和传输门禁。当前单卡节点不能通过 TKE 配置生成 NVLink；
需要更换为同机多卡 GPU 实例并用 `nvidia-smi topo -m` 验证。

完整的模型缓存、模板分支和 FSDP/DDP 对照要求见
[联邦实验预检与启动规范](../docs/联邦实验预检与启动规范.md)。

## 自动恢复实验

中心端必须已挂载 `superexec-serverapp-results-pvc`。该脚本会每轮保存
全局 LoRA adapter 或全参数 state；仅在 ServerApp 明确失败或长期未形成新检查点时，才从最近
一个成功检查点提交剩余轮次。它不会重复聚合失败轮次，最多恢复 3 次：

```bash
./scripts/run-resilient-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp --dataset vicgalle/alpaca-gpt4 --rounds 50
```

快速验证完整的 FSDP、双客户端回传、FedAvg 和 checkpoint 链路，使用已实测完成的
两轮冒烟配置：

```bash
./scripts/run-resilient-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp \
  --dataset vicgalle/alpaca-gpt4 \
  --rounds 2 \
  --experiment-id openllama-fsdp-smoke \
  --max-restarts 3 \
  --poll-seconds 60 \
  --stall-seconds 7200 \
  --set dataset.max-train-samples=3000 \
  --set train.training-arguments.max-steps=10 \
  --set train.evaluate-after-fit=false
```

训练实验默认不在阻塞联邦轮次的 ClientApp 内评估模型。历史冒烟 Run 的 52 分 15 秒
包含每轮 5 条样本的验证、生成和指标计算；两轮纯训练关键路径合计约 33 分 1 秒。
需要轮内诊断时再显式设置 `train.evaluate-after-fit=true` 和
`train.num-eval-samples=5`。实验 ID 必须保持唯一；重复运行时请换一个
`--experiment-id`。

每次 ServerApp 尝试都会在中心端 `/app/results/<experiment-id>/` 自动保存：

- `experiment_config.json`：Run ID、模型、数据集、策略和完整有效 run-config；
- `experiment_state.json`：运行状态、最近完成轮次和 checkpoint；
- `experiment_summary.json`：纯训练、评估、客户端轮次及实验端到端耗时，以及客户端训练 loss 加权均值；
- `experiment_attempts.json`：失败、恢复及续跑历史；
- `run_<run-id>_*.json`：每次尝试的不可混淆副本。

韧性脚本完成后会自动将这些小型 JSON 导出到
`evaluation-results/<experiment-id>/` 并生成 `REPORT.md`。也可以手动导出：

```bash
./scripts/export-experiment-report.sh openllama-fsdp-smoke
```

中心 PVC 是权威记录；本地目录是便于浏览、比较和写报告的镜像。adapter 权重只保留
在中心 PVC，除非另有明确的归档需求。

## 监控与恢复

```bash
./scripts/status.sh
./scripts/logs.sh a --follow
./scripts/list-results.sh
```

停止、删除和重启脚本属于恢复操作。会修改运行态资源的脚本要求显式传入
`--yes`，且绝不会删除结果 PVC、模型缓存或任意未指定的 Job。

SuperExec 的 stdout/stderr 会同时写入持久化日志目录：中心端为
`/app/persistent-logs`，A/B 端为各自输出 PVC 下的 `persistent-logs`。Pod 重启后
使用 `kubectl exec` 列出并读取这些带时间戳的日志文件，可保留重启前原因。

## FedScale v1 最小冒烟测试

在已经跑通 DDP/FSDP 联邦训练之前，先用一个不依赖模型下载、GPU 或 Kubernetes 的
小型协议 harness 验证 FedScale v1 的关键数据路径：FSDP rank-local shard 到
canonical block 的流式映射、公共 block mask、裁剪、INT16 wire quantization、
本地 residual、DP noise、aggregate-only 聚合语义和拓扑感知 cohort plan。

```bash
./scripts/run-fedscale-smoke.sh
```

也可以直接调整参数并保存结果：

```bash
python3 scripts/fedscale_smoke.py \
  --rounds 2 --clients 2 --world-size 2 \
  --mask-ratio 0.5 --output /tmp/fedscale-smoke.json
```

该测试的 aggregate-only collector 是 SecAgg+ 的非密码学 test double，只验证固定
向量语义、整数求和和“服务端只保留聚合值”的接口边界；它不构成真实密码学安全证明。
真实 Flower/Kubernetes 接入前，必须用 Flower SecAgg+ 或等价成熟协议替换该 collector。

## FedScale FSDP 跨中心桥接冒烟

协议 harness 通过后，以下命令使用两个真实 ICC、OpenLLaMA-3B 全参数 FSDP 和一轮一
步训练，验证 ServerApp RoundPlan、公共 canonical block INT8 更新、客户端 layout/mask
校验，以及服务端 block-wise 等权聚合：

```bash
./scripts/run-fedscale-fsdp-smoke.sh
```

该命令启用 `train.full-update-compression='fedscale-int8'`，并使用
`train.full-local-initialization=true` 避免首轮传输完整 FP16 基座。它是 bridge smoke，
不是安全聚合或差分隐私实验：结果中的 `fedscale_secagg_enabled=0` 是预期保护栏。
