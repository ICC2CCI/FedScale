# 联邦实验对比看板

这个轻量服务直接读取 Flower ServerApp 的实验结果目录，提供：

- 左右两个实验 ID 选择器；
- 按 `experiment-id` 查看联邦运行日志：提交信息、运行尝试、每轮聚合完成、当前状态与最终汇总；运行中默认每 10 秒自动刷新；
- 正式对照参数逐项一致性检查；
- 对两侧都已持久化联邦轮次或中间指标、但尚未完成的实验提供“阶段性结果对照”，并与最终胜负结论明确区分；
- 客户端关键路径、训练阶段、平均耗时、压缩耗时和 loss 对比；
- 基于 `critical_path_client_round_seconds` 的自动胜负与提速比例；
- 对可能包含集群排队的 Run 端到端耗时作单独提示。
- “TKE 云端结果”数据源，实时读取 A/B 客户端的逐步耗时、吞吐量、GPU/CPU 内存和利用率。
- “实验监督器控制台”，可配置分布式训练模式（FSDP/FedScale/DDP）、联邦训练轮数、基座模型、训练数据集和微调方式（LoRA/全参数），同时调整轮询间隔、卡住判定阈值、失败重试次数，并启动监督器或发送暂停/恢复/停止指令。
- ServerApp 的 `experiment_events.jsonl` 与监督器的 `.supervisor-events.jsonl` 会被合并成细粒度时间线；运行期间每 30 秒写入心跳，即使尚未完成一轮也能看到当前阶段。
- “联邦聚合端：集群内训练与资源”面板：由中心端汇集同一联邦轮 TKE-A/B 的训练结果，比较前向、反向、集群内通信、优化器更新、step、总训练时间、吞吐量，以及 GPU/CPU 资源使用。
- “逐轮联邦时间链路”面板：按联邦轮次对齐 TKE-A/B FSDP 训练、客户端整轮、Top-K 压缩、联邦周期、中心聚合后处理和全局 checkpoint 间隔；缺失的历史字段显示为“—”。
- 训练端新实验会额外记录容器网络流量、DDP NCCL bucket 的通信耗时/字节数，以及 FSDP full-state 导出、状态转换、序列化耗时和导出大小。

页面服务本身不需要 npm；SwanLab 同步器使用看板镜像内固定版本的 Python SDK。
当前集群部署通过 `GPU_MODEL=NVIDIA V100` 标注共享硬件；若部署到其他集群，请修改该环境变量。

## SwanLab 私有化实验追踪

看板会只读扫描中心端结果 PVC，并将运行配置、联邦轮次耗时、客户端聚合指标、最终评估指标和终态异步写入 SwanLab。不会上传原始数据、模型参数或 checkpoint；SwanLab 不可用时不会影响 Flower 实验。

当前采用 Kubernetes 私有化部署方式：固定 `swanlab/self-hosted` Helm Chart `0.6.2`，使用 chart 管理的 MinIO 和 CBS PVC，所有数据留在 `flower-superlink` 命名空间。部署前可先渲染检查：

```bash
helm template swanlab-self-hosted swanlab/self-hosted \
  --version 0.6.2 \
  --namespace flower-superlink \
  -f flower-llm/configs/swanlab-values.yaml
```

确认后安装：

```bash
cd flower-llm
bash scripts/deploy-swanlab.sh
```

SwanLab 默认是 ClusterIP，仅通过端口转发访问：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig config-center \
  -n flower-superlink port-forward svc/swanlab-self-hosted 8081:80
```

首次登录 SwanLab 页面后创建 API Key，再以 Secret 注入同步器：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig config-center \
  -n flower-superlink create secret generic swanlab-api-key \
  --from-literal=api-key='<SwanLab API Key>' \
  --dry-run=client -o yaml | \
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig config-center apply -f -
```

然后重新部署实验看板，使它读取 Secret：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig config-center \
  -n flower-superlink rollout restart deployment/experiment-dashboard
```

看板镜像单独安装 `swanlab==0.9.4` 和 Kubernetes Python client，不修改 Flower 训练镜像；构建并推送看板镜像后，`experiment-dashboard.yaml` 中的 `SWANLAB_API_HOST` 指向集群内部 `swanlab-self-hosted` Service。看板项目名使用 `DASHBOARD_SWANLAB_PROJECT`，不要设置 `SWANLAB_PROJECT`，后者是 SDK 保留的嵌套配置变量。

```bash
python3 server.py --host 0.0.0.0 --results-root ../evaluation-results --port 8080
```

在浏览器中使用服务器地址访问，例如：

```text
http://<服务器地址>:8080/?source=cloud&left=fsdp-full-topk-3b-s512-s10-gc-pair-r1&right=ddp-full-cpuoffload-topk-3b-s512-s10-gc-pair-r2
```

## 本地启动

仓库现有导出结果可以直接作为中心数据源：

```bash
cd flower-llm/experiment-dashboard
python3 server.py --results-root ../evaluation-results --port 8080
```

访问 `http://127.0.0.1:8080`。也可以通过查询参数固定一组实验：

```text
http://127.0.0.1:8080/?source=cloud&left=<FSDP_ID>&right=<DDP_ID>
```

在本仓库机器上启动时，服务会自动寻找 `../config-center`、`../config-tke-a`、`../config-tke-b`，因此可以直接切换到“TKE 云端结果”。该模式会显示聚合端视角的集群内训练性能与资源汇总，并保留每个 TKE 客户端的原始明细用于追溯。每次刷新通过无代理 `kubectl exec` 读取 JSON 指标，缓存30秒，不传输模型权重或 checkpoint。

前向、反向、集群内通信、优化器更新等细分项来自新生成的 `metrics_detailed.json`；联邦时间链路同时使用中心端的 `federated_metrics_round_*.json` 和 TKE 客户端的逐轮 `metrics_detailed.json`。旧实验若没有对应字段，看板会显示“—”，不会用零值伪造测量结果。
网络/NCCL 与状态导出也遵循相同规则；历史实验无法从 checkpoint 或汇总耗时反推出精确的 NCCL/导出耗时，需用更新后的训练镜像重新运行。

## API

| 接口 | 说明 |
|---|---|
| `GET /api/health` | 服务和结果目录状态 |
| `GET /api/sources` | 本地中心结果和 TKE 云端数据源状态 |
| `GET /api/experiments?source=center\|cloud` | 指定数据源的实验列表 |
| `GET /api/experiments/<id>` | 单个实验的标准化参数和指标 |
| `GET /api/experiments/<id>/logs?source=center\|cloud` | 单个实验的结构化联邦运行日志；云端模式通过 `kubectl exec` 读取中心结果 PVC |
| `GET /api/compare?source=<source>&left=<id>&right=<id>` | 两个实验的参数与实测结果对比 |
| `GET /api/supervisor?source=cloud` | 读取监督器控制、心跳和最近事件 |
| `POST /api/supervisor?source=cloud` | 写入监督器控制；支持 `configure`、`start`、`pause`、`resume`、`stop`、`retry`；`start` 会创建新的监督器 Job |

监督器控制请求示例：

```json
{
  "action": "configure",
  "poll_seconds": 30,
  "stall_seconds": 7200,
  "max_restarts": 3,
  "strategy": "fsdp",
  "rounds": 10,
  "model": "Qwen/Qwen2.5-7B",
  "dataset": "HuggingFaceH4/ultrachat_200k",
  "finetuning_type": "lora"
}
```

`configure` 保存的是下一次提交 Run 的实验配置，不会改写当前正在运行的 Run；`start` 会从最近一次监督器 Job 复制运行模板，并拒绝重复启动仍处于活动或等待状态的 Job；`pause` 只暂停后续调度，不会强行中断当前 Run；`stop` 才会请求停止当前 Flower Run。字段由服务端校验：策略只能是 `fsdp`、`fedscale` 或 `ddp`，轮数为 1—1000，模型和数据集使用 Hugging Face `组织名/仓库名` 格式。

## 容器与集群部署

### 最小认证配置

先生成 PBKDF2 密码哈希。命令只输出哈希，不会把明文密码写入文件：

```bash
AUTH_HASH=$(python3 flower-llm/scripts/generate-dashboard-auth-hash.py)
```

在看板所在的 namespace 创建 Secret。部署在中心集群时使用 `flower-superlink`；部署在 TKE-A 时使用 `flower-supernode-a`：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a \
  -n flower-supernode-a create secret generic experiment-dashboard-auth \
  --from-literal=username=lab-operator \
  --from-literal=password-hash="$AUTH_HASH" \
  --dry-run=client -o yaml | \
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a apply -f -
```

清单会从这个 Secret 注入 `DASHBOARD_AUTH_USERNAME` 和 `DASHBOARD_AUTH_PASSWORD_HASH`。`/api/health` 为 Kubernetes 探针保留未认证访问，其余页面、日志接口和监督器控制接口都要求 Basic Auth；生产环境必须再通过 HTTPS 暴露，避免密码被明文传输。

```bash
docker build -f flower-llm/experiment-dashboard/Dockerfile \
  -t ccr.ccs.tencentyun.com/flwr_pcl/flower-experiment-dashboard:control-console-v1 flower-llm
docker push ccr.ccs.tencentyun.com/flwr_pcl/flower-experiment-dashboard:control-console-v1
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-center \
  -n flower-superlink create secret generic experiment-dashboard-kubeconfigs \
  --from-file=config-center=flower-llm/config-center \
  --from-file=config-tke-a=flower-llm/config-tke-a \
  --from-file=config-tke-b=flower-llm/config-tke-b
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-center apply \
  -f flower-llm/configs/experiment-dashboard.yaml
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-center \
  -n flower-superlink \
  port-forward svc/experiment-dashboard 8080:8080
```

如果希望把看板放在 TKE-A，使用 `flower-llm/configs/experiment-dashboard-tke-a.yaml`。它只使用 `cloud` 数据源，通过三份 kubeconfig 直接访问中心和 TKE A/B；先在 `flower-supernode-a` 创建同名 Secret：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a \
  -n flower-supernode-a create secret generic experiment-dashboard-kubeconfigs \
  --from-file=config-center=flower-llm/config-center \
  --from-file=config-tke-a=flower-llm/config-tke-a \
  --from-file=config-tke-b=flower-llm/config-tke-b \
  --dry-run=client -o yaml | \
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a apply -f -

env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a \
  apply -f flower-llm/configs/experiment-dashboard-tke-a.yaml
```

TKE 部署要使用包含本次控制台代码的新镜像，并确保中心 kubeconfig 对 `superexec-serverapp` 具有 `get`、`list`、`create`、`exec` 所需权限；看板通过中心 Pod 内的 Python 原子写入控制文件，不直接修改训练过程文件。

看板使用 Kubernetes Python client 访问三套 kubeconfig，不需要在镜像中额外安装 `kubectl`。

部署清单将 `superexec-serverapp-results-pvc` 以只读方式挂载到 `/app/results`，不会修改 checkpoint 或实验记录。云端 kubeconfig 使用独立 Secret 只读挂载；若不创建该 Secret，中心结果仍可使用，但“TKE 云端”按钮会自动禁用。

## 测试

```bash
cd flower-llm/experiment-dashboard
python3 -m unittest discover -s tests -v
```
