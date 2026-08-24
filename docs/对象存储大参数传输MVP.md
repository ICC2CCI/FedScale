# 大参数对象存储传输 MVP

此模式解决完整 FP16 模型（OpenLLaMA-3B 约 6.85GB）不能稳定经过 Flower/Skupper
控制通道的问题：Flower 只传递小型 `ConfigRecord` 元数据，A/B 客户端直接上传模型对象；
中心端启动 64Gi 内存的临时聚合 Job，完成后发布下一轮 `global.pt`。这不是加密安全聚合，
也不改变本地训练的数据不出域边界。

## 先决条件

1. 在 `flower-superlink` 部署独立 100Gi MinIO（或使用 COS/S3）：
   `configs/object-store-minio.yaml`。不要复用 SwanLab 的 20Gi 存储。
2. 对象存储必须经三个集群均可访问的**私有 TLS** 地址暴露。`ClusterIP` 只对中心集群
   可见，不能直接填入 A/B 的 `S3_ENDPOINT`。
3. 从 `configs/object-store-secrets.example.yaml` 复制并创建四类 Secret：MinIO root、
   A、B、Server/Aggregator。A/B 的写权限须分别限定为客户端对象；Aggregator 才能写
   `global.pt` 与 `round_state.json`。
4. 构建并推送 `flowertune-llm/Dockerfile.aggregation` 指定的聚合镜像；将其完整地址写入
   `OBJECT_STORE_AGGREGATION_IMAGE`。应用 `configs/object-store-server-rbac.yaml` 后，才
   可滚动更新 ServerApp/A/B ClientApp Deployment。

首次部署还要使用受控管理员凭据创建 `fl-models` bucket，并创建受限的 A/B、Server、
Aggregator S3 用户/策略；MinIO Deployment 不会自动创建 bucket。seed 工具为
`python -m flowertune_llm.seed_global_model --model ... --experiment-id ...`，应在挂载了
完整模型缓存、具有足够 CPU/RAM 与 Aggregator Secret 的一次性 Job 中运行。

部署清单只提供中心集群内 MinIO Service，刻意没有提供可直接上线的公网 Ingress。实际
环境必须由现有私网入口、跨集群网络和证书体系提供 TLS；不要使用 HTTP endpoint 或将
MinIO root 密码放入 Git。

## 首个全局模型与启动

先由受控管理/seed Job 将与 A/B 完全相同基座 snapshot 导出的 FP16 federation state 上传至
`s3://fl-models/<experiment-id>/round-0/global.pt`，并写入 SHA-256 metadata。不能把
HuggingFace 原始权重目录直接当作该对象；格式必须是 `torch.save` 的 state dict，与客户端
训练 Job 导出的 `model_weights.pt` 一致。

启动时使用显式参数；脚本会在申请 GPU 前执行配置编译/校验：

```bash
./scripts/run-federated.sh \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp --dataset vicgalle/alpaca-gpt4 \
  --finetuning-type full --quantization 0 --rounds 2 --save-every-round 1 \
  --experiment-id object-store-smoke \
  --object-store-initial-global-uri \
    s3://fl-models/object-store-smoke/round-0/global.pt \
  --set dataset.max-train-samples=3334 \
  --set train.training-arguments.max-steps=10 \
  --set train.evaluate-after-fit=false
```

对象存储模式固定 `train.full-update-compression=none`，不能与 `fedscale-int8`、Top-K 或
`--full-local-init` 混用。恢复第 N 轮时将 `--resume-round N` 和该轮 `global.pt` URI 一起传入；
同一轮已有 `GLOBAL_READY` 状态时聚合 Job 会复用它，不会重复聚合。

## 验收与边界

- 每轮都有 `round_state.json`，状态只能从 `AGGREGATING` 变为 `GLOBAL_READY`。
- 每次下载在 SHA-256 与对象大小校验通过后才原子替换本地 checkpoint。
- 聚合 Job 失败时两个客户端对象会保留，后续可以诊断或安全重试；ServerApp 不会在 28Gi
  内存中加载两个模型。
- 当前 MVP 是两客户端、全量 `.pt`、内存 FedAvg，最终评估仍需独立发起；生产演进再做
  safetensors、流式 tensor 聚合、对象生命周期与更强的身份/密钥管理。
