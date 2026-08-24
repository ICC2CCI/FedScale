# REST Fleet 传输切换

当全参数联邦训练通过跨中心 Skupper 通道回传多 GB 模型状态时，默认 gRPC
Fleet API 可能出现 `GOAWAY ... ping_timeout`。本目录中的
`Dockerfile.superlink-rest` 为 `flwr/superlink:1.28.0` 添加 Flower REST Fleet
API 所需的 `starlette` 与 `uvicorn` 依赖。

## 构建与发布

本项目默认使用中心端集群内持久化 Registry，地址为
`172.16.18.27:5000`。该 ClusterIP 不对公网暴露。首次部署 Registry 与为
中心端 containerd 配置该受信任 HTTP 源的步骤见：

- `cluster-image-registry.yaml`
- `containerd-private-registry-configure.yaml`

发布镜像时，先建立到 Registry 的本机端口转发，再执行：

```bash
docker build -f configs/Dockerfile.superlink-rest \
  -t localhost:5000/flwr_superlink_rest:1.28.0-1 .
docker push localhost:5000/flwr_superlink_rest:1.28.0-1
```

## 切换要求

不要在存在运行中的 Flower 任务时切换。当前 `superlink-deployment.yaml` 和
两个 SuperNode Deployment 已包含下列 REST 参数：

```yaml
- "--fleet-api-type"
- "rest"
- "--fleet-api-address"
- "0.0.0.0:9092"
```

将两个 SuperNode 的参数改为：

```yaml
- "--rest"
- "--superlink"
- "http://fleet-api:9092"
```

完成滚动发布后，先提交一个小型联邦烟雾实验，确认两端注册、任务下发、对象上传
和聚合均成功，再启动全参数 FSDP/DDP 对照。
