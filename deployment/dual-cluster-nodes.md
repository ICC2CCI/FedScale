# 双集群节点约定（Central + ICC1 + ICC2）

对应规划：`docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md`

| 节点 | 角色 | 示例主机 | 仓库目录 | client-id | 训练数据 |
|---|---|---|---|---|---|
| Central Server | 聚合 + MinIO | `192.168.235.42` | `fedscale-icc-server` | — | — |
| ICC1 | Client 0 | `192.168.206.116` (`dgx1-16`) | `fedscale-icc-1` | `0` | `data/splits/icc1_client0_train.json` |
| ICC2 | Client 1 | `192.168.205.130` (`dgx1-30`) | `fedscale-icc-2` | `1` | `data/splits/client1_train.json` |

更细的本机路径见：

- [`ICC1_PATHS.md`](../ICC1_PATHS.md)
- [`ICC2_PREP.md`](../ICC2_PREP.md)
- [`central-server-prep.md`](central-server-prep.md)

## 启动顺序

1. Central：Docker MinIO + `aggregation_server.py`
2. （首次）ICC 上 `scripts/bootstrap_initial_state.sh` 上传 round-0
3. ICC1 / ICC2：`accelerate launch ... experiments/run_s3r12v3_fsdp.py`
