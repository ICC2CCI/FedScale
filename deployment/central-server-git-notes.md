# Central Server 文档 / Git 协作说明

目标：本机部署记录与 ICC 上游（`ICC2CCI/FedScale`）更新尽量不打架。

## 原则

1. **少改共享规划文档**  
   如 `docs/algorithm/2026-09-09-dual-cluster-fsdp-deployment.md` 由 ICC 维护架构决策。  
   本机「已部署 / IP / 密钥路径」写在 `deployment/central-server-*.md`，不要大段改规划正文与 checklist。

2. **优先新增文件，少改已有文件中间段落**  
   新增 `deployment/docker-compose.yml`、`scripts/docker-*.sh`、本系列 md 多为**追加**，与上游并行改同一文件的概率低。  
   若必须改已有文件（如 `deployment/README.md`），只在文末或表格**追加一行链接**。

3. **密钥与运行时不出库**  
   已 gitignore：`deployment/central-server.env`、`data/minio/`、`tools/minio/`、`logs/`。  
   只提交 `central-server.env.example`。

4. **用独立分支提交可共享改动**  
   ```bash
   git fetch origin
   git checkout -b deploy/central-server-minio origin/main
   # 只 add 可共享文件（compose、example env、脚本、通用说明）
   git pull --rebase origin main   # 推送前再同步一次
   ```
   不要在跟踪的 `main` 上堆本机临时改动。

5. **拉上游时**  
   ```bash
   git fetch origin
   git rebase origin/main    # 或 merge；有冲突时优先保留上游规划文档，本机状态留在 central-server-prep.md
   ```

## 建议提交 / 不提交

| 路径 | 建议 |
|---|---|
| `deployment/docker-compose.yml` | 可提交（通用） |
| `deployment/central-server.env.example` | 可提交 |
| `scripts/docker-central-*.sh`、`install-docker.sh` | 可提交 |
| `deployment/central-server-prep.md` | 可提交，但含本机 IP；若不想暴露可改成模板后提交 |
| `deployment/central-server.env` | **不提交** |
| `data/minio/`、`tools/minio/`、`logs/` | **不提交** |
| `docs/algorithm/2026-09-09-*.md` | **尽量不改**；有进展用 PR 小幅补日志，或等 ICC 合并后再记 |

## 若仍发生冲突

- 冲突在 `docs/algorithm/...`：接受上游（`ours`/`theirs` 按 rebase 方向），把本机进度写回 `central-server-prep.md`。
- 冲突在 `.gitignore`：两边条目合并保留即可。
- 冲突在 `deployment/README.md`：保留双方链接行。
