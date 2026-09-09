#!/usr/bin/env bash
# Central Server 一键准备检查 / 启动入口
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/deployment/central-server.env"
# shellcheck disable=SC1090
source "${ENV_FILE}"

echo "=== Central Server 角色准备 ==="
echo "IP: ${CENTRAL_SERVER_IP}"
echo "Repo: ${ROOT}"
echo

echo "[1] 硬件"
echo "  CPU cores: $(nproc)"
free -h | awk '/Mem:/ {print "  Memory: "$2" total, "$7" available"}'
df -h "${ROOT}" | awk 'NR==2 {print "  Disk avail: "$4" on "$6}'
echo

echo "[2] MinIO 二进制"
if [[ -x "${MINIO_BIN}" ]]; then
  "${MINIO_BIN}" --version 2>&1 | head -1
else
  echo "  MISSING: ${MINIO_BIN}"
fi
if [[ -x "${MC_BIN}" ]]; then
  "${MC_BIN}" --version 2>&1 | head -1
else
  echo "  MISSING: ${MC_BIN}"
fi
echo

echo "[3] Python 环境 fedscale-server"
if [[ -x /home/pcllgr/miniconda3/envs/fedscale-server/bin/python ]]; then
  /home/pcllgr/miniconda3/envs/fedscale-server/bin/python -c \
    "import fastapi,uvicorn,boto3,minio,torch; print('  ok torch', torch.__version__)"
else
  echo "  MISSING conda env fedscale-server"
fi
echo

echo "[4] 代码占位"
for f in \
  experiments/server/aggregation_server.py \
  experiments/server/minio_client.py \
  experiments/server/block_scheduler.py \
  experiments/shared/block_selection.py \
  experiments/shared/protocol.py
do
  if [[ -f "${ROOT}/${f}" ]]; then
    echo "  present: ${f}"
  else
    echo "  TODO:    ${f}"
  fi
done
echo

echo "[5] 端口占用"
ss -lnt | grep -E ":${AGGREGATION_PORT}|:${MINIO_API_PORT}|:${MINIO_CONSOLE_PORT}|:${DASHBOARD_PORT}" || echo "  目标端口空闲"
echo

echo "[6] 对客户端地址"
echo "  SERVER_URL=${SERVER_URL}"
echo "  MINIO_ENDPOINT=${MINIO_ENDPOINT}"
echo "  MINIO user=${MINIO_ROOT_USER}"
echo

echo "常用命令："
echo "  bash scripts/minio.sh start"
echo "  bash scripts/minio.sh init-bucket"
echo "  bash scripts/minio.sh status"
echo "  # 聚合服务（代码就绪后）："
echo "  conda activate fedscale-server"
echo "  python experiments/server/aggregation_server.py --port ${AGGREGATION_PORT} --minio-endpoint http://127.0.0.1:${MINIO_API_PORT} --num-clients ${NUM_CLIENTS} --num-rounds ${NUM_ROUNDS} --ratio ${RATIO}"
