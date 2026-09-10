#!/usr/bin/env bash
# 启动一轮 S3R12v3 FSDP 双集群训练，结果统一写入 results/YYYYMMDDHHMM/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/deployment/central-server.env"

SSH=(ssh -i "${HOME}/.ssh/id_ed25519" -o StrictHostKeyChecking=no -o BatchMode=yes)
# 使用北京时间命名，便于本地阅读（可用 RUN_ID=... 覆盖）
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date +%Y%m%d%H%M)}"
RUN_DIR="${ROOT}/results/${RUN_ID}"
TAG="${TAG:-s3r12v3-fsdp-delta-fp16}"
TRANSFER_DTYPE="${TRANSFER_DTYPE:-fp16}"

mkdir -p \
  "${RUN_DIR}/figures" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/eval" \
  "${ROOT}/logs"

cat > "${RUN_DIR}/run_meta.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "tag": "${TAG}",
  "created_at": "$(date -Iseconds)",
  "protocol": "S3R12v3 + FSDP + incremental global_delta",
  "transfer_dtype": "${TRANSFER_DTYPE}",
  "num_clients": ${NUM_CLIENTS:-2},
  "num_rounds": ${NUM_ROUNDS:-20},
  "ratio": ${RATIO:-0.2},
  "server": "${SERVER_URL}",
  "minio": "${MINIO_ENDPOINT}",
  "clients": {
    "0": {"host": "pcl@192.168.206.116", "repo": "~/liuchao/fedscale-icc-1"},
    "1": {"host": "pcl@192.168.205.130", "repo": "~/liuchao/fedscale-icc-2"}
  }
}
EOF

cat > "${RUN_DIR}/README.md" <<EOF
# Run ${RUN_ID}

- tag: \`${TAG}\`
- transfer_dtype: \`${TRANSFER_DTYPE}\`（通信 block 落盘精度；auto=跟随模型）
- protocol: S3R12v3 FSDP + \`global_delta\` 增量下发
- artifacts:
  - \`round_log.json\` / \`metrics.jsonl\` — 服务端逐轮指标（train loss、timing、transfer）
  - \`figures/\` — 曲线图（训练结束后或中途 \`plot_s3r12v3_fsdp_run.py\`）
  - \`logs/\` — 聚合服务与两端 client 日志副本
  - \`eval/eval_by_round.json\` — 离线 eval_loss（训练结束后在 ICC1 补算）
EOF

echo "${RUN_DIR}" > "${ROOT}/logs/current_rerun_results_dir.txt"
ln -sfn "${RUN_DIR}" "${ROOT}/results/current"
echo "RUN_DIR=${RUN_DIR}"

# --- stop leftovers (avoid pkill -f self-match on this script) ---
if [[ -f "${ROOT}/logs/aggregation_server.pid" ]]; then
  kill "$(cat "${ROOT}/logs/aggregation_server.pid")" 2>/dev/null || true
  sleep 1
  kill -9 "$(cat "${ROOT}/logs/aggregation_server.pid")" 2>/dev/null || true
fi
"${SSH[@]}" pcl@192.168.206.116 'bash -lc "test -f ~/liuchao/fedscale-icc-1/logs/client0_fsdp.pid && kill -9 \$(cat ~/liuchao/fedscale-icc-1/logs/client0_fsdp.pid) 2>/dev/null || true"' || true
"${SSH[@]}" pcl@192.168.205.130 'bash -lc "test -f ~/liuchao/fedscale-icc-2/logs/client1_fsdp.pid && kill -9 \$(cat ~/liuchao/fedscale-icc-2/logs/client1_fsdp.pid) 2>/dev/null || true"' || true
sleep 2

# --- MinIO ---
if ! curl -fsS --max-time 3 http://127.0.0.1:9000/minio/health/live >/dev/null; then
  echo "MinIO not healthy; starting compose..."
  sg docker -c "cd '${ROOT}/deployment' && docker compose --env-file '${ROOT}/deployment/central-server.env' -f docker-compose.yml up -d"
  sleep 3
fi

# --- aggregation server ---
# rotate previous log into run dir if any
if [[ -f "${ROOT}/logs/aggregation_server.log" ]]; then
  mv "${ROOT}/logs/aggregation_server.log" \
    "${ROOT}/logs/aggregation_server.log.bak.$(date +%Y%m%d-%H%M%S)" || true
fi

nohup /home/pcllgr/miniconda3/envs/fedscale-server/bin/python \
  experiments/server/aggregation_server.py \
  --host 0.0.0.0 \
  --port "${AGGREGATION_PORT:-8080}" \
  --minio-endpoint http://127.0.0.1:9000 \
  --minio-access-key "${MINIO_ROOT_USER}" \
  --minio-secret-key "${MINIO_ROOT_PASSWORD}" \
  --minio-bucket "${MINIO_BUCKET}" \
  --num-clients "${NUM_CLIENTS:-2}" \
  --num-rounds "${NUM_ROUNDS:-20}" \
  --ratio "${RATIO:-0.2}" \
  --transfer-dtype "${TRANSFER_DTYPE}" \
  --results-dir "${RUN_DIR}" \
  > "${RUN_DIR}/logs/aggregation_server.log" 2>&1 &
echo $! > "${ROOT}/logs/aggregation_server.pid"
ln -sfn "${RUN_DIR}/logs/aggregation_server.log" "${ROOT}/logs/aggregation_server.log"

for i in $(seq 1 30); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${AGGREGATION_PORT:-8080}/api/round/current" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${AGGREGATION_PORT:-8080}/api/round/current"; echo
echo "aggregation_server pid=$(cat "${ROOT}/logs/aggregation_server.pid")"

# --- clients ---
"${SSH[@]}" pcl@192.168.206.116 "bash -s" <<EOF
set -e
source ~/miniconda3/bin/activate flwr-ft
cd ~/liuchao/fedscale-icc-1
mkdir -p logs
pkill -f 'run_s3r12v3_fsdp.py --client-id 0' 2>/dev/null || true
sleep 1
nohup accelerate launch --config_file configs/accelerate_config.yaml --main_process_port 29510 \\
  experiments/run_s3r12v3_fsdp.py \\
  --client-id 0 \\
  --server-url http://192.168.235.42:8080 \\
  --minio-endpoint http://192.168.235.42:9000 \\
  --minio-access-key fedscale \\
  --minio-secret-key fedscale-minio-2026 \\
  --model-path model/Qwen/Qwen2.5-0.5B \\
  --data-path data/splits/icc1_client0_train.json \\
  --transfer-dtype ${TRANSFER_DTYPE} \\
  --eval-path data/medical_flashcards_eval.json \\
  --online-eval \\
  --skip-round0-download \\
  > logs/client0_fsdp.log 2>&1 &
echo \$! > logs/client0_fsdp.pid
echo "ICC1 started pid=\$(cat logs/client0_fsdp.pid)"
EOF

"${SSH[@]}" pcl@192.168.205.130 "bash -s" <<EOF
set -e
source ~/miniconda3/bin/activate flwr-ft
cd ~/liuchao/fedscale-icc-2
mkdir -p logs
pkill -f 'run_s3r12v3_fsdp.py --client-id 1' 2>/dev/null || true
sleep 1
nohup accelerate launch --config_file configs/accelerate_config.yaml --main_process_port 29511 \\
  experiments/run_s3r12v3_fsdp.py \\
  --client-id 1 \\
  --server-url http://192.168.235.42:8080 \\
  --minio-endpoint http://192.168.235.42:9000 \\
  --minio-access-key fedscale \\
  --minio-secret-key fedscale-minio-2026 \\
  --model-path model/Qwen/Qwen2.5-0.5B \\
  --data-path data/splits/client1_train.json \\
  --transfer-dtype ${TRANSFER_DTYPE} \\
  --eval-path data/medical_flashcards_eval.json \\
  --online-eval \\
  --skip-round0-download \\
  > logs/client1_fsdp.log 2>&1 &
echo \$! > logs/client1_fsdp.pid
echo "ICC2 started pid=\$(cat logs/client1_fsdp.pid)"
EOF

# pull initial client log tails into run dir
sleep 5
scp -i "${HOME}/.ssh/id_ed25519" -o StrictHostKeyChecking=no \
  pcl@192.168.206.116:~/liuchao/fedscale-icc-1/logs/client0_fsdp.log \
  "${RUN_DIR}/logs/client0_fsdp.log" 2>/dev/null || true
scp -i "${HOME}/.ssh/id_ed25519" -o StrictHostKeyChecking=no \
  pcl@192.168.205.130:~/liuchao/fedscale-icc-2/logs/client1_fsdp.log \
  "${RUN_DIR}/logs/client1_fsdp.log" 2>/dev/null || true

echo
echo "Started run ${RUN_ID}"
echo "  results: ${RUN_DIR}"
echo "  status:  bash scripts/check_rerun_status.sh"
echo "  plot:    python scripts/plot_s3r12v3_fsdp_run.py ${RUN_DIR}"
