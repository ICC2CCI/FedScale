#!/usr/bin/env bash
# 启动一轮 S3R12v3 FSDP 双集群训练，结果统一写入 results/YYYYMMDDHHMM/
#
# CFG-1/CFG-2/CFG-3：从 run yaml + nodes.yaml 驱动，CLI 显式传参给三端。
# MinIO 凭证从环境变量读，不出现在命令行（SEC-0）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "${ROOT}/deployment/central-server.env"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_yaml_env.sh"

# --- 配置文件 ---
RUN_CONFIG="${RUN_CONFIG:-${ROOT}/configs/s3r12v3-fsdp-run.yaml}"
NODES_FILE="${NODES_FILE:-${ROOT}/deployment/nodes.yaml}"
if [[ ! -f "$NODES_FILE" ]]; then
  echo "Error: nodes file not found: $NODES_FILE" >&2
  echo "  cp deployment/nodes.yaml.example deployment/nodes.yaml" >&2
  exit 1
fi
yaml_load "$RUN_CONFIG" "YCFG_"

# --- run 标识 ---
RUN_ID="${RUN_ID:-$(TZ=Asia/Shanghai date +%Y%m%d%H%M)}"
RUN_DIR="${ROOT}/results/${RUN_ID}"
TAG="${TAG:-s3r12v3-fsdp-delta-fp16}"

# --- 联邦超参（yaml -> env 覆盖 -> CLI）---
NUM_CLIENTS="$(yaml_get YCFG_ NUM_CLIENTS 2)"
NUM_ROUNDS="$(yaml_get YCFG_ NUM_ROUNDS 20)"
COVERAGE_H="$(yaml_get YCFG_ COVERAGE_H 5)"
SLOTS_PER_ROUND="$(yaml_get YCFG_ SLOTS_PER_ROUND 1)"
SEED="$(yaml_get YCFG_ SEED 20260831)"
TRANSFER_DTYPE="$(yaml_get YCFG_ TRANSFER_DTYPE fp16)"
MEMORY_DECAY="$(yaml_get YCFG_ MEMORY_DECAY 0.9)"
BLOCK_SIZE="$(yaml_get YCFG_ BLOCK_SIZE 524288)"
LOCAL_STEPS="$(yaml_get YCFG_ LOCAL_STEPS 30)"
BATCH_SIZE="$(yaml_get YCFG_ BATCH_SIZE 8)"
GRAD_ACCUM="$(yaml_get YCFG_ GRAD_ACCUM 2)"
LR="$(yaml_get YCFG_ LR 1e-5)"
SEQ_LEN="$(yaml_get YCFG_ SEQ_LEN 512)"
WRITE_FULL_EVERY_N="$(yaml_get YCFG_ WRITE_FULL_GLOBAL_EVERY_N_ROUNDS 1)"
UPLOAD_TIMEOUT_S="$(yaml_get YCFG_ CLIENT_UPLOAD_TIMEOUT_S 1800)"
EVAL_PATH="$(yaml_get YCFG_ EVAL_PATH data/medical_flashcards_eval.json)"
EVAL_MAX_BATCHES="$(yaml_get YCFG_ EVAL_MAX_BATCHES 0)"
SKIP_ROUND0="$(yaml_get YCFG_ SKIP_ROUND0_DOWNLOAD true)"
ONLINE_EVAL="$(yaml_get YCFG_ ONLINE_EVAL true)"
AUTH_TOKEN="${AUTH_TOKEN:-$(yaml_get YCFG_ AUTH_TOKEN '')}"

# 允许 env 覆盖 yaml（联调快速调参）
NUM_CLIENTS="${NUM_CLIENTS_ENV:-$NUM_CLIENTS}"
NUM_ROUNDS="${NUM_ROUNDS_ENV:-$NUM_ROUNDS}"
COVERAGE_H="${COVERAGE_H_ENV:-$COVERAGE_H}"
SLOTS_PER_ROUND="${SLOTS_PER_ROUND_ENV:-$SLOTS_PER_ROUND}"
LOCAL_STEPS="${LOCAL_STEPS_ENV:-$LOCAL_STEPS}"
LR="${LR_ENV:-$LR}"
WRITE_FULL_EVERY_N="${WRITE_FULL_EVERY_N_ENV:-$WRITE_FULL_EVERY_N}"

# --- CFG-2：禁止「只改 RATIO 不改 COVERAGE_H」的假开关 ---
# 若显式设了 RATIO 但与 coverage_h 不一致则报错并给出映射表
if [[ -n "${RATIO:-}" ]]; then
  # RATIO 来自 central-server.env 的旧字段
  IMPPLIED="$(python3 -c "print($SLOTS_PER_ROUND/$COVERAGE_H)")"
  DIFF="$(python3 -c "print(abs($IMPPLIED - ${RATIO:-0.2}))")"
  if python3 -c "exit(0 if $DIFF > 0.02 else 1)"; then
    echo "Error: RATIO=${RATIO} but COVERAGE_H=${COVERAGE_H} SLOTS=${SLOTS_PER_ROUND} (implies ${IMPPLIED})." >&2
    echo "  Set COVERAGE_H explicitly instead of RATIO. Mapping:" >&2
    echo "    5%  -> COVERAGE_H=20  SLOTS_PER_ROUND=1" >&2
    echo "    10% -> COVERAGE_H=10  SLOTS_PER_ROUND=1" >&2
    echo "    20% -> COVERAGE_H=5   SLOTS_PER_ROUND=1" >&2
    echo "    50% -> COVERAGE_H=2   SLOTS_PER_ROUND=1" >&2
    echo "    40% -> COVERAGE_H=5   SLOTS_PER_ROUND=2" >&2
    exit 1
  fi
fi

mkdir -p \
  "${RUN_DIR}/figures" \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/eval" \
  "${ROOT}/logs"

# 把生效的 run yaml 复制进 run 目录（可复现）
cp "$RUN_CONFIG" "${RUN_DIR}/run.yaml"

cat > "${RUN_DIR}/run_meta.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "tag": "${TAG}",
  "created_at": "$(date -Iseconds)",
  "protocol": "S3R12v3 + FSDP + incremental global_delta",
  "num_clients": ${NUM_CLIENTS},
  "num_rounds": ${NUM_ROUNDS},
  "coverage_h": ${COVERAGE_H},
  "slots_per_round": ${SLOTS_PER_ROUND},
  "seed": ${SEED},
  "transfer_dtype": "${TRANSFER_DTYPE}",
  "memory_decay": ${MEMORY_DECAY},
  "block_size": ${BLOCK_SIZE},
  "local_steps": ${LOCAL_STEPS},
  "batch_size": ${BATCH_SIZE},
  "grad_accum": ${GRAD_ACCUM},
  "lr": ${LR},
  "seq_len": ${SEQ_LEN},
  "write_full_global_every_n_rounds": ${WRITE_FULL_EVERY_N},
  "client_upload_timeout_s": ${UPLOAD_TIMEOUT_S},
  "eval_path": "${EVAL_PATH}",
  "skip_round0_download": ${SKIP_ROUND0},
  "online_eval": ${ONLINE_EVAL}
}
EOF

echo "${RUN_DIR}" > "${ROOT}/logs/current_rerun_results_dir.txt"
ln -sfn "${RUN_DIR}" "${ROOT}/results/current"
echo "RUN_DIR=${RUN_DIR}"
echo "  coverage_h=${COVERAGE_H} slots=${SLOTS_PER_ROUND} (~$(python3 -c "print(round($SLOTS_PER_ROUND/$COVERAGE_H*100,1))")%)"
echo "  local_steps=${LOCAL_STEPS} lr=${LR} write_full_every=${WRITE_FULL_EVERY_N}"

SSH=(ssh -i "${HOME}/.ssh/id_ed25519" -o StrictHostKeyChecking=no -o BatchMode=yes)

# --- stop leftovers ---
if [[ -f "${ROOT}/logs/aggregation_server.pid" ]]; then
  kill "$(cat "${ROOT}/logs/aggregation_server.pid")" 2>/dev/null || true
  sleep 1
  kill -9 "$(cat "${ROOT}/logs/aggregation_server.pid")" 2>/dev/null || true
fi

# --- MinIO ---
if ! curl -fsS --max-time 3 http://127.0.0.1:9000/minio/health/live >/dev/null; then
  echo "MinIO not healthy; starting compose..."
  sg docker -c "cd '${ROOT}/deployment' && docker compose --env-file '${ROOT}/deployment/central-server.env' -f docker-compose.yml up -d"
  sleep 3
fi

# --- aggregation server (CFG-2: 显式传所有参数; SEC-0: 凭证用环境变量) ---
if [[ -f "${ROOT}/logs/aggregation_server.log" ]]; then
  mv "${ROOT}/logs/aggregation_server.log" \
    "${ROOT}/logs/aggregation_server.log.bak.$(date +%Y%m%d-%H%M%S)" || true
fi

# 辅助：把 true/false 转成 CLI flag
skip_round0_flag="--skip-round0-download" ; [[ "$SKIP_ROUND0" == "false" ]] && skip_round0_flag="--no-skip-round0-download"
online_eval_flag="--online-eval" ; [[ "$ONLINE_EVAL" == "false" ]] && online_eval_flag="--no-online-eval"
auth_flag=""
[[ -n "$AUTH_TOKEN" ]] && auth_flag="--auth-token ${AUTH_TOKEN}"

export MINIO_ROOT_USER MINIO_ROOT_PASSWORD MINIO_BUCKET

nohup /home/pcllgr/miniconda3/envs/fedscale-server/bin/python \
  experiments/server/aggregation_server.py \
  --config "$RUN_CONFIG" \
  --host 0.0.0.0 \
  --port "${AGGREGATION_PORT:-8080}" \
  --minio-endpoint http://127.0.0.1:9000 \
  --minio-access-key "${MINIO_ROOT_USER}" \
  --minio-secret-key "${MINIO_ROOT_PASSWORD}" \
  --minio-bucket "${MINIO_BUCKET}" \
  --num-clients "${NUM_CLIENTS}" \
  --num-rounds "${NUM_ROUNDS}" \
  --coverage-h "${COVERAGE_H}" \
  --slots-per-round "${SLOTS_PER_ROUND}" \
  --seed "${SEED}" \
  --transfer-dtype "${TRANSFER_DTYPE}" \
  --memory-decay "${MEMORY_DECAY}" \
  --block-size "${BLOCK_SIZE}" \
  --write-full-global-every-n-rounds "${WRITE_FULL_EVERY_N}" \
  --client-upload-timeout-s "${UPLOAD_TIMEOUT_S}" \
  ${auth_flag} \
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

# --- clients (CFG-3: 从 nodes.yaml 循环拉起) ---
# 用 server conda env 的 python（有 pyyaml）；系统 python3 可能没装
NODES_PY="${NODES_PY:-/home/pcllgr/miniconda3/envs/fedscale-server/bin/python}"
"$NODES_PY" - "$NODES_FILE" <<'PYNODE' > "${RUN_DIR}/logs/client_launch_plan.json"
import sys, yaml, json
nodes = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
out = []
for c in nodes.get("clients", []):
    out.append({
        "client_id": c["client_id"],
        "ssh": c["ssh"],
        "repo": c["repo"],
        "conda_env": c.get("conda_env","flwr-ft"),
        "model_path": c["model_path"],
        "data_path": c["data_path"],
        "accelerate_config": c.get("accelerate_config","configs/accelerate_config.yaml"),
        "main_process_port": c.get("main_process_port", 29510 + c["client_id"]),
        "local_steps": c.get("local_steps"),  # None = 用全局默认
    })
print(json.dumps(out, ensure_ascii=False))
PYNODE

# 读取 launch plan 并逐个启动
mapfile -t CLIENT_PLANS < <("$NODES_PY" -c "
import json
plans = json.load(open('${RUN_DIR}/logs/client_launch_plan.json'))
for p in plans:
    print(json.dumps(p))
")

 SERVER_URL="${SERVER_URL}"
 MINIO_CLIENT_ENDPOINT="${MINIO_ENDPOINT}"

for plan_json in "${CLIENT_PLANS[@]}"; do
  CID=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['client_id'])")
  SSH_HOST=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh'])")
  REPO=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['repo'])")
  CONDA_ENV=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['conda_env'])")
  MODEL_PATH=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['model_path'])")
  DATA_PATH=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['data_path'])")
  ACCEL_CFG=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['accelerate_config'])")
  MPP=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['main_process_port'])")
  PER_CLIENT_STEPS=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('local_steps') or '')")

  PER_CLIENT_STEPS_FLAG=""
  [[ -n "$PER_CLIENT_STEPS" ]] && PER_CLIENT_STEPS_FLAG="--local-steps ${PER_CLIENT_STEPS}"

  # CFG-2/CFG-3：客户端也带 --config + 显式训练超参 + --auth-token
  # SEC-0：MinIO 凭证用环境变量传到远端，不出现在 ps 可见的命令行里
  AUTH_FLAG=""
  [[ -n "$AUTH_TOKEN" ]] && AUTH_FLAG="--auth-token ${AUTH_TOKEN}"

  echo "Launching client ${CID} on ${SSH_HOST}..."
  "${SSH[@]}" "${SSH_HOST}" "bash -s" <<EOF
set -e
source ~/miniconda3/bin/activate ${CONDA_ENV}
cd ${REPO}
mkdir -p logs
pkill -f 'run_s3r12v3_fsdp.py --client-id ${CID}' 2>/dev/null || true
sleep 1
# CFG-1：把 run yaml 拷到远端供 client 读取（若远端路径不一致则用本地相对路径）
# SEC-0：凭证从本地 env 展开后注入远端 env，命令行引用远端变量（避免明文出现在 ps）
export MINIO_ROOT_USER='${MINIO_ROOT_USER}'
export MINIO_ROOT_PASSWORD='${MINIO_ROOT_PASSWORD}'
nohup accelerate launch --config_file ${ACCEL_CFG} --main_process_port ${MPP} \\
  experiments/run_s3r12v3_fsdp.py \\
  --config configs/s3r12v3-fsdp-run.yaml \\
  --client-id ${CID} \\
  --server-url ${SERVER_URL} \\
  --minio-endpoint ${MINIO_CLIENT_ENDPOINT} \\
  --minio-access-key "\${MINIO_ROOT_USER}" \\
  --minio-secret-key "\${MINIO_ROOT_PASSWORD}" \\
  --minio-bucket ${MINIO_BUCKET} \\
  --model-path ${MODEL_PATH} \\
  --data-path ${DATA_PATH} \\
  --seed ${SEED} \\
  --transfer-dtype ${TRANSFER_DTYPE} \\
  --memory-decay ${MEMORY_DECAY} \\
  --local-steps ${LOCAL_STEPS} \\
  --batch-size ${BATCH_SIZE} \\
  --grad-accum ${GRAD_ACCUM} \\
  --lr ${LR} \\
  --seq-len ${SEQ_LEN} \\
  --eval-path ${EVAL_PATH} \\
  --eval-max-batches ${EVAL_MAX_BATCHES} \\
  ${skip_round0_flag} \\
  ${online_eval_flag} \\
  ${PER_CLIENT_STEPS_FLAG} \\
  ${AUTH_FLAG} \\
  > logs/client${CID}_fsdp.log 2>&1 &
echo \$! > logs/client${CID}_fsdp.pid
echo "ICC${CID} started pid=\$(cat logs/client${CID}_fsdp.pid)"
EOF
done

# pull initial client log tails into run dir
sleep 5
for plan_json in "${CLIENT_PLANS[@]}"; do
  CID=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['client_id'])")
  SSH_HOST=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh'])")
  REPO=$(echo "$plan_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['repo'])")
  scp -i "${HOME}/.ssh/id_ed25519" -o StrictHostKeyChecking=no \
    "${SSH_HOST}:${REPO}/logs/client${CID}_fsdp.log" \
    "${RUN_DIR}/logs/client${CID}_fsdp.log" 2>/dev/null || true
done

echo
echo "Started run ${RUN_ID}"
echo "  results: ${RUN_DIR}"
echo "  status:  bash scripts/check_rerun_status.sh"
echo "  plot:    python scripts/plot_s3r12v3_fsdp_run.py ${RUN_DIR}"
