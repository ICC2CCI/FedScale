#!/usr/bin/env bash
# Run a resumable, GPU-safe DDP/FSDP/FedScale benchmark matrix serially.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL="openlm-research/open_llama_3b_v2"
DATASET="vicgalle/alpaca-gpt4"
ROUNDS_CSV="10,20,30,40,50"
MATRIX_ID="benchmark-matrix-$(date -u +%Y%m%dT%H%M%SZ)"
MAX_RESTARTS=3
POLL_SECONDS=120
STALL_SECONDS=12000
DRY_RUN=false

usage() {
  cat <<'EOF'
用法：
  ./scripts/run-benchmark-matrix.sh [选项]

串行运行 DDP、FSDP 和 FSDP+FedScale-INT8 的 10/20/30/40/50 轮基准矩阵。
每项失败后记录状态并继续；已完成项会从 manifest 中跳过。

选项：
  --matrix-id ID          可恢复的矩阵标识。
  --rounds CSV            轮次数，例如 10,20,30,40,50。
  --max-restarts N        单项最多自动恢复次数。默认：3。
  --poll-seconds N        轮询间隔。默认：120。
  --stall-seconds N       无 checkpoint 时的卡死阈值。默认：12000。
  --dry-run               只打印 15 项（或指定矩阵）的提交计划。
EOF
}

while (($#)); do
  case "$1" in
    --matrix-id) MATRIX_ID="${2:-}"; shift 2 ;;
    --rounds) ROUNDS_CSV="${2:-}"; shift 2 ;;
    --max-restarts) MAX_RESTARTS="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    --stall-seconds) STALL_SECONDS="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ "$MATRIX_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || die "matrix-id 格式非法。"
for number in "$MAX_RESTARTS" "$POLL_SECONDS" "$STALL_SECONDS"; do
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || die "数值参数必须为正整数。"
done
IFS=',' read -r -a ROUNDS <<< "$ROUNDS_CSV"
[[ ${#ROUNDS[@]} -gt 0 ]] || die "--rounds 不能为空。"
for rounds in "${ROUNDS[@]}"; do
  [[ "$rounds" =~ ^[1-9][0-9]*$ ]] || die "轮次数必须为正整数：$rounds"
done

MATRIX_DIR="$FLOWER_DIR/evaluation-results/$MATRIX_ID"
MANIFEST="$MATRIX_DIR/matrix-manifest.json"
mkdir -p "$MATRIX_DIR"

manifest_init() {
  python3 - "$MANIFEST" "$MATRIX_ID" "$ROUNDS_CSV" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    p.write_text(json.dumps({
        "matrix_id": sys.argv[2], "rounds": [int(x) for x in sys.argv[3].split(",")],
        "model": "openlm-research/open_llama_3b_v2", "dataset": "vicgalle/alpaca-gpt4",
        "fixed_config": {"finetuning_type": "full", "quantization": 0,
          "max_train_samples": 3334, "max_steps": 50, "num_eval_samples": 100},
        "experiments": []
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

manifest_status() {
  python3 - "$MANIFEST" "$1" "$2" "$3" "$4" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p, exp_id, strategy, rounds, status = Path(sys.argv[1]), *sys.argv[2:]
data = json.loads(p.read_text(encoding="utf-8"))
entry = next((x for x in data["experiments"] if x["experiment_id"] == exp_id), None)
if entry is None:
    entry = {"experiment_id": exp_id, "strategy": strategy, "rounds": int(rounds)}
    data["experiments"].append(entry)
entry["status"] = status
entry["updated_at"] = datetime.now(timezone.utc).isoformat()
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

manifest_is_done() {
  python3 - "$MANIFEST" "$1" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
entry = next((x for x in data["experiments"] if x["experiment_id"] == sys.argv[2]), {})
raise SystemExit(0 if entry.get("status") in {"completed", "completed_with_evaluation_failure", "failed", "stopped"} else 1)
PY
}

active_runs() {
  local output
  output="$(flwr_cli list cross-cloud --limit 100 --format json 2>/dev/null)" || return 1
  printf '%s' "$output" | python3 -c '
import json, sys
try: value=json.load(sys.stdin)
except Exception: raise SystemExit(1)
active={"pending","starting","running"}
def walk(item):
    if isinstance(item,dict): return (str(item.get("status","")).lower() in active)+sum(walk(v) for v in item.values())
    if isinstance(item,list): return sum(walk(v) for v in item)
    return 0
print(walk(value))'
}

ensure_superlink() {
  if (echo >/dev/tcp/127.0.0.1/9093) >/dev/null 2>&1; then return; fi
  "$SCRIPT_DIR/port-forward.sh" >"$MATRIX_DIR/superlink-port-forward.log" 2>&1 &
  PORT_FORWARD_PID=$!
  for _ in $(seq 1 20); do
    sleep 1
    if (echo >/dev/tcp/127.0.0.1/9093) >/dev/null 2>&1; then return; fi
  done
  die "无法建立 SuperLink 端口转发。"
}

wait_for_idle() {
  local count
  while :; do
    count="$(active_runs)" || die "无法查询 SuperLink 的实时 Run 状态；拒绝重复提交。"
    (( count == 0 )) && return
    echo "检测到已有联邦实验运行中；${POLL_SECONDS}s 后重试。"
    sleep "$POLL_SECONDS"
  done
}

manifest_init
require_cluster_configs

plans=()
for rounds in "${ROUNDS[@]}"; do
  plans+=("ddp:$rounds" "fsdp:$rounds" "fedscale:$rounds")
done

echo "矩阵：$MATRIX_ID"
printf '  %s\n' "${plans[@]}"
if [[ "$DRY_RUN" == true ]]; then exit 0; fi

ensure_superlink
trap '[[ -n "${PORT_FORWARD_PID:-}" ]] && kill "$PORT_FORWARD_PID" 2>/dev/null || true' EXIT

for plan in "${plans[@]}"; do
  strategy="${plan%%:*}"; rounds="${plan##*:}"
  experiment_id="${MATRIX_ID}-${strategy}-r${rounds}"
  if manifest_is_done "$experiment_id"; then
    echo "跳过终态实验：$experiment_id"
    continue
  fi

  wait_for_idle
  manifest_status "$experiment_id" "$strategy" "$rounds" "submitting"
  args=("$SCRIPT_DIR/run-resilient-federated.sh" --model "$MODEL" --dataset "$DATASET" --rounds "$rounds" --experiment-id "$experiment_id" --max-restarts "$MAX_RESTARTS" --poll-seconds "$POLL_SECONDS" --stall-seconds "$STALL_SECONDS" --finetuning-type full --quantization 0 --full-local-init --save-every-round 1 --set dataset.max-train-samples=3334 --set train.training-arguments.per-device-train-batch-size=1 --set train.training-arguments.gradient-accumulation-steps=1 --set train.training-arguments.max-steps=50 --set train.evaluate-after-fit=false --set train.num-eval-samples=100)
  case "$strategy" in
    ddp) args+=(--strategy ddp --set "train.full-update-compression='none'") ;;
    fsdp) args+=(--strategy fsdp --set "train.full-update-compression='none'") ;;
    fedscale) args+=(--strategy fsdp --set "train.full-update-compression='fedscale-int8'" --set train.fedscale-block-size=1048576 --set train.fedscale-mask-ratio=0.0001) ;;
  esac
  echo "=== 开始：$experiment_id ==="
  if "${args[@]}" | tee "$MATRIX_DIR/${experiment_id}.log"; then
    manifest_status "$experiment_id" "$strategy" "$rounds" "completed"
  else
    manifest_status "$experiment_id" "$strategy" "$rounds" "failed"
    echo "! ${experiment_id} 失败，继续下一个组合。" >&2
  fi
done

python3 "$SCRIPT_DIR/render-benchmark-matrix.py" "$MATRIX_DIR" --output "$MATRIX_DIR/SUMMARY.md"
echo "✓ 矩阵结束：$MATRIX_DIR/SUMMARY.md"
