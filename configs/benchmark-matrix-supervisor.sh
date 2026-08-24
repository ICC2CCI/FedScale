#!/usr/bin/env bash
# Submit and monitor a serial federated fine-tuning matrix from flower-superlink.
# Live SuperLink state, rather than historical result files, controls locking.

set -euo pipefail
export HOME=/tmp/flower-home

MATRIX_ID="${MATRIX_ID:-quick-qwen25-7b-fp16-lora-fedscale-r2-20260821}"
RESULTS_ROOT=/app/results
MATRIX_DIR="$RESULTS_ROOT/$MATRIX_ID"
MANIFEST="$MATRIX_DIR/matrix-manifest.json"
CONTROL_FILE="$RESULTS_ROOT/.supervisor-control.json"
STATUS_FILE="$RESULTS_ROOT/.supervisor-status.json"
EVENTS_FILE="$RESULTS_ROOT/.supervisor-events.jsonl"
APP_DIR="${APP_DIR:-/opt/flowertune-llm}"
CONFIG_COMPILER="${CONFIG_COMPILER:-/validator/benchmark_matrix_config.py}"
POLL_SECONDS="${POLL_SECONDS:-120}"
STALL_SECONDS="${STALL_SECONDS:-12000}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
SELECTED_STRATEGY="${SUPERVISOR_STRATEGY:-fedscale}"
SUPERVISOR_ROUNDS="${SUPERVISOR_ROUNDS:-10}"
MODEL_NAME="${SUPERVISOR_MODEL:-Qwen/Qwen2.5-7B}"
DATASET_NAME="${SUPERVISOR_DATASET:-HuggingFaceH4/ultrachat_200k}"
FINETUNING_TYPE="${SUPERVISOR_FINETUNING_TYPE:-lora}"
SUPERVISOR_PLAN="${SUPERVISOR_PLAN:-}"
export SUPERVISOR_STRATEGY="$SELECTED_STRATEGY" SUPERVISOR_ROUNDS MODEL_NAME="$MODEL_NAME" \
  SUPERVISOR_MODEL="$MODEL_NAME" SUPERVISOR_DATASET="$DATASET_NAME" \
  SUPERVISOR_FINETUNING_TYPE="$FINETUNING_TYPE"
mkdir -p "$HOME/.flwr" "$MATRIX_DIR"

CURRENT_EXP=""
CURRENT_RUN=""
CONTROL_STATE="running"
CONTROL_GENERATION="0"

apply_control() {
  local values
values=$(python3 - "$CONTROL_FILE" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    payload = json.loads(p.read_text()) if p.exists() else {}
except (OSError, json.JSONDecodeError):
    payload = {}
def integer(name, fallback, lower, upper):
    try: value = int(payload.get(name, fallback))
    except (TypeError, ValueError): value = fallback
    return max(lower, min(upper, value))
print("\t".join([
    str(payload.get("desired_state", "running")),
    str(integer("poll_seconds", 120, 5, 3600)),
    str(integer("stall_seconds", 7200, 60, 86400)),
    str(integer("max_restarts", 3, 0, 10)),
    # Bash read treats consecutive tab delimiters as one separator. Emit a
    # sentinel for an empty persisted matrix_id so later fields cannot shift.
    str(payload.get("matrix_id") or "<none>"),
    str(payload.get("generation", 0)),
    str(payload.get("strategy", os.environ.get("SUPERVISOR_STRATEGY", "fedscale"))).lower(),
    str(integer("rounds", int(os.environ.get("SUPERVISOR_ROUNDS", "10")), 1, 1000)),
    str(payload.get("model", os.environ.get("SUPERVISOR_MODEL", "Qwen/Qwen2.5-7B"))),
    str(payload.get("dataset", os.environ.get("SUPERVISOR_DATASET", "HuggingFaceH4/ultrachat_200k"))),
    str(payload.get("finetuning_type", os.environ.get("SUPERVISOR_FINETUNING_TYPE", "lora"))).lower(),
]))
PY
)
  local desired poll stall restarts matrix generation strategy rounds model dataset finetuning_type
  IFS=$'\t' read -r desired poll stall restarts matrix generation strategy rounds model dataset finetuning_type <<< "$values"
  [[ "$desired" =~ ^(running|paused|stopped)$ ]] || desired=running
  [[ "$strategy" =~ ^(fsdp|fedscale|ddp)$ ]] || strategy=fedscale
  [[ "$finetuning_type" =~ ^(lora|full)$ ]] || finetuning_type=lora
  [[ "$matrix" == "<none>" ]] && matrix=""
  # In matrix mode, execution policy comes from the Job environment. The
  # control file is persistent and may still contain the previous single-run
  # values (including a long stall timeout and automatic retries). It may
  # still pause/stop the Job, but must not rewrite the matrix policy.
  if [[ -z "$SUPERVISOR_PLAN" ]]; then
    POLL_SECONDS="$poll"
    STALL_SECONDS="$stall"
    MAX_RESTARTS="$restarts"
    SELECTED_STRATEGY="$strategy"
    SUPERVISOR_ROUNDS="$rounds"
  fi
  MODEL_NAME="$model"
  DATASET_NAME="$dataset"
  FINETUNING_TYPE="$finetuning_type"
  # A matrix Job owns its matrix ID and plan. The control file is persisted on
  # the results PVC, so an old single-run selection must not redirect a fresh
  # matrix into an older result directory.
  if [[ -z "$SUPERVISOR_PLAN" && -n "$matrix" && "$matrix" != "$MATRIX_ID" ]]; then
    MATRIX_ID="$matrix"
    MATRIX_DIR="$RESULTS_ROOT/$MATRIX_ID"
    MANIFEST="$MATRIX_DIR/matrix-manifest.json"
    mkdir -p "$MATRIX_DIR"
  fi
  CONTROL_STATE="$desired"
  CONTROL_GENERATION="$generation"
}

build_plan() {
  MATRIX_PLANS=()
  if [[ -n "$SUPERVISOR_PLAN" ]]; then
    local item
    IFS=',' read -r -a raw_plan <<< "$SUPERVISOR_PLAN"
    for item in "${raw_plan[@]}"; do
      item="${item//[[:space:]]/}"
      [[ "$item" =~ ^(fsdp|fedscale|ddp):([1-9][0-9]*)$ ]] || {
        echo "invalid SUPERVISOR_PLAN item: $item" >&2
        return 2
      }
      MATRIX_PLANS+=("$item")
    done
  else
    MATRIX_PLANS+=("$SELECTED_STRATEGY:$SUPERVISOR_ROUNDS")
  fi
  ((${#MATRIX_PLANS[@]} > 0)) || {
    echo "SUPERVISOR_PLAN must contain at least one experiment" >&2
    return 2
  }
}

write_status() {
  local phase="$1" message="${2:-}" now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - "$STATUS_FILE" "$now" "$phase" "$message" "$MATRIX_ID" "$CURRENT_EXP" "$CURRENT_RUN" "$CONTROL_STATE" "$POLL_SECONDS" "$STALL_SECONDS" "$MAX_RESTARTS" "$CONTROL_GENERATION" "$SELECTED_STRATEGY" "$SUPERVISOR_ROUNDS" "$MODEL_NAME" "$DATASET_NAME" "$FINETUNING_TYPE" <<'PY'
import json, os, sys, tempfile
path, now, phase, message, matrix, experiment, run_id, desired, poll, stall, restarts, generation, strategy, rounds, model, dataset, finetuning_type = sys.argv[1:]
payload = {
    "updated_at": now, "heartbeat_at": now, "phase": phase,
    "message": message, "matrix_id": matrix, "experiment_id": experiment,
    "run_id": run_id or None, "desired_state": desired,
    "poll_seconds": int(poll), "stall_seconds": int(stall),
    "max_restarts": int(restarts), "control_generation": int(generation or 0),
    "strategy": strategy, "rounds": int(rounds), "model": model,
    "dataset": dataset, "finetuning_type": finetuning_type,
}
fd, temporary = tempfile.mkstemp(prefix=".supervisor-status.", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

record_event() {
  local event_type="$1" title="$2" message="${3:-}" now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - "$EVENTS_FILE" "$now" "$event_type" "$title" "$message" "$MATRIX_ID" "$CURRENT_EXP" "$CURRENT_RUN" "$CONTROL_STATE" <<'PY'
import json, os, sys
path, now, event_type, title, message, matrix, experiment, run_id, desired = sys.argv[1:]
event = {"timestamp": now, "type": event_type, "title": title, "message": message,
         "details": {"matrix_id": matrix, "experiment_id": experiment or None,
                      "run_id": run_id or None, "desired_state": desired}}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    handle.flush(); os.fsync(handle.fileno())
PY
}

apply_control
write_status "starting" "监督器已启动，等待调度"
record_event "supervisor_started" "监督器已启动" "等待 Flower Run 调度"

cat > "$HOME/.flwr/config.toml" <<'EOF'
[superlink.cross-cloud]
address = "superlink-service:9093"
insecure = true
EOF

build_plan
PLAN_CSV="$(IFS=,; echo "${MATRIX_PLANS[*]}")"
python3 - "$MANIFEST" "$MATRIX_ID" "$MODEL_NAME" "$DATASET_NAME" "$FINETUNING_TYPE" "$PLAN_CSV" <<'PY'
import json, sys
from pathlib import Path
p, matrix, model, dataset, finetuning_type, plan_csv = Path(sys.argv[1]), *sys.argv[2:]
try:
    data = json.loads(p.read_text()) if p.exists() else {}
except (OSError, json.JSONDecodeError):
    data = {}
data.setdefault("matrix_id", matrix)
data.setdefault("model", model)
data.setdefault("dataset", dataset)
data.setdefault("finetuning_type", finetuning_type)
data.setdefault("quantization", 0)
data.setdefault("max_train_samples", 300)
data.setdefault("max_steps", 1)
data.setdefault("num_eval_samples", 1)
data.setdefault("experiments", [])
plans = [item for item in plan_csv.split(",") if item]
data["plan"] = plans
data["rounds"] = sorted({int(item.split(":", 1)[1]) for item in plans})
for item in plans:
    strategy, rounds = item.split(":", 1)
    experiment_id = f"{matrix}-{strategy}-r{rounds}"
    if not any(entry.get("id") == experiment_id for entry in data["experiments"]):
        data["experiments"].append({
            "id": experiment_id,
            "strategy": strategy,
            "rounds": int(rounds),
            "status": "pending",
            "resume_round": 0,
        })
p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
PY

mark() {
  local exp="$1" strategy="$2" rounds="$3" status="$4" run_id="${5:-}" checkpoint="${6:-}" resume="${7:-0}" detail="${8:-}"
  CURRENT_EXP="$exp"
  CURRENT_RUN="$run_id"
  python3 - "$MANIFEST" "$exp" "$strategy" "$rounds" "$status" "$run_id" "$checkpoint" "$resume" "$detail" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
p = Path(sys.argv[1]); exp, strategy, rounds, status, run_id, checkpoint, resume, detail = sys.argv[2:]
d = json.loads(p.read_text()); e = next((x for x in d["experiments"] if x["id"] == exp), None)
if e is None:
    e = {"id": exp, "strategy": strategy, "rounds": int(rounds)}; d["experiments"].append(e)
e.update({"status": status, "resume_round": int(resume), "updated_at": datetime.now(timezone.utc).isoformat()})
if run_id: e["run_id"] = run_id
if checkpoint: e["latest_checkpoint"] = checkpoint
if detail: e["detail"] = detail[-2000:]
fd, tmp = tempfile.mkstemp(prefix=p.name, dir=p.parent)
with os.fdopen(fd, "w") as f: json.dump(d, f, indent=2); f.write("\n")
os.replace(tmp, p)
PY
  record_event "experiment_state" "实验状态：$status" "${detail:-状态已更新}"
}

state_info() {
  python3 - "$RESULTS_ROOT/$1/experiment_state.json" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
p = Path(sys.argv[1])
try: d = json.loads(p.read_text()) if p.exists() else {}
except (OSError, json.JSONDecodeError): d = {}
try: epoch = int(datetime.fromisoformat(d.get("updated_at", "").replace("Z", "+00:00")).timestamp())
except (TypeError, ValueError): epoch = 0
print("\t".join(str(d.get(k, "")) for k in ("status", "run_id", "latest_checkpoint", "latest_completed_round")) + f"\t{epoch}")
PY
}

parse_field() {
  local field="$1"
  python3 -c '
import json, re, sys
field, text = sys.argv[1], sys.stdin.read()
try: value = json.loads(text)
except json.JSONDecodeError:
    match = re.search(r"\b\d{12,}\b", text)
    print(match.group(0) if field == "run" and match else "unknown")
    raise SystemExit
def walk(x):
    if isinstance(x, dict):
        keys = ("run-id", "run_id", "runId") if field == "run" else ("status", "run_status", "run-status")
        for k in keys:
            if k in x: return str(x[k])
        for v in x.values():
            r = walk(v)
            if r: return r
    if isinstance(x, list):
        for v in x:
            r = walk(v)
            if r: return r
    return ""
print(walk(value) or "unknown")
' "$field"
}

live_status() {
  local output
  output=$(/python/venv/bin/flwr list cross-cloud --run-id "$1" --format json 2>&1) || return 2
  printf '%s' "$output" | python3 -c '
import json, sys
try: x=json.load(sys.stdin)
except Exception: raise SystemExit(2)
if isinstance(x, dict) and x.get("success") is False:
    print("not_found")
    raise SystemExit
def walk(v):
    if isinstance(v, dict):
        for key in ("status", "run_status", "run-status"):
            if key in v: return str(v[key]).lower()
        for value in v.values():
            found = walk(value)
            if found: return found
    if isinstance(v, list):
        for value in v:
            found = walk(value)
            if found: return found
    return ""
print(walk(x) or "unknown")
'
}

checkpoint_is_usable() {
  local exp="$1" strategy="$2" checkpoint="$3" path size
  [[ -n "$checkpoint" ]] || return 1
  path="$RESULTS_ROOT/$exp/$checkpoint/model_state.pt"
  [[ -f "$path" ]] || return 1
  size=$(stat -c '%s' "$path" 2>/dev/null || printf '0')
  # LoRA adapter checkpoints are small, but an empty/truncated state must
  # never become a resume source.
  (( size >= 65536 ))
}

active_run_count() {
  local output
  output=$(/python/venv/bin/flwr list cross-cloud --limit 100 --format json 2>&1) || return 2
  printf '%s' "$output" | python3 -c '
import json, sys
try: x=json.load(sys.stdin)
except Exception: raise SystemExit(2)
if isinstance(x, dict) and x.get("success") is False: raise SystemExit(2)
active={"pending","starting","running"}
def walk(v):
    if isinstance(v,dict): return (str(v.get("status","")).lower() in active)+sum(walk(i) for i in v.values())
    if isinstance(v,list): return sum(walk(i) for i in v)
    return 0
print(walk(x))'
}

manifest_terminal() {
  python3 - "$MANIFEST" "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
entry = next(
    (item for item in data.get("experiments", []) if item.get("id") == sys.argv[2]),
    {},
)
raise SystemExit(0 if entry.get("status") in {
    "completed", "completed_with_evaluation_failure", "failed", "stopped", "blocked"
} else 1)
PY
}

wait_for_idle() {
  local count
  while :; do
    apply_control
    if [[ "$CONTROL_STATE" == "paused" ]]; then
      write_status "paused" "监督器已暂停调度，等待恢复"
      sleep "$POLL_SECONDS"
      continue
    fi
    if [[ "$CONTROL_STATE" == "stopped" ]]; then
      write_status "stopped" "监督器已停止调度"
      return 1
    fi
    count=$(active_run_count) || return 2
    (( count == 0 )) && return 0
    write_status "waiting_for_idle" "已有 Flower Run 运行中，等待其结束"
    record_event "poll" "等待 Flower Run 空闲" "$count 个运行仍处于活动状态"
    echo "$count live Flower run(s) still active; waiting ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

# Verify both client supernodes are online before submitting a Run.
# ObjectStoreFedAvg requires expected_roles=("client-a","client-b"); if only
# one supernode is connected when configure_train samples the grid, the other
# never receives a train message and the Run stalls until timeout.
wait_for_nodes() {
  local required="${1:-2}" count output
  while :; do
    output=$(/python/venv/bin/flwr supernode list cross-cloud --format json 2>&1) || {
      echo "flwr supernode list failed; retrying in ${POLL_SECONDS}s" >&2
      sleep "$POLL_SECONDS"; continue
    }
    count=$(printf '%s' "$output" | python3 -c '
import json, sys
try: x = json.load(sys.stdin)
except Exception: raise SystemExit(2)
if isinstance(x, dict) and x.get("success") is False: raise SystemExit(2)
# Count only nodes whose status is "online". The SuperLink retains
# offline/unregistered historical nodes in its in-memory state; those
# must not satisfy the quorum check.
def count_online(v):
    if isinstance(v, list):
        return sum(count_online(i) for i in v)
    if isinstance(v, dict):
        if "node-id" in v and str(v.get("status", "")).lower() == "online":
            return 1
        return sum(count_online(i) for i in v.values())
    return 0
print(count_online(x))
') || {
      echo "failed to parse supernode list; retrying in ${POLL_SECONDS}s" >&2
      sleep "$POLL_SECONDS"; continue
    }
    (( count >= required )) && { echo "$count/$required supernodes online"; return 0; }
    echo "only $count/$required supernodes online; waiting ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

monitor_run() {
  local exp="$1" strategy="$2" rounds="$3" run_id="$4" checkpoint="$5" resume="$6" last_progress="$7" stall_seconds="$8"
  local status state_run state_checkpoint state_round state_epoch live now count
  while :; do
    sleep "$POLL_SECONDS"
    apply_control
    if [[ "$CONTROL_STATE" == "stopped" ]]; then
      write_status "stopping" "收到停止指令，正在停止当前 Flower Run"
      record_event "control" "收到停止指令" "请求停止 Run $run_id"
      /python/venv/bin/flwr stop "$run_id" cross-cloud || {
        mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "停止 Flower Run 失败"
        return 2
      }
      mark "$exp" "$strategy" "$rounds" stopped "$run_id" "$checkpoint" "$resume" "由控制台停止"
      return 1
    fi
    IFS=$'\t' read -r status state_run state_checkpoint state_round state_epoch <<< "$(state_info "$exp")"
    [[ -n "$state_run" ]] && run_id="$state_run"
    if [[ -n "$state_checkpoint" && "$state_checkpoint" != "$checkpoint" ]]; then checkpoint="${state_checkpoint##*/}"; resume="${checkpoint##*_}"; last_progress=$(date +%s); mark "$exp" "$strategy" "$rounds" running "$run_id" "$checkpoint" "$resume" "checkpoint persisted"; fi
    case "$status" in
      completed|completed_with_evaluation_failure) mark "$exp" "$strategy" "$rounds" "$status" "$run_id" "$checkpoint" "$resume" "ServerApp terminal state"; return 0;;
      failed|stopped) mark "$exp" "$strategy" "$rounds" "$status" "$run_id" "$checkpoint" "$resume" "ServerApp terminal state"; return 1;;
    esac
    live=$(live_status "$run_id") || { mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "cannot query live Run"; return 2; }
    case "$live" in
      pending|starting|running) ;;
          finished:*|finished|failed|stopped|completed) mark "$exp" "$strategy" "$rounds" failed "$run_id" "$checkpoint" "$resume" "live Run ended without ServerApp terminal state: $live"; return 1;;
      not_found)
        count=$(active_run_count) || { mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "live Run vanished and active-run query failed"; return 2; }
        (( count == 0 )) || { mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "live Run vanished but another Run is active"; return 2; }
        mark "$exp" "$strategy" "$rounds" failed "$run_id" "$checkpoint" "$resume" "Run no longer exists in SuperLink"; return 1;;
      *) mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "unknown live Run status: $live"; return 2;;
    esac
    if [[ "$CONTROL_STATE" == "paused" ]]; then
      write_status "paused" "监督器已暂停后续调度，但当前 Run 继续运行"
    else
      write_status "monitoring" "正在监控 Flower Run $run_id"
    fi
    record_event "heartbeat" "监督器轮询" "Run $run_id 当前状态：$live；实验状态：$status"
    now=$(date +%s); [[ "$state_epoch" =~ ^[0-9]+$ && "$state_epoch" -gt "$last_progress" ]] && last_progress="$state_epoch"
    if (( now - last_progress > stall_seconds )); then
      /python/venv/bin/flwr stop "$run_id" cross-cloud || { mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "failed to stop stalled Run"; return 2; }
      mark "$exp" "$strategy" "$rounds" stopped "$run_id" "$checkpoint" "$resume" "stalled for ${stall_seconds}s and stopped"; return 1
    fi
  done
}

run_one() {
  local strategy="$1" rounds="$2" exp="$MATRIX_ID-$strategy-r$rounds" status run_id checkpoint resume state_epoch restart=0 remaining cfg output rc stall_seconds
  CURRENT_EXP="$exp"
  SELECTED_STRATEGY="$strategy"
  SUPERVISOR_ROUNDS="$rounds"
  apply_control
  if [[ -z "$SUPERVISOR_PLAN" ]]; then
    strategy="$SELECTED_STRATEGY"
    rounds="$SUPERVISOR_ROUNDS"
  fi
  SELECTED_STRATEGY="$strategy"
  SUPERVISOR_ROUNDS="$rounds"
  exp="$MATRIX_ID-$strategy-r$rounds"
  CURRENT_EXP="$exp"
  if [[ "$CONTROL_STATE" == "paused" || "$CONTROL_STATE" == "stopped" ]]; then
    write_status "$CONTROL_STATE" "监督器未启动新的 Flower Run"
    record_event "control" "调度被暂停" "当前没有启动新的 Flower Run"
    return 2
  fi
  stall_seconds="$STALL_SECONDS"
  IFS=$'\t' read -r status run_id checkpoint resume state_epoch <<< "$(state_info "$exp")"; checkpoint="${checkpoint##*/}"; [[ "$resume" =~ ^[0-9]+$ ]] || resume=0
  if [[ -n "$checkpoint" ]] && ! checkpoint_is_usable "$exp" "$strategy" "$checkpoint"; then
    mark "$exp" "$strategy" "$rounds" failed "" "" 0 "discarded unusable checkpoint"
    status=failed; run_id=""; checkpoint=""; resume=0
  fi
  case "$status" in completed|completed_with_evaluation_failure) mark "$exp" "$strategy" "$rounds" "$status" "$run_id" "$checkpoint" "$resume" "already terminal"; return 0;; esac
  if [[ "$status" == running && "$run_id" =~ ^[0-9]+$ ]]; then
    if monitor_run "$exp" "$strategy" "$rounds" "$run_id" "$checkpoint" "$resume" "${state_epoch:-$(date +%s)}" "$stall_seconds"; then rc=0; else rc=$?; fi
    (( rc == 0 || rc == 2 )) && return "$rc"
    IFS=$'\t' read -r status run_id checkpoint resume state_epoch <<< "$(state_info "$exp")"; checkpoint="${checkpoint##*/}"; [[ "$resume" =~ ^[0-9]+$ ]] || resume=0
  fi
  while :; do
    remaining=$((rounds-resume)); (( remaining > 0 )) || { mark "$exp" "$strategy" "$rounds" failed "$run_id" "$checkpoint" "$resume" "checkpoint reached target without ServerApp completion"; return 1; }
    wait_for_idle || { mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "cannot establish idle federation"; return 2; }
    wait_for_nodes 2 || { mark "$exp" "$strategy" "$rounds" blocked "$run_id" "$checkpoint" "$resume" "both supernodes never came online"; return 2; }
    # Compile a unique, typed mapping and validate its strategy invariants
    # before invoking Flower. This prevents malformed duplicate keys from
    # entering the SuperLink queue.
    cfg=$(python3 "$CONFIG_COMPILER" compile \
      --strategy "$strategy" \
      --rounds "$remaining" \
      --experiment-id "$exp" \
      --results-root "$RESULTS_ROOT" \
      --resume-round "$resume" \
      --model "$MODEL_NAME" \
      --dataset "$DATASET_NAME" \
      --finetuning-type "$FINETUNING_TYPE") || {
        mark "$exp" "$strategy" "$rounds" blocked "" "$checkpoint" "$resume" "configuration compilation failed"
        return 2
      }
    mark "$exp" "$strategy" "$rounds" submitting "" "$checkpoint" "$resume" "submitting Flower Run"
    write_status "submitting" "正在提交 Flower Run"
    record_event "submission" "正在提交 Flower Run" "策略 $strategy，剩余 $remaining 轮"
    output=$(/python/venv/bin/flwr run "$APP_DIR" cross-cloud --run-config "$cfg" --format json 2>&1) || { mark "$exp" "$strategy" "$rounds" failed "" "$checkpoint" "$resume" "submission failed: $output"; return 1; }
    run_id=$(printf '%s' "$output" | parse_field run); [[ "$run_id" =~ ^[0-9]+$ ]] || { mark "$exp" "$strategy" "$rounds" blocked "" "$checkpoint" "$resume" "submission response lacked Run ID: $output"; return 2; }
    mark "$exp" "$strategy" "$rounds" running "$run_id" "$checkpoint" "$resume" "Flower Run submitted"
    CURRENT_RUN="$run_id"
    write_status "monitoring" "Flower Run 已提交，进入细粒度监控"
    record_event "submission" "Flower Run 已提交" "Run $run_id"
    if monitor_run "$exp" "$strategy" "$rounds" "$run_id" "$checkpoint" "$resume" "$(date +%s)" "$stall_seconds"; then rc=0; else rc=$?; fi
    (( rc == 0 || rc == 2 )) && return "$rc"
    IFS=$'\t' read -r status run_id checkpoint resume state_epoch <<< "$(state_info "$exp")"; checkpoint="${checkpoint##*/}"; [[ "$resume" =~ ^[0-9]+$ ]] || resume=0
    (( restart < MAX_RESTARTS && resume > 0 )) && [[ -n "$checkpoint" ]] || { mark "$exp" "$strategy" "$rounds" failed "$run_id" "$checkpoint" "$resume" "retry limit reached or no durable checkpoint"; return 1; }
    restart=$((restart+1)); sleep $((60*restart))
  done
}

run_matrix() {
  local plan strategy rounds exp rc failed=0
  for plan in "${MATRIX_PLANS[@]}"; do
    strategy="${plan%%:*}"
    rounds="${plan##*:}"
    exp="$MATRIX_ID-$strategy-r$rounds"
    if manifest_terminal "$exp"; then
      echo "Skipping terminal experiment: $exp"
      record_event "experiment_skip" "跳过终态实验" "$exp 已有终态记录"
      continue
    fi
    echo "=== Starting $exp ==="
    if run_one "$strategy" "$rounds"; then
      :
    else
      rc=$?
      failed=$((failed + 1))
      echo "! $exp returned $rc; recording it and continuing with the next combination" >&2
      record_event "experiment_error" "实验出错，继续下一个组合" "$exp 返回码 $rc"
      # An explicit dashboard stop is an operator request, not an experiment
      # error. Leave the remaining plan pending so it can be resumed later.
      if [[ "$CONTROL_STATE" == "stopped" ]]; then
        write_status "stopped" "监督器已停止，剩余矩阵组合保持待运行"
        return 2
      fi
    fi
  done
  if ((failed > 0)); then
    write_status "completed_with_errors" "矩阵执行完成，${failed} 个组合失败，其余组合已继续执行"
    record_event "matrix_completed" "矩阵执行完成（含失败项）" "失败组合数：$failed"
  else
    write_status "completed" "矩阵执行完成，所有组合均进入终态"
    record_event "matrix_completed" "矩阵执行完成" "所有组合均进入终态"
  fi
  return 0
}

[[ -d "$APP_DIR/flowertune_llm" && -f "$APP_DIR/pyproject.toml" ]] || {
  echo "Flower App source is missing: $APP_DIR" >&2
  exit 1
}
[[ -f "$CONFIG_COMPILER" ]] || {
  echo "Benchmark configuration compiler is missing: $CONFIG_COMPILER" >&2
  exit 1
}
python3 "$CONFIG_COMPILER" validate-matrix
apply_control
build_plan
if run_matrix; then
  exit 0
fi
echo "Supervisor stopped before completing the matrix" >&2
exit 2
