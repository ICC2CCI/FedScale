#!/usr/bin/env bash
# Submit a federated run and resume it from the last durable global checkpoint.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL="openlm-research/open_llama_3b_v2"
STRATEGY="fsdp"
DATASET="vicgalle/alpaca-gpt4"
TOTAL_ROUNDS="50"
EXPERIMENT_ID="resilient-$(date -u +%Y%m%dT%H%M%SZ)"
MAX_RESTARTS="3"
POLL_SECONDS="120"
STALL_SECONDS="12000"
EXTRA_CONFIG=()
FINETUNING_TYPE=""
QUANTIZATION=""
FULL_LOCAL_INITIALIZATION=false
SAVE_EVERY_ROUND="1"

usage() {
  cat <<'EOF'
用法：
  ./scripts/run-resilient-federated.sh [选项]

提交实验；失败时从中心端结果 PVC 内最近的成功 LoRA 检查点恢复，最多重启指定次数。

选项：
  --model MODEL
  --strategy fsdp|ddp
  --dataset DATASET
  --rounds N                 目标全局总轮数。默认：50
  --experiment-id ID         结果目录标识。默认：带 UTC 时间戳。
  --max-restarts N           最多自动恢复次数。默认：3
  --poll-seconds N           状态轮询间隔。默认：120
  --stall-seconds N          状态长期无更新时判定为卡死。默认：12000
  --finetuning-type T        lora 或 full。
  --quantization N           0、4 或 8；full 必须为 0。
  --full-local-init          全参数首轮使用三端一致本地基座。
  --save-every-round N       每 N 个成功轮次保存 checkpoint。默认：1。
  --set KEY=VALUE            追加 Flower run-config；可重复。
EOF
}

while (($#)); do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --strategy) STRATEGY="${2:-}"; shift 2 ;;
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --rounds) TOTAL_ROUNDS="${2:-}"; shift 2 ;;
    --experiment-id) EXPERIMENT_ID="${2:-}"; shift 2 ;;
    --max-restarts) MAX_RESTARTS="${2:-}"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
    --stall-seconds) STALL_SECONDS="${2:-}"; shift 2 ;;
    --finetuning-type) FINETUNING_TYPE="${2:-}"; shift 2 ;;
    --quantization) QUANTIZATION="${2:-}"; shift 2 ;;
    --full-local-init) FULL_LOCAL_INITIALIZATION=true; shift ;;
    --save-every-round) SAVE_EVERY_ROUND="${2:-}"; shift 2 ;;
    --set) EXTRA_CONFIG+=("${2:-}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done

[[ "$STRATEGY" == fsdp || "$STRATEGY" == ddp ]] || die "--strategy 只能是 fsdp 或 ddp。"
[[ -z "$FINETUNING_TYPE" || "$FINETUNING_TYPE" == lora || "$FINETUNING_TYPE" == full ]] || die "--finetuning-type 只能是 lora 或 full。"
[[ -z "$QUANTIZATION" || "$QUANTIZATION" == 0 || "$QUANTIZATION" == 4 || "$QUANTIZATION" == 8 ]] || die "--quantization 只能是 0、4 或 8。"
[[ "$FINETUNING_TYPE" != full || -z "$QUANTIZATION" || "$QUANTIZATION" == 0 ]] || die "full 微调必须使用 --quantization 0。"
[[ "$FULL_LOCAL_INITIALIZATION" != true || "$FINETUNING_TYPE" == full ]] || die "--full-local-init 必须与 --finetuning-type full 一起使用。"
for number in "$TOTAL_ROUNDS" "$MAX_RESTARTS" "$POLL_SECONDS" "$STALL_SECONDS" "$SAVE_EVERY_ROUND"; do
  [[ "$number" =~ ^[1-9][0-9]*$ ]] || die "数值参数必须为正整数。"
done
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || die "experiment-id 格式非法。"
require_flower_connection

lock_dir="/tmp/flower-resilient-${EXPERIMENT_ID}.lock"
mkdir "$lock_dir" 2>/dev/null || die "实验 ${EXPERIMENT_ID} 已有一个监督脚本在运行。"
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

state_file="/app/results/${EXPERIMENT_ID}/experiment_state.json"
latest_checkpoint() {
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" exec -n "$CENTER_NAMESPACE" deployment/superexec-serverapp -- \
    /usr/bin/bash -lc "find /app/results/${EXPERIMENT_ID} -maxdepth 1 -mindepth 1 -type d \( -name 'peft_[0-9]*' -o -name 'full_[0-9]*' \) -printf '%f\\n' 2>/dev/null | sort -t_ -k2,2n | tail -n1" 2>/dev/null || true
}
state_value() {
  local key="$1"
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" exec -n "$CENTER_NAMESPACE" deployment/superexec-serverapp -- \
    /usr/bin/bash -lc "test -f '${state_file}' && sed -n 's/^[[:space:]]*\"${key}\": \"\\?\([^\",]*\).*/\1/p' '${state_file}' | head -n1" 2>/dev/null || true
}

submit() {
  local remaining="$1" resume_round="$2" checkpoint="$3"
  local cmd=("$(dirname "$0")/run-federated.sh" --model "$MODEL" --strategy "$STRATEGY" --dataset "$DATASET" --rounds "$remaining" --experiment-id "$EXPERIMENT_ID" --save-every-round "$SAVE_EVERY_ROUND")
  [[ -z "$FINETUNING_TYPE" ]] || cmd+=(--finetuning-type "$FINETUNING_TYPE")
  [[ -z "$QUANTIZATION" ]] || cmd+=(--quantization "$QUANTIZATION")
  [[ "$FULL_LOCAL_INITIALIZATION" != true || "$resume_round" != 0 ]] || cmd+=(--full-local-init)
  if [[ -n "$checkpoint" ]]; then
    cmd+=(--resume-peft-path "/app/results/${EXPERIMENT_ID}/${checkpoint}" --resume-round "$resume_round")
  fi
  for item in "${EXTRA_CONFIG[@]}"; do cmd+=(--set "$item"); done
  local submit_output
  submit_output="$("${cmd[@]}" 2>&1)" || {
    printf '%s\n' "$submit_output" >&2
    return 1
  }
  printf '%s\n' "$submit_output"
  current_run_id="$(printf '%s\n' "$submit_output" | grep -Eo '[0-9]{16,}' | tail -n1 || true)"
  [[ -n "$current_run_id" ]] || die "已提交 Run 但无法解析 Run ID；为避免重复提交，停止自动恢复。"
  echo "Flower Run ID: ${current_run_id}"
}

attempt=0
resume_round=0
checkpoint=""
current_run_id=""

export_report() {
  local exporter="$(dirname "$0")/export-experiment-report.sh"
  if ! "$exporter" "$EXPERIMENT_ID"; then
    echo "Warning: 实验已完成，但本地报告导出失败；中心 PVC 记录仍然有效。" >&2
  fi
}

while :; do
  remaining=$((TOTAL_ROUNDS - resume_round))
  (( remaining > 0 )) || {
    echo "✓ ${EXPERIMENT_ID} 已达到目标 ${TOTAL_ROUNDS} 轮。"
    export_report
    exit 0
  }
  echo "=== 提交尝试 ${attempt}/${MAX_RESTARTS}：从第 ${resume_round} 轮恢复，剩余 ${remaining} 轮 ==="
  submit "$remaining" "$resume_round" "$checkpoint"

  last_change=$(date +%s)
  while :; do
    sleep "$POLL_SECONDS"
    status=$(state_value status)
    current_checkpoint=$(latest_checkpoint)
    if [[ -n "$current_checkpoint" && "$current_checkpoint" != "$checkpoint" ]]; then
      checkpoint="$current_checkpoint"
      resume_round="${checkpoint##*_}"
      last_change=$(date +%s)
      echo "✓ 已持久化全局检查点：${checkpoint}"
    fi
    case "$status" in
      completed|completed_with_evaluation_failure)
        echo "✓ 实验 ${EXPERIMENT_ID} 已完成。"
        export_report
        exit 0
        ;;
      failed)
        echo "! ServerApp 已标记本次尝试失败。"
        break
        ;;
    esac
    now=$(date +%s)
    if (( now - last_change > STALL_SECONDS )); then
      echo "! 超过 ${STALL_SECONDS}s 未生成新的成功检查点，判定为卡死。"
      [[ -n "$current_run_id" ]] || die "无法安全停止卡死 Run：缺少 Run ID。"
      echo "停止卡死 Run ${current_run_id}，随后恢复。"
      flwr_cli stop "$current_run_id" cross-cloud || die "停止卡死 Run 失败；拒绝提交重复实验。"
      break
    fi
  done

  (( attempt < MAX_RESTARTS )) || die "已达到最大自动恢复次数 ${MAX_RESTARTS}；保留最后检查点 ${checkpoint:-无}。"
  [[ -n "$checkpoint" ]] || die "失败前没有成功全局检查点，拒绝从不确定状态自动重启。"
  attempt=$((attempt + 1))
  backoff=$((60 * attempt))
  echo "将在 ${backoff}s 后从 ${checkpoint} 恢复。"
  sleep "$backoff"
done
