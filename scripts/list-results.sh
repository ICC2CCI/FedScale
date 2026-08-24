#!/usr/bin/env bash
# List result directories, checkpoints, and per-round metric files.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_cluster_configs
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" exec -n "$CENTER_NAMESPACE" \
  deployment/superexec-serverapp -- bash -lc '
json_string() {
  local file="$1" key="$2"
  [ -f "$file" ] || return 0
  sed -n "s/^[[:space:]]*\"${key}\":[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$file" | head -n1
}
json_number() {
  local file="$1" key="$2"
  [ -f "$file" ] || return 0
  sed -n "s/^[[:space:]]*\"${key}\":[[:space:]]*\([0-9.]*\).*/\1/p" "$file" | head -n1
}
format_duration() {
  local seconds="${1:-}"
  [ -n "$seconds" ] || { printf "-"; return; }
  local total="${seconds%%.*}"
  [ -n "$total" ] || { printf "-"; return; }
  case "$total" in *[!0-9]*) printf "-"; return ;; esac
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local remainder=$((total % 60))
  if [ "$hours" -gt 0 ]; then
    printf "%dh%02dm%02ds" "$hours" "$minutes" "$remainder"
  else
    printf "%dm%02ds" "$minutes" "$remainder"
  fi
}
for d in /app/results/*/; do
  [ -d "$d" ] || continue
  checkpoint_count=$(find "$d" -maxdepth 1 -type d \( -name "peft_*" -o -name "full_*" \) | wc -l)
  metrics_count=$(find "$d" -maxdepth 1 -name "federated_metrics_round_*.json" | wc -l)
  state="${d}experiment_state.json"
  config_record="${d}experiment_config.json"
  [ -f "$config_record" ] || config_record="${d}experiment_manifest.json"
  summary="${d}experiment_summary.json"
  status=$(json_string "$summary" status)
  [ -n "$status" ] || status=$(json_string "$state" status)
  model=$(json_string "$config_record" model)
  strategy=$(json_string "$config_record" distributed_strategy)
  duration=$(json_number "$summary" duration_seconds)
  [ -n "$duration" ] || duration=$(json_number "$state" duration_seconds)
  printf "%s  status=%s  strategy=%s  model=%s  duration=%s  checkpoints=%s  metrics=%s\\n" \
    "$(basename "$d")" "${status:--}" "${strategy:--}" "${model:--}" \
    "$(format_duration "$duration")" "$checkpoint_count" "$metrics_count"
done'
