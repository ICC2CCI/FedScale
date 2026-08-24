#!/usr/bin/env bash
# 在提交联邦实验前验证三端就绪状态，并自动预热缺失模型缓存。

set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL="${1:-}"
[[ -n "$MODEL" ]] || die "用法：$0 MODEL [--no-warm]"
shift || true
AUTO_WARM=true
while (($#)); do
  case "$1" in
    --no-warm) AUTO_WARM=false; shift ;;
    *) die "未知参数：$1" ;;
  esac
done

MODEL_CACHE_NAME="models--${MODEL//\//--}"
require_cluster_configs

check_deployment() {
  local kubeconfig="$1" namespace="$2" deployment="$3"
  local ready
  ready="$(kubectl_direct --kubeconfig "$kubeconfig" get deployment "$deployment" -n "$namespace" -o jsonpath='{.status.readyReplicas}')"
  [[ "$ready" == "1" ]] || die "$namespace/$deployment 未就绪（readyReplicas=${ready:-0}）。"
}

check_model_cache() {
  local kubeconfig="$1" namespace="$2" deployment="$3"
  kubectl_direct --kubeconfig "$kubeconfig" exec -n "$namespace" "deployment/$deployment" -- \
    sh -ceu "snapshot_root=/app/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots; [ -d \"\$snapshot_root\" ] || exit 1; snapshot=\$(find \"\$snapshot_root\" -mindepth 1 -maxdepth 1 -type d | head -n 1); test -n \"\$snapshot\"; test -r \"\$snapshot/config.json\"; test -r \"\$snapshot/tokenizer_config.json\"; find -L \"\$snapshot\" -maxdepth 1 -type f \( -name '*.bin' -o -name '*.safetensors' \) -size +1M | grep -q ." >/dev/null 2>&1
}

ensure_model_cache() {
  local target="$1" kubeconfig="$2" namespace="$3" deployment="$4" role="$5"
  if check_model_cache "$kubeconfig" "$namespace" "$deployment"; then
    echo "✓ $role 已缓存 $MODEL"
    return
  fi
  [[ "$AUTO_WARM" == true ]] || die "$role 未找到 $MODEL 的完整本地 snapshot；可执行 ./scripts/warm-model-cache.sh $MODEL --target $target。"
  echo "! $role 缺少 $MODEL 缓存，自动预热。"
  "$(dirname "$0")/warm-model-cache.sh" "$MODEL" --target "$target"
  check_model_cache "$kubeconfig" "$namespace" "$deployment" || die "$role 自动预热后仍未找到完整 $MODEL snapshot。"
  echo "✓ $role 已缓存 $MODEL"
}

model_snapshot_id() {
  local kubeconfig="$1" namespace="$2" deployment="$3"
  kubectl_direct --kubeconfig "$kubeconfig" exec -n "$namespace" "deployment/$deployment" -- \
    sh -ceu "snapshot_root=/app/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots; snapshot=\$(find \"\$snapshot_root\" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1); basename \"\$snapshot\"" 2>/dev/null
}

echo "联邦实验预检：$MODEL"
check_deployment "$CENTER_KUBECONFIG" "$CENTER_NAMESPACE" superexec-serverapp
check_deployment "$A_KUBECONFIG" "$A_NAMESPACE" superexec-clientapp-a
check_deployment "$B_KUBECONFIG" "$B_NAMESPACE" superexec-clientapp-b

# ServerApp 需要初始化全局参数；仅 A/B 有缓存仍会导致 Run 秒级失败。
ensure_model_cache center "$CENTER_KUBECONFIG" "$CENTER_NAMESPACE" superexec-serverapp "中心端 ServerApp"
ensure_model_cache a "$A_KUBECONFIG" "$A_NAMESPACE" superexec-clientapp-a "客户端 A"
ensure_model_cache b "$B_KUBECONFIG" "$B_NAMESPACE" superexec-clientapp-b "客户端 B"

center_snapshot="$(model_snapshot_id "$CENTER_KUBECONFIG" "$CENTER_NAMESPACE" superexec-serverapp)"
a_snapshot="$(model_snapshot_id "$A_KUBECONFIG" "$A_NAMESPACE" superexec-clientapp-a)"
b_snapshot="$(model_snapshot_id "$B_KUBECONFIG" "$B_NAMESPACE" superexec-clientapp-b)"
[[ -n "$center_snapshot" && "$center_snapshot" == "$a_snapshot" && "$center_snapshot" == "$b_snapshot" ]] || die "三端模型 snapshot 不一致：center=${center_snapshot:-missing}, A=${a_snapshot:-missing}, B=${b_snapshot:-missing}。禁止使用本地全模型初始化。"
echo "✓ 三端模型 snapshot 一致：$center_snapshot"

echo '✓ 预检通过：可提交实验。'
