#!/usr/bin/env bash
# Usage: ./logs.sh {server|a|b|all} [--follow]

set -euo pipefail
source "$(dirname "$0")/_common.sh"

TARGET="${1:-all}"
FOLLOW="${2:-}"
[[ "$FOLLOW" == "" || "$FOLLOW" == "--follow" ]] || die "Usage: $0 {server|a|b|all} [--follow]"
require_cluster_configs

log_args=(--tail=160)
[[ "$FOLLOW" == "--follow" ]] && log_args=(-f)

show_server() { kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" logs -n "$CENTER_NAMESPACE" deployment/superexec-serverapp "${log_args[@]}"; }
show_a() { kubectl_direct --kubeconfig "$A_KUBECONFIG" logs -n "$A_NAMESPACE" deployment/superexec-clientapp-a "${log_args[@]}"; }
show_b() { kubectl_direct --kubeconfig "$B_KUBECONFIG" logs -n "$B_NAMESPACE" deployment/superexec-clientapp-b "${log_args[@]}"; }

case "$TARGET" in
  server) show_server ;;
  a) show_a ;;
  b) show_b ;;
  all)
    [[ "$FOLLOW" != "--follow" ]] || die "Use server, a, or b with --follow."
    echo '=== ServerApp ==='; show_server
    echo '=== Client A ==='; show_a
    echo '=== Client B ==='; show_b
    ;;
  *) die "Usage: $0 {server|a|b|all} [--follow]" ;;
esac
