#!/usr/bin/env bash
# Keep a local Flower control connection open. Stop with Ctrl-C.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

[[ $# -eq 0 ]] || die "Usage: $0"
require_cluster_configs
echo "Forwarding localhost:9093 to SuperLink control API. Press Ctrl-C to stop."
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
  port-forward svc/superlink-service 9093:9093
