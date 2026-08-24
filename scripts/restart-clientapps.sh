#!/usr/bin/env bash
# Restart both ClientApp Deployments, for example after a stale deleted-Job poll.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

[[ "${1:-}" == "--yes" ]] || die "Restart requires an explicit --yes."
require_cluster_configs

kubectl_direct --kubeconfig "$A_KUBECONFIG" rollout restart deployment/superexec-clientapp-a -n "$A_NAMESPACE"
kubectl_direct --kubeconfig "$B_KUBECONFIG" rollout restart deployment/superexec-clientapp-b -n "$B_NAMESPACE"
kubectl_direct --kubeconfig "$A_KUBECONFIG" rollout status deployment/superexec-clientapp-a -n "$A_NAMESPACE" --timeout=120s
kubectl_direct --kubeconfig "$B_KUBECONFIG" rollout status deployment/superexec-clientapp-b -n "$B_NAMESPACE" --timeout=120s
