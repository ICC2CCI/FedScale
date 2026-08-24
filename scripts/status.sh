#!/usr/bin/env bash
# Print the center, A, and B cluster resources relevant to federated training.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_cluster_configs
echo '=== Center ==='
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" get deployments,pods,svc -n "$CENTER_NAMESPACE"
echo '=== Client cluster A ==='
kubectl_direct --kubeconfig "$A_KUBECONFIG" get jobs,pods -n "$A_NAMESPACE" -o wide
echo '=== Client cluster B ==='
kubectl_direct --kubeconfig "$B_KUBECONFIG" get jobs,pods -n "$B_NAMESPACE" -o wide
