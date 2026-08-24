#!/usr/bin/env bash
# Delete one explicitly named ephemeral train-round Job and its Pods.
# Usage: ./delete-training-job.sh {a|b} train-round-... --yes

set -euo pipefail
source "$(dirname "$0")/_common.sh"

CLUSTER="${1:-}"
JOB_NAME="${2:-}"
CONFIRM="${3:-}"
[[ "$CLUSTER" == "a" || "$CLUSTER" == "b" ]] || die "First argument must be a or b."
[[ "$JOB_NAME" =~ ^train-round-[0-9]+-[01]-[0-9]+$ ]] || die "Refusing non-training Job name: $JOB_NAME"
[[ "$CONFIRM" == "--yes" ]] || die "Deletion requires an explicit --yes."
require_cluster_configs

if [[ "$CLUSTER" == "a" ]]; then
  kubectl_direct --kubeconfig "$A_KUBECONFIG" delete job -n "$A_NAMESPACE" "$JOB_NAME"
else
  kubectl_direct --kubeconfig "$B_KUBECONFIG" delete job -n "$B_NAMESPACE" "$JOB_NAME"
fi
