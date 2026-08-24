#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KUBECONFIG=${KUBECONFIG:-${ROOT_DIR}/config-center}
NAMESPACE=${SWANLAB_NAMESPACE:-flower-superlink}
RELEASE=${SWANLAB_RELEASE:-swanlab-self-hosted}
CHART_VERSION=${SWANLAB_CHART_VERSION:-0.6.2}

helm repo add swanlab https://helm.swanlab.cn >/dev/null 2>&1 || true
helm repo update >/dev/null

helm upgrade --install "${RELEASE}" swanlab/self-hosted \
  --version "${CHART_VERSION}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --values "${ROOT_DIR}/configs/swanlab-values.yaml" \
  "$@"
