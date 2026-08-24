#!/usr/bin/env bash
# Deploy the tested ConfigMap script and one supervisor Job. Results stay on the
# existing results PVC; only the disposable Job is replaced.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONFIG_DIR="$FLOWER_DIR/configs"
SCRIPT="$CONFIG_DIR/benchmark-matrix-supervisor.sh"
CONFIG_COMPILER="$CONFIG_DIR/benchmark_matrix_config.py"
JOB_MANIFEST="$CONFIG_DIR/benchmark-matrix-job.yaml"
APP_BUNDLE="$(mktemp /tmp/benchmark-matrix-app.XXXXXX.tar.gz)"

[[ -f "$SCRIPT" && -f "$CONFIG_COMPILER" && -f "$JOB_MANIFEST" ]] || die "Missing benchmark supervisor deployment files."
bash -n "$SCRIPT"
python3 "$CONFIG_COMPILER" validate-matrix
trap 'rm -f "$APP_BUNDLE"' EXIT

# Prepare and verify the exact target model/data before creating a Job. The
# model warm-up is skipped when a complete snapshot already exists; the data
# warm-up is performed only on A/B because the ServerApp does not load train
# partitions.
"$SCRIPT_DIR/preflight.sh" Qwen/Qwen2.5-7B
"$SCRIPT_DIR/warm-dataset-cache.sh" HuggingFaceH4/ultrachat_200k --target all

tar -C "$FLOWER_DIR" --exclude='__pycache__' -czf "$APP_BUNDLE" \
  flowertune-llm/pyproject.toml flowertune-llm/flowertune_llm

kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
  create configmap benchmark-matrix-supervisor --from-file=supervise.sh="$SCRIPT" \
  --dry-run=client -o yaml | \
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f -

kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
  create configmap benchmark-matrix-validator --from-file=benchmark_matrix_config.py="$CONFIG_COMPILER" \
  --dry-run=client -o yaml | \
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f -

kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
  create configmap benchmark-matrix-app --from-file=app.tar.gz="$APP_BUNDLE" \
  --dry-run=client -o yaml | \
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f -

kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
  delete job benchmark-matrix-supervisor --ignore-not-found --wait=true
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f "$JOB_MANIFEST"
