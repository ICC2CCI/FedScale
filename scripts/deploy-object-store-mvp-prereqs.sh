#!/usr/bin/env bash
# Provision only the dedicated S3 data plane and credentials. Deployment
# rollouts happen separately, after this script proves MinIO is ready.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ROOT_USER="fl-mvp-admin"
ROOT_PASSWORD="$(openssl rand -base64 36 | tr -d '=+/' | cut -c1-40)"

if kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" get secret object-store-minio-root >/dev/null 2>&1; then
  die "object-store-minio-root already exists; refuse to rotate a live MinIO root credential"
fi

create_secret() {
  local kubeconfig="$1" namespace="$2" name="$3" endpoint="$4"
  kubectl_direct --kubeconfig "$kubeconfig" -n "$namespace" create secret generic "$name" \
    --from-literal=S3_ENDPOINT="$endpoint" \
    --from-literal=S3_BUCKET=fl-models \
    --from-literal=S3_ACCESS_KEY="$ROOT_USER" \
    --from-literal=S3_SECRET_KEY="$ROOT_PASSWORD" \
    --from-literal=S3_REGION=us-east-1 \
    --from-literal=S3_ADDRESSING_STYLE=path \
    --dry-run=client -o yaml | kubectl_direct --kubeconfig "$kubeconfig" apply -f -
}

kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" create secret generic object-store-minio-root \
  --from-literal=MINIO_ROOT_USER="$ROOT_USER" --from-literal=MINIO_ROOT_PASSWORD="$ROOT_PASSWORD" \
  --dry-run=client -o yaml | kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f -

create_secret "$CENTER_KUBECONFIG" "$CENTER_NAMESPACE" object-store-server http://object-store-minio:9000
create_secret "$CENTER_KUBECONFIG" "$CENTER_NAMESPACE" object-store-aggregator http://object-store-minio:9000
create_secret "$A_KUBECONFIG" "$A_NAMESPACE" object-store-client-a http://object-store-s3:9000
create_secret "$B_KUBECONFIG" "$B_NAMESPACE" object-store-client-b http://object-store-s3:9000

kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f "$FLOWER_DIR/configs/object-store-minio.yaml"
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f "$FLOWER_DIR/configs/object-store-server-rbac.yaml"
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f "$FLOWER_DIR/configs/object-store-skupper-center.yaml"
kubectl_direct --kubeconfig "$A_KUBECONFIG" apply -f "$FLOWER_DIR/configs/object-store-skupper-a.yaml"
kubectl_direct --kubeconfig "$B_KUBECONFIG" apply -f "$FLOWER_DIR/configs/object-store-skupper-b.yaml"
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" rollout status deployment/object-store-minio --timeout=300s
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" apply -f "$FLOWER_DIR/configs/object-store-bootstrap.yaml"
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" wait --for=condition=complete job/object-store-bootstrap --timeout=300s
