#!/usr/bin/env bash
# Download and validate a Hugging Face dataset in the A/B cache PVCs.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

DATASET="${1:-}"
TARGET="all"

usage() {
  cat <<'EOF'
用法：
  ./scripts/warm-dataset-cache.sh DATASET [--target a|b|all]

当前训练客户端使用 HF_DATASETS_OFFLINE=1；该脚本在实验前将数据集
下载到 A/B 的 /app/.cache/huggingface，并验证目标 split 可离线读取。
EOF
}

[[ -n "$DATASET" && "$DATASET" != -* ]] || { usage >&2; exit 2; }
shift || true
while (($#)); do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done
[[ "$TARGET" == a || "$TARGET" == b || "$TARGET" == all ]] || die "--target 只能是 a、b 或 all。"
require_cluster_configs

check_dataset_cache() {
  local kubeconfig="$1" namespace="$2" deployment="$3"
  kubectl_direct --kubeconfig "$kubeconfig" exec -n "$namespace" "deployment/$deployment" -- \
    env DATASET_TO_CHECK="$DATASET" HF_HOME=/app/.cache/huggingface \
      HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python -c '
from datasets import load_dataset
import os
name = os.environ["DATASET_TO_CHECK"]
ds = load_dataset(name, split=None, cache_dir=os.path.join(os.environ["HF_HOME"], "datasets"))
if "train_sft" not in ds:
    raise SystemExit(f"missing train_sft split; available={list(ds)}")
size = len(ds["train_sft"])
if size < 200000:
    raise SystemExit(f"train_sft is unexpectedly small: {size}")
split = ds["train_sft"]
print(f"{name}: train_sft={size} columns={split.column_names}")
' >/dev/null 2>&1
}

warm_one() {
  local kubeconfig="$1" namespace="$2" deployment="$3" role="$4"
  if check_dataset_cache "$kubeconfig" "$namespace" "$deployment"; then
    echo "✓ $role 已缓存并可离线读取 $DATASET"
    return
  fi
  echo "→ 正在预热 $role 的 $DATASET..."
  kubectl_direct --kubeconfig "$kubeconfig" exec -n "$namespace" "deployment/$deployment" -- \
    env DATASET_TO_WARM="$DATASET" HF_HOME=/app/.cache/huggingface \
      HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 \
      python -c '
from datasets import load_dataset
import os
name = os.environ["DATASET_TO_WARM"]
ds = load_dataset(name, split=None, cache_dir=os.path.join(os.environ["HF_HOME"], "datasets"))
if "train_sft" not in ds:
    raise SystemExit(f"missing train_sft split; available={list(ds)}")
size = len(ds["train_sft"])
split = ds["train_sft"]
print(f"Cached {name}: train_sft={size} columns={split.column_names}")
'
  check_dataset_cache "$kubeconfig" "$namespace" "$deployment" || die "$role 的 $DATASET 缓存校验失败。"
  echo "✓ $role 已完成 $DATASET 预热。"
}

[[ "$TARGET" == all || "$TARGET" == a ]] && warm_one "$A_KUBECONFIG" "$A_NAMESPACE" superexec-clientapp-a "客户端 A"
[[ "$TARGET" == all || "$TARGET" == b ]] && warm_one "$B_KUBECONFIG" "$B_NAMESPACE" superexec-clientapp-b "客户端 B"
