#!/usr/bin/env bash
# Download a complete Hugging Face snapshot into the cache volume used by Flower.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

MODEL="${1:-}"
TARGET="all"

usage() {
  cat <<'EOF'
用法：
  ./scripts/warm-model-cache.sh MODEL [--target center|a|b|all]

将模型完整下载到指定 Flower 端的 /app/.cache/huggingface。已通过完整性
校验的缓存会被跳过。普通 run-federated.sh 的预检会自动调用此脚本。
EOF
}

[[ -n "$MODEL" && "$MODEL" != -* && "$MODEL" != *"'"* ]] || { usage >&2; exit 2; }
shift || true
while (($#)); do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1" ;;
  esac
done
[[ "$TARGET" == center || "$TARGET" == a || "$TARGET" == b || "$TARGET" == all ]] || die "--target 只能是 center、a、b 或 all。"

MODEL_CACHE_NAME="models--${MODEL//\//--}"
require_cluster_configs

cache_complete() {
  local kubeconfig="$1" namespace="$2" deployment="$3"
  kubectl_direct --kubeconfig "$kubeconfig" exec -n "$namespace" "deployment/$deployment" -- \
    env MODEL_CACHE_ROOT="/app/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots" python -c '
from pathlib import Path
import json
import os
import sys

root = Path(os.environ["MODEL_CACHE_ROOT"])
snapshots = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
if not snapshots:
    raise SystemExit(1)
snapshot = snapshots[0]
for required in ("config.json", "tokenizer_config.json"):
    if not (snapshot / required).is_file():
        raise SystemExit(1)
index_path = snapshot / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit(1)
weight_map = json.loads(index_path.read_text())["weight_map"]
missing = [name for name in sorted(set(weight_map.values()))
           if not (snapshot / name).is_file() or (snapshot / name).stat().st_size < 1048576]
if missing:
    print("missing model shards:", missing, file=sys.stderr)
    raise SystemExit(1)
' >/dev/null 2>&1
}

warm_one() {
  local kubeconfig="$1" namespace="$2" deployment="$3" role="$4"
  if cache_complete "$kubeconfig" "$namespace" "$deployment"; then
    echo "✓ $role 已缓存 $MODEL，跳过预热。"
    return
  fi
  echo "→ 正在预热 $role 的 $MODEL（下载一次，后续 Run 复用缓存）..."
  kubectl_direct --kubeconfig "$kubeconfig" exec -n "$namespace" "deployment/$deployment" -- \
    env MODEL_TO_WARM="$MODEL" HF_HOME=/app/.cache/huggingface HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    python -c '
from huggingface_hub import snapshot_download
import os
model = os.environ["MODEL_TO_WARM"]
path = snapshot_download(repo_id=model, cache_dir=os.path.join(os.environ["HF_HOME"], "hub"))
print(f"Cached {model} at {path}")
'
  cache_complete "$kubeconfig" "$namespace" "$deployment" || die "$role 的 $MODEL 下载后完整性校验失败。"
  echo "✓ $role 已完成 $MODEL 预热。"
}

[[ "$TARGET" == all || "$TARGET" == center ]] && warm_one "$CENTER_KUBECONFIG" "$CENTER_NAMESPACE" superexec-serverapp "中心端 ServerApp"
[[ "$TARGET" == all || "$TARGET" == a ]] && warm_one "$A_KUBECONFIG" "$A_NAMESPACE" superexec-clientapp-a "客户端 A"
[[ "$TARGET" == all || "$TARGET" == b ]] && warm_one "$B_KUBECONFIG" "$B_NAMESPACE" superexec-clientapp-b "客户端 B"
