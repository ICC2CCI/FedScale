#!/usr/bin/env bash
# Export small experiment metadata from the center PVC and render REPORT.md.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

usage() {
  cat <<'EOF'
用法：
  ./scripts/export-experiment-report.sh EXPERIMENT_ID [OUTPUT_DIR]

从中心结果 PVC 导出实验 JSON 元数据并生成中文 REPORT.md。默认输出到
evaluation-results/EXPERIMENT_ID；不会复制 peft_N 模型权重。
EOF
}

[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
EXPERIMENT_ID="$1"
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || \
  die "EXPERIMENT_ID 格式非法。"
OUTPUT_DIR="${2:-$FLOWER_DIR/evaluation-results/$EXPERIMENT_ID}"

require_cluster_configs
mkdir -p "$OUTPUT_DIR"

pod_name="$(kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" \
  -n "$CENTER_NAMESPACE" get pod -l app=superexec-serverapp \
  -o jsonpath='{.items[0].metadata.name}')"
[[ -n "$pod_name" ]] || die "未找到中心端 SuperExec Pod。"

remote_dir="/app/results/$EXPERIMENT_ID"
kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
  exec "$pod_name" -- test -d "$remote_dir" || \
  die "中心结果目录不存在：$remote_dir"

mapfile -t metadata_files < <(
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
    exec "$pod_name" -- find "$remote_dir" -maxdepth 1 -type f -name '*.json' \
    -printf '%f\n' | sort
)
[[ ${#metadata_files[@]} -gt 0 ]] || die "结果目录中没有 JSON 元数据。"

for filename in "${metadata_files[@]}"; do
  kubectl_direct --kubeconfig "$CENTER_KUBECONFIG" -n "$CENTER_NAMESPACE" \
    cp "$pod_name:$remote_dir/$filename" "$OUTPUT_DIR/$filename"
done

python3 "$SCRIPT_DIR/render-experiment-report.py" "$OUTPUT_DIR"
echo "✓ 已导出实验报告：$OUTPUT_DIR/REPORT.md"
