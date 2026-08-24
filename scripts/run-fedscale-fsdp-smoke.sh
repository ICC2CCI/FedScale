#!/usr/bin/env bash
# Submit one real cross-centre FSDP FedScale bridge round.
#
# This verifies canonical public-block encoding and RoundPlan enforcement.  It
# intentionally does not claim SecAgg/DP protection: those are separate v1
# stages and remain disabled in this bridge smoke.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ID="${1:-fedscale-fsdp-bridge-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${2:-}"
if [[ -n "$DRY_RUN" && "$DRY_RUN" != "--dry-run" ]]; then
  echo "Usage: $0 [EXPERIMENT_ID] [--dry-run]" >&2
  exit 2
fi

extra_args=()
[[ "$DRY_RUN" == "--dry-run" ]] && extra_args+=(--dry-run)

exec "$SCRIPT_DIR/run-federated.sh" \
  --model openlm-research/open_llama_3b_v2 \
  --strategy fsdp \
  --dataset vicgalle/alpaca-gpt4 \
  --finetuning-type full \
  --quantization 0 \
  --full-local-init \
  --rounds 1 \
  --experiment-id "$EXPERIMENT_ID" \
  --set "train.full-update-compression='fedscale-int8'" \
  --set train.fedscale-block-size=1048576 \
  --set train.fedscale-mask-ratio=0.0001 \
  --set dataset.max-train-samples=300 \
  --set train.training-arguments.per-device-train-batch-size=1 \
  --set train.training-arguments.gradient-accumulation-steps=1 \
  --set train.training-arguments.max-steps=1 \
  --set train.evaluate-after-fit=false \
  --set train.num-eval-samples=1 \
  "${extra_args[@]}"
