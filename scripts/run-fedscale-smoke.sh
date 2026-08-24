#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-fedscale-smoke-result.json}"

exec python3 "$SCRIPT_DIR/fedscale_smoke.py" \
  --rounds 2 \
  --clients 2 \
  --world-size 2 \
  --block-size 7 \
  --window-size 10 \
  --mask-ratio 0.5 \
  --clip-norm 0.5 \
  --noise-sigma 0.01 \
  --quant-scale 0.01 \
  --output "$OUTPUT"
