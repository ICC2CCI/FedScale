#!/usr/bin/env bash
# Stop a Flower run. Requires an active local port-forward.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

RUN_ID="${1:-}"
[[ "$RUN_ID" =~ ^[0-9]+$ ]] || die "Usage: $0 RUN_ID"
require_flower_connection
flwr_cli stop "$RUN_ID" cross-cloud
