#!/usr/bin/env bash
# List Flower runs. Requires an active local port-forward.

set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_flower_connection
flwr_cli list cross-cloud
