#!/usr/bin/env bash
# Shared configuration for the cross-cluster Flower operations scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOWER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$FLOWER_DIR/.." && pwd)"

CENTER_KUBECONFIG="${CENTER_KUBECONFIG:-$FLOWER_DIR/config-center}"
A_KUBECONFIG="${A_KUBECONFIG:-$FLOWER_DIR/config-tke-a}"
B_KUBECONFIG="${B_KUBECONFIG:-$FLOWER_DIR/config-tke-b}"
CENTER_NAMESPACE="${CENTER_NAMESPACE:-flower-superlink}"
A_NAMESPACE="${A_NAMESPACE:-flower-supernode-a}"
B_NAMESPACE="${B_NAMESPACE:-flower-supernode-b}"
FLOWER_HOME="${FLOWER_HOME:-/tmp/flower-flwr-home}"
FLOWER_PYTHONPATH="${FLOWER_PYTHONPATH:-/home/fusion/.local/lib/python3.13/site-packages}"

die() { echo "Error: $*" >&2; exit 1; }

require_file() { [[ -f "$1" ]] || die "Missing required file: $1"; }

kubectl_direct() {
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
    -u ALL_PROXY -u all_proxy kubectl "$@"
}

flwr_cli() {
  env HOME="$FLOWER_HOME" PYTHONPATH="$FLOWER_PYTHONPATH" flwr "$@"
}

require_cluster_configs() {
  require_file "$CENTER_KUBECONFIG"
  require_file "$A_KUBECONFIG"
  require_file "$B_KUBECONFIG"
}

require_flower_connection() {
  require_file "$FLOWER_HOME/.flwr/config.toml"
}
