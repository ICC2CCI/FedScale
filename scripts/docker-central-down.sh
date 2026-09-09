#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/deployment/central-server.env"
COMPOSE_FILE="${ROOT}/deployment/docker-compose.yml"
cd "${ROOT}/deployment"
docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" down
echo "Compose services stopped"
