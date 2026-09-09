#!/usr/bin/env bash
# 将 MinIO 切到 Docker Compose（无自动重启）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/deployment/central-server.env"
COMPOSE_FILE="${ROOT}/deployment/docker-compose.yml"
# shellcheck disable=SC1090
source "${ENV_FILE}"

run_compose() {
  if docker info >/dev/null 2>&1; then
    docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" "$@"
  elif sg docker -c "docker info" >/dev/null 2>&1; then
    sg docker -c "docker compose --env-file $(printf '%q' "${ENV_FILE}") -f $(printf '%q' "${COMPOSE_FILE}") $(printf '%q ' "$@")"
  else
    echo "ERROR: 当前用户无法访问 Docker daemon。执行: newgrp docker"
    exit 1
  fi
}

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker 未安装。请先运行: bash scripts/install-docker.sh"
  exit 1
fi

echo "==> 停止旧的二进制 / systemd MinIO"
systemctl --user disable --now fedscale-minio.service 2>/dev/null || true
if [[ -x "${ROOT}/scripts/minio.sh" ]]; then
  bash "${ROOT}/scripts/minio.sh" stop || true
fi
pkill -f "${MINIO_BIN} server" 2>/dev/null || true
sleep 1

mkdir -p "${MINIO_DATA_DIR}" "${LOG_DIR}"

echo "==> 拉取并启动 MinIO 容器（restart: no）"
cd "${ROOT}/deployment"
run_compose pull
run_compose up -d

echo "==> 等待就绪"
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${MINIO_API_PORT}/minio/health/live" >/dev/null 2>&1; then
    echo "MinIO healthy"
    break
  fi
  sleep 1
  if [[ "$i" -eq 60 ]]; then
    echo "ERROR: MinIO 未在预期时间内就绪"
    run_compose ps
    run_compose logs --tail=50
    exit 1
  fi
done

echo "==> 确保 bucket 存在"
if [[ -x "${MC_BIN}" ]]; then
  "${MC_BIN}" alias set fedscale "http://127.0.0.1:${MINIO_API_PORT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null
  "${MC_BIN}" mb -p "fedscale/${MINIO_BUCKET}" >/dev/null || true
  "${MC_BIN}" ls "fedscale/${MINIO_BUCKET}" || true
fi

echo
run_compose ps
echo
echo "API:     http://${CENTRAL_SERVER_IP}:${MINIO_API_PORT}"
echo "Console: http://${CENTRAL_SERVER_IP}:${MINIO_CONSOLE_PORT}"
