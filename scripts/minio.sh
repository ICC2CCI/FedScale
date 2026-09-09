#!/usr/bin/env bash
# 启动 / 停止 Central Server 上的 MinIO
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/deployment/central-server.env"
# shellcheck disable=SC1090
source "${ENV_FILE}"

PID_FILE="${LOG_DIR}/minio.pid"
LOG_FILE="${LOG_DIR}/minio.log"

mkdir -p "${MINIO_DATA_DIR}" "${LOG_DIR}"

start() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "MinIO already running (pid=$(cat "${PID_FILE}"))"
    return 0
  fi
  if [[ ! -x "${MINIO_BIN}" ]]; then
    echo "ERROR: MinIO binary missing: ${MINIO_BIN}"
    exit 1
  fi
  export MINIO_ROOT_USER MINIO_ROOT_PASSWORD
  nohup "${MINIO_BIN}" server "${MINIO_DATA_DIR}" \
    --address ":${MINIO_API_PORT}" \
    --console-address ":${MINIO_CONSOLE_PORT}" \
    >"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "MinIO started pid=$(cat "${PID_FILE}")"
  echo "  API:     http://${CENTRAL_SERVER_IP}:${MINIO_API_PORT}"
  echo "  Console: http://${CENTRAL_SERVER_IP}:${MINIO_CONSOLE_PORT}"
  echo "  user/password: ${MINIO_ROOT_USER} / ${MINIO_ROOT_PASSWORD}"
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
      echo "MinIO stopped (pid=${pid})"
    else
      echo "MinIO pid file stale, removing"
    fi
    rm -f "${PID_FILE}"
  else
    echo "MinIO not running"
  fi
}

status() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "MinIO running pid=$(cat "${PID_FILE}")"
    ss -lnt | grep -E ":${MINIO_API_PORT}|:${MINIO_CONSOLE_PORT}" || true
  else
    echo "MinIO not running"
    exit 1
  fi
}

init_bucket() {
  local mc="${MC_BIN}"
  if [[ ! -x "${mc}" ]]; then
    echo "ERROR: mc binary missing: ${mc}"
    exit 1
  fi
  # wait for API
  for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${MINIO_API_PORT}/minio/health/live" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  "${mc}" alias set fedscale "http://127.0.0.1:${MINIO_API_PORT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null
  if "${mc}" ls "fedscale/${MINIO_BUCKET}" >/dev/null 2>&1; then
    echo "Bucket already exists: ${MINIO_BUCKET}"
  else
    "${mc}" mb "fedscale/${MINIO_BUCKET}"
    echo "Created bucket: ${MINIO_BUCKET}"
  fi
  # 预创建逻辑前缀占位（可选）
  echo "ok" | "${mc}" pipe "fedscale/${MINIO_BUCKET}/.keep" >/dev/null || true
  "${mc}" ls "fedscale/${MINIO_BUCKET}"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  init-bucket) init_bucket ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|init-bucket}"
    exit 1
    ;;
esac
