#!/usr/bin/env bash
# 在 Docker 尚未安装前，用 systemd --user 保活二进制 MinIO（防进程掉线）
# Docker 就绪后请改用: bash scripts/docker-central-up.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/deployment/central-server.env"
# shellcheck disable=SC1090
source "${ENV_FILE}"

UNIT_DIR="${HOME}/.config/systemd/user"
UNIT_FILE="${UNIT_DIR}/fedscale-minio.service"
mkdir -p "${UNIT_DIR}" "${MINIO_DATA_DIR}" "${LOG_DIR}"

# 停掉 nohup 方式，避免双开
bash "${ROOT}/scripts/minio.sh" stop || true

cat >"${UNIT_FILE}" <<EOF
[Unit]
Description=FedScale MinIO (binary, interim until Docker)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=MINIO_ROOT_USER=${MINIO_ROOT_USER}
Environment=MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}
ExecStart=${MINIO_BIN} server ${MINIO_DATA_DIR} --address :${MINIO_API_PORT} --console-address :${MINIO_CONSOLE_PORT}
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now fedscale-minio.service
# 允许用户未登录时也跑 user 服务（需一次可能要密码的 loginctl）
if command -v loginctl >/dev/null 2>&1; then
  loginctl enable-linger "$(id -un)" 2>/dev/null || \
    echo "提示: 若希望注销后仍保活，请执行: sudo loginctl enable-linger $(id -un)"
fi

systemctl --user status fedscale-minio.service --no-pager || true
curl -fsS "http://127.0.0.1:${MINIO_API_PORT}/minio/health/live" && echo " health ok"
