#!/usr/bin/env bash
# 安装系统 Docker（需 sudo 密码一次），并设为开机自启
set -euo pipefail

if [[ "$(id -u)" -eq 0 ]]; then
  echo "请用普通用户执行：bash scripts/install-docker.sh"
  exit 1
fi

echo "==> 安装 Docker CE + Compose 插件"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> 当前用户加入 docker 组"
sudo usermod -aG docker "$USER"

echo "==> 开机自启 Docker"
sudo systemctl enable --now docker

echo "==> 验证"
sudo docker version
sudo docker compose version

cat <<EOF

安装完成。请执行下面任一方式让 docker 组生效后再部署 MinIO：
  newgrp docker
  # 或重新登录 SSH / IDE 终端

然后：
  bash scripts/docker-central-up.sh
EOF
