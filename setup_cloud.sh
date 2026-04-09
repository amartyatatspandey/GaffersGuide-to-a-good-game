#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-gaffers-guide:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gaffers-guide-batch}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$PWD}"

echo "[1/5] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "[2/5] Installing Docker if needed..."
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

echo "[3/5] Verifying Docker service..."
sudo systemctl enable docker || true
sudo systemctl start docker || true

echo "[4/5] Pulling image: ${IMAGE_NAME}"
sudo docker pull "${IMAGE_NAME}"

echo "[5/5] Running batch container..."
sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
sudo docker run --gpus all --name "${CONTAINER_NAME}" \
  -v "${WORKSPACE_DIR}:/app" \
  -w /app \
  "${IMAGE_NAME}" \
  python backend/scripts/cloud_batch_processor.py

echo "Cloud batch run complete."
