#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"
DESKTOP_DIR="${REPO_ROOT}/desktop-app"

if [[ ! -d "${BACKEND_DIR}" || ! -d "${DESKTOP_DIR}" ]]; then
  echo "Expected backend/ and desktop-app/ to exist in repository root."
  exit 1
fi

echo "Starting backend API on http://127.0.0.1:8000 ..."
(
  cd "${BACKEND_DIR}"
  if [[ -f ".venv/bin/activate" ]]; then
    # Preferred local backend venv.
    source .venv/bin/activate
  fi
  uvicorn main:app --host 127.0.0.1 --port 8000
) &
BACKEND_PID=$!

cleanup() {
  if ps -p "${BACKEND_PID}" >/dev/null 2>&1; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting desktop Next.js + Electron shell ..."
cd "${DESKTOP_DIR}"
npm run dev:desktop
