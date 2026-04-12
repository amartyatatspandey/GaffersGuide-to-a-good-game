#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"
DESKTOP_DIR="${REPO_ROOT}/desktop-app"

if [[ ! -d "${BACKEND_DIR}" || ! -d "${DESKTOP_DIR}" ]]; then
  echo "Expected backend/ and desktop-app/ to exist in repository root."
  exit 1
fi

port_in_use() {
  local p="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$p" -sTCP:LISTEN -n -P >/dev/null 2>&1
    return $?
  fi
  if command -v nc >/dev/null 2>&1; then
    nc -z 127.0.0.1 "$p" >/dev/null 2>&1
    return $?
  fi
  echo "Warning: neither lsof nor nc found; cannot check if port ${p} is free." >&2
  return 1
}

BACKEND_PORT="${BACKEND_PORT:-8000}"

if port_in_use "${BACKEND_PORT}"; then
  echo "Port ${BACKEND_PORT} is already in use (FastAPI backend). Stop the other process or set BACKEND_PORT."
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${BACKEND_PORT}" -sTCP:LISTEN -n -P || true
  fi
  exit 1
fi

if [[ -n "${NEXT_PORT:-}" ]]; then
  if port_in_use "${NEXT_PORT}"; then
    echo "NEXT_PORT=${NEXT_PORT} is already in use. Free the port or choose another NEXT_PORT."
    if command -v lsof >/dev/null 2>&1; then
      lsof -iTCP:"${NEXT_PORT}" -sTCP:LISTEN -n -P || true
    fi
    exit 1
  fi
else
  export NEXT_PORT=""
  for p in {3000..3010}; do
    if ! port_in_use "$p"; then
      export NEXT_PORT="$p"
      break
    fi
  done
  if [[ -z "${NEXT_PORT}" ]]; then
    echo "No free TCP port found between 3000 and 3010 for Next.js. Set NEXT_PORT explicitly."
    exit 1
  fi
fi

echo "Desktop dev server will use NEXT_PORT=${NEXT_PORT} (Electron reads the same env)."

echo "Starting backend API on http://127.0.0.1:${BACKEND_PORT} ..."
(
  cd "${BACKEND_DIR}"
  if [[ -f ".venv/bin/activate" ]]; then
    # Preferred local backend venv.
    source .venv/bin/activate
  fi
  uvicorn main:app --host 127.0.0.1 --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

cleanup() {
  if ps -p "${BACKEND_PID}" >/dev/null 2>&1; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# Always export NEXT_PUBLIC_BACKEND_URL so Electron preload picks it up,
# even when the backend runs on the default port 8000.
export NEXT_PUBLIC_BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
echo "Backend URL exported: NEXT_PUBLIC_BACKEND_URL=${NEXT_PUBLIC_BACKEND_URL}"

# Wait for backend to accept connections (max 30s) before starting the UI.
echo "Waiting for backend on port ${BACKEND_PORT} ..."
BACKEND_WAIT=0
until port_in_use "${BACKEND_PORT}" || [[ ${BACKEND_WAIT} -ge 30 ]]; do
  sleep 1
  BACKEND_WAIT=$((BACKEND_WAIT + 1))
done
if ! port_in_use "${BACKEND_PORT}"; then
  echo "Backend did not start within 30s. Check for errors above."
  exit 1
fi
echo "Backend is ready on port ${BACKEND_PORT}."

echo "Starting desktop Next.js + Electron shell ..."
cd "${DESKTOP_DIR}"
export NEXT_PORT
npm run dev:desktop
