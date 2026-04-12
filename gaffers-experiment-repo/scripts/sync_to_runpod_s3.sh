#!/usr/bin/env bash
# Upload this repository (experiment repo only) to a RunPod network volume via S3-compatible API.
# Requires: AWS CLI v2, and RunPod S3 credentials in the environment (never commit keys).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUCKET="${RUNPOD_S3_BUCKET:-6x4jz4yym1}"
ENDPOINT="${RUNPOD_S3_ENDPOINT:-https://s3api-eu-ro-1.runpod.io}"
PREFIX="${RUNPOD_S3_PREFIX:-gaffers-experiment-repo}"

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "error: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (RunPod volume S3 credentials)." >&2
  exit 1
fi

# Many S3-compatible endpoints expect virtual-host or path style; RunPod generally works with path-style.
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

DRYRUN=()
if [[ "${DRY_RUN:-}" == "1" ]]; then
  DRYRUN=(--dryrun)
fi

DELETE_FLAG=()
if [[ "${RUNPOD_S3_SYNC_DELETE:-}" == "1" ]]; then
  DELETE_FLAG=(--delete)
fi

EXCLUDE_OUTPUT=()
if [[ "${EXCLUDE_LOCAL_OUTPUT:-}" == "1" ]]; then
  EXCLUDE_OUTPUT=(--exclude "experiment-backend/output/*")
fi

echo "sync: ${REPO_ROOT} -> s3://${BUCKET}/${PREFIX}/"
echo "endpoint: ${ENDPOINT}"

aws s3 sync "${REPO_ROOT}" "s3://${BUCKET}/${PREFIX}" \
  --endpoint-url "${ENDPOINT}" \
  "${DELETE_FLAG[@]}" \
  "${DRYRUN[@]}" \
  "${EXCLUDE_OUTPUT[@]}" \
  --exclude ".git/*" \
  --exclude "**/.venv/*" \
  --exclude "**/node_modules/*" \
  --exclude "**/__pycache__/*" \
  --exclude "**/.pytest_cache/*" \
  --exclude "**/*.pyc" \
  --exclude "**/.DS_Store" \
  --exclude "**/experiment-frontend/dist/*"

echo "done."
