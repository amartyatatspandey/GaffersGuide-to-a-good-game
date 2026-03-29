#!/usr/bin/env bash
# Run on RunPod (or any machine with /workspace layout): export *_coords.pkl for all training MP4s.
# Requires per-match {stem}_homographies.json under output (from calibrator or setup_and_run).
#
# Usage on pod:
#   bash run_coords_batch.sh
# Optional: FRAME_STRIDE=2 bash run_coords_batch.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export GAFFERS_HOMOGRAPHY_DIR="${GAFFERS_HOMOGRAPHY_DIR:-$WORKSPACE/output}"
export GAFFERS_COORDS_OUTPUT_DIR="${GAFFERS_COORDS_OUTPUT_DIR:-$WORKSPACE/output}"

DATA_TRAIN="${WORKSPACE}/data/training_samples"
OUT_DIR="${GAFFERS_COORDS_OUTPUT_DIR}"
LOG="${COORDS_BATCH_LOG:-$WORKSPACE/coords_batch.log}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

if [[ -f "${WORKSPACE}/backend/scripts/run_coords_only.py" ]]; then
  COORDS_SCRIPT="${WORKSPACE}/backend/scripts/run_coords_only.py"
elif [[ -f "${WORKSPACE}/scripts/run_coords_only.py" ]]; then
  COORDS_SCRIPT="${WORKSPACE}/scripts/run_coords_only.py"
else
  echo "ERROR: run_coords_only.py not found under backend/scripts or scripts"
  exit 1
fi

exec >>"$LOG" 2>&1
echo "[$(date -Iseconds)] coords batch start (FRAME_STRIDE=${FRAME_STRIDE})"
echo "Using script: ${COORDS_SCRIPT}"

if [[ ! -d "$DATA_TRAIN" ]]; then
  echo "ERROR: missing $DATA_TRAIN"
  exit 1
fi

python3 -u "$COORDS_SCRIPT" \
  --input-dir "$DATA_TRAIN" \
  --output-dir "$OUT_DIR" \
  --frame-stride "$FRAME_STRIDE"

echo "[$(date -Iseconds)] coords batch done"
echo "Logs: $LOG"
