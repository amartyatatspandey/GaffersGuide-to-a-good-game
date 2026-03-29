#!/usr/bin/env bash
# fix_environment.sh — RunPod /workspace "god-mode" prep for Gaffer's Guide GPU batch.
# Run from anywhere: bash fix_environment.sh   (expects /workspace layout on the pod)

set -u

WORKSPACE="${WORKSPACE:-/workspace}"
BACKUP="${WORKSPACE}/data/safe_code_backup"
DATA_TRAIN="${WORKSPACE}/data/training_samples"
MODEL_DIR="${WORKSPACE}/models/pretrained"
OUT_DIR="${WORKSPACE}/output"
LOG_TAG="[fix_environment]"

log() { echo "${LOG_TAG} $*"; }

die() { log "ERROR: $*"; exit 1; }

[[ -d "${WORKSPACE}" ]] || die "WORKSPACE not found: ${WORKSPACE}"

log "Using WORKSPACE=${WORKSPACE}"

# --- 1. Dependencies (force (re)install) ---
log "Installing Python packages..."
python3 -m pip install --upgrade pip setuptools wheel 2>/dev/null || true
python3 -m pip install --no-cache-dir --upgrade \
  opencv-python-headless \
  supervision \
  ultralytics \
  filterpy \
  google-genai \
  || die "pip install failed"

# --- 2. Directory layout ---
log "Creating directories..."
mkdir -p "${DATA_TRAIN}" "${MODEL_DIR}" "${OUT_DIR}"

# --- 3. Consolidate videos: all .mp4 under backup → training_samples ---
if [[ -d "${BACKUP}" ]]; then
  log "Moving .mp4 files from ${BACKUP} → ${DATA_TRAIN}"
  while IFS= read -r -d '' f; do
    base="$(basename "${f}")"
    dest="${DATA_TRAIN}/${base}"
    if [[ -e "${dest}" ]]; then
      log "Skip duplicate name (already exists): ${base}"
      continue
    fi
    mv "${f}" "${DATA_TRAIN}/" && log "Moved: ${base}"
  done < <(find "${BACKUP}" -type f \( -iname '*.mp4' \) -print0 2>/dev/null)
else
  log "No backup dir at ${BACKUP} (skipping video move)."
fi

# --- 4. best.pt: first sensible hit → models/pretrained/best.pt ---
mapfile -t PT_FILES < <(find "${WORKSPACE}" -type f \( -name 'best.pt' -o -name 'yolov8*.pt' \) 2>/dev/null | sort)
if [[ ${#PT_FILES[@]} -gt 0 ]]; then
  primary="${PT_FILES[0]}"
  if [[ "${primary}" != "${MODEL_DIR}/best.pt" ]]; then
    mkdir -p "${MODEL_DIR}"
    mv -f "${primary}" "${MODEL_DIR}/best.pt"
    log "Moved to ${MODEL_DIR}/best.pt from: ${primary}"
  else
    log "best.pt already at ${MODEL_DIR}/best.pt"
  fi
  if [[ ${#PT_FILES[@]} -gt 1 ]]; then
    log "Note: additional weight files exist (${#PT_FILES[@]} total); using first after sort: ${primary}"
  fi
else
  log "WARNING: No best.pt (or yolov8*.pt) found under ${WORKSPACE}. YOLO will fail until you upload weights."
fi

# --- 5. Homography JSON → output (per-match; no global stub) ---
# cloud_batch_processor.py requires ONE real file per video:
#   output/{video_stem}_homographies.json
# with a non-empty "homographies" array. It will SKIP matches without it (no wrong-map fallback).
# Generate per match: python3 scripts/run_calibrator_on_video.py --video data/training_samples/MATCH.mp4
mapfile -t H_FILES < <(find "${WORKSPACE}" -type f -name '*homographies*.json' 2>/dev/null | sort)
for h in "${H_FILES[@]}"; do
  base="$(basename "${h}")"
  if [[ "${h}" == "${OUT_DIR}/${base}" ]]; then
    continue
  fi
  cp -f "${h}" "${OUT_DIR}/${base}"
  log "Copied homography map: ${base}"
done
if [[ ${#H_FILES[@]} -eq 0 ]]; then
  log "No *homographies*.json found under ${WORKSPACE}. Run calibrator per video before cloud_batch_processor.py."
fi

# --- 6. PYTHONPATH ---
export PYTHONPATH="${WORKSPACE}"
PROFILE_SNIPPET="${WORKSPACE}/set_gaffers_env.sh"
cat > "${PROFILE_SNIPPET}" << EOF
export PYTHONPATH=${WORKSPACE}
export PYTHONWARNINGS=ignore::FutureWarning
EOF
chmod +x "${PROFILE_SNIPPET}" 2>/dev/null || true
log "Wrote ${PROFILE_SNIPPET} — source it in new shells: source ${PROFILE_SNIPPET}"

log "Done."
echo ""
echo "=== Next: start GPU batch (logs to /workspace/gpu_processing.log) ==="
echo "source ${PROFILE_SNIPPET}"
echo "cd ${WORKSPACE} && nohup python3 -u scripts/cloud_batch_processor.py >> /workspace/gpu_processing.log 2>&1 &"
echo "tail -f /workspace/gpu_processing.log"
echo ""
