#!/usr/bin/env bash
# RunPod /workspace: per-match homography generation then cloud_batch_processor.
# Expects layout: /workspace/{scripts,data,models,output,references}, PYTHONPATH=/workspace.

set -euo pipefail
WORKSPACE=/workspace
cd "$WORKSPACE"
export PYTHONPATH="$WORKSPACE"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
DATA_TRAIN="$WORKSPACE/data/training_samples"
OUT_DIR="$WORKSPACE/output"
CAL_WEIGHTS="$WORKSPACE/references/sn-calibration/resources"
LOG="${SETUP_RUN_LOG:-$WORKSPACE/setup_and_run.log}"
exec >>"$LOG" 2>&1
echo "[$(date -Iseconds)] setup_and_run start"
mkdir -p "$OUT_DIR/cloud_results" "$WORKSPACE/models/pretrained"
if [[ ! -f "$WORKSPACE/models/pretrained/best.pt" ]]; then
  echo "ERROR: missing $WORKSPACE/models/pretrained/best.pt"
  exit 1
fi
shopt -s nullglob
vids=( "$DATA_TRAIN"/*.mp4 )
shopt -u nullglob
if [[ ${#vids[@]} -eq 0 ]]; then
  echo "ERROR: no mp4 in $DATA_TRAIN"
  exit 1
fi
for mp4 in "${vids[@]}"; do
  stem=$(basename "$mp4" .mp4)
  out_json="$OUT_DIR/${stem}_homographies.json"
  echo "[$(date -Iseconds)] calibrate $stem"
  python3 -u scripts/run_calibrator_on_video.py --video "$mp4" --output "$out_json" --weights-dir "$CAL_WEIGHTS" --sample-every 30
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d.get(\"homographies\"), \"empty\"" "$out_json"
done
echo "[$(date -Iseconds)] start cloud_batch_processor"
python3 -u scripts/cloud_batch_processor.py
echo "[$(date -Iseconds)] setup_and_run done"
