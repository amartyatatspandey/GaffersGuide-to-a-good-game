#!/usr/bin/env bash
# Resumable SoccerNet ReID zip downloads (same endpoints as SoccerNet pip Downloader).
# Run from repo root: bash backend/scripts/download_reid_curl.sh
set -euo pipefail

REID_DIR="${1:-backend/data/soccernet_reid/soccernetv3/reid}"
BASE_URL="https://exrcsdrive.kaust.edu.sa/public.php/webdav"
USER="Ffr8fsJcljh2Ds5"
PASS="SoccerNet"

mkdir -p "${REID_DIR}"

for split in train valid test challenge; do
  url="${BASE_URL}/${split}.zip"
  out="${REID_DIR}/${split}.zip"
  echo "$(date -Iseconds) START ${split}.zip -> ${out}"
  curl --fail --location \
    --user "${USER}:${PASS}" \
    --continue-at - \
    --retry 100 --retry-delay 10 --retry-all-errors \
    --connect-timeout 30 \
    --output "${out}" \
    "${url}"
  echo "$(date -Iseconds) DONE ${split}.zip"
done

echo "$(date -Iseconds) All ReID zips finished."
