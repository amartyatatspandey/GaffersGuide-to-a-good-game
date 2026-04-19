---
source_file: "backend/scripts/pipeline_core/run_calibrator_on_video.py"
type: "rationale"
community: "Community 8"
location: "L117"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Resolve homography JSON for ``video_path`` (env override or per-stem under ``out

## Connections
- [[AdvancedPitchCalibrator]] - `uses` [INFERRED]
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[ensure_homography_json_for_video()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8