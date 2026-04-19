---
source_file: "backend/scripts/pipeline_core/run_calibrator_on_video.py"
type: "rationale"
community: "Community 8"
location: "L1"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Run pitch calibrator on a video file (e.g. match_test.mp4).  By default uses ``A

## Connections
- [[AdvancedPitchCalibrator]] - `uses` [INFERRED]
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[run_calibrator_on_video.py]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8