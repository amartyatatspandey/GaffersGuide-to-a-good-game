---
source_file: "backend/scripts/pipeline_core/dynamic_homography.py"
type: "rationale"
community: "Community 8"
location: "L61"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Standalone calibrator: one video frame (BGR) in -> 3x3 homography (pitch -> imag

## Connections
- [[DynamicPitchCalibrator]] - `rationale_for` [EXTRACTED]
- [[PitchLineModel]] - `uses` [INFERRED]
- [[SegmentationNetwork]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_8