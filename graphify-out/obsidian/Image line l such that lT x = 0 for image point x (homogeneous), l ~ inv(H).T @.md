---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L280"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Image line l such that l^T x = 0 for image point x (homogeneous), l ~ inv(H).T @

## Connections
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]
- [[_line_image_from_pitch()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8