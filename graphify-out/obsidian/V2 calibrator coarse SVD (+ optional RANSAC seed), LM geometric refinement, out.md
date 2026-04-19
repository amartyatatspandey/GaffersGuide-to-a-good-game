---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L402"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# V2 calibrator: coarse SVD (+ optional RANSAC seed), LM geometric refinement, out

## Connections
- [[AdvancedPitchCalibrator]] - `rationale_for` [EXTRACTED]
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_8