---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L188"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Mean L2 distance in seg pixels between projections of four pitch corners.

## Connections
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]
- [[_mean_corner_disagreement_px()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8