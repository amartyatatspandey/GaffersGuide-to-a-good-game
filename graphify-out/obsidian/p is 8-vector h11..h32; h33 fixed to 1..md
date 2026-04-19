---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L241"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# p is 8-vector h11..h32; h33 fixed to 1.

## Connections
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]
- [[_params_to_H()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8