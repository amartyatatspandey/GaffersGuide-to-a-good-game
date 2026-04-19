---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L303"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Each observation is (line_class, u_pixel, v_pixel) for one extremity.

## Connections
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]
- [[_build_residuals_factory()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8