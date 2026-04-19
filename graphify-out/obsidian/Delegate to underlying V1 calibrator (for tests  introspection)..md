---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L457"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Delegate to underlying V1 calibrator (for tests / introspection).

## Connections
- [[.collect_pitch_observations()_1]] - `rationale_for` [EXTRACTED]
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_8