---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L64"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Convert sn-calibration polyline (row, col) to pixel segments (u, v).

## Connections
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]
- [[_polyline_to_pixel_segments()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8