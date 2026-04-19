---
source_file: "backend/scripts/pipeline_core/advanced_pitch_calibration.py"
type: "rationale"
community: "Community 8"
location: "L138"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Image points (N, 2) in pixel coords and list of point_dict keys for world corres

## Connections
- [[DynamicPitchCalibrator]] - `uses` [INFERRED]
- [[PitchObservationBundle]] - `uses` [INFERRED]
- [[_collect_corner_image_points()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_8