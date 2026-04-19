---
source_file: "backend/scripts/pipeline_core/dynamic_homography.py"
type: "rationale"
community: "camera.py (42+)"
location: "L80"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/camera.py_(42+)
---

# Estimate 3x3 homography (pitch plane -> image) from line extremities.     Return

## Connections
- [[SegmentationNetwork]] - `uses` [INFERRED]
- [[SoccerPitch]] - `uses` [INFERRED]
- [[_extremities_to_homography()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/camera.py_(42+)