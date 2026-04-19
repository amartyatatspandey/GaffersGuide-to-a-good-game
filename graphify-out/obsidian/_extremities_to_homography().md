---
source_file: "backend/scripts/pipeline_core/dynamic_homography.py"
type: "code"
community: "camera.py (42+)"
location: "L74"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/camera.py_(42+)
---

# _extremities_to_homography()

## Connections
- [[.collect_pitch_observations()]] - `calls` [EXTRACTED]
- [[Estimate 3x3 homography (pitch plane - image) from line extremities.     Return]] - `rationale_for` [EXTRACTED]
- [[_build_line_matches()]] - `calls` [EXTRACTED]
- [[dynamic_homography.py]] - `contains` [EXTRACTED]
- [[estimate_homography_from_line_correspondences()]] - `calls` [INFERRED]
- [[normalization_transform()]] - `calls` [INFERRED]

#graphify/code #graphify/EXTRACTED #community/camera.py_(42+)