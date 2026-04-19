---
source_file: "backend/scripts/pipeline_core/dynamic_homography.py"
type: "rationale"
community: "Community 8"
location: "L107"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_8
---

# Run segmentation through coarse SVD homography at SEG resolution.          :para

## Connections
- [[.collect_pitch_observations()]] - `rationale_for` [EXTRACTED]
- [[PitchLineModel]] - `uses` [INFERRED]
- [[SegmentationNetwork]] - `uses` [INFERRED]

#graphify/rationale #graphify/INFERRED #community/Community_8