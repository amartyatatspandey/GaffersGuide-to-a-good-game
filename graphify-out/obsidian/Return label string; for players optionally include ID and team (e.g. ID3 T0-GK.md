---
source_file: "backend/scripts/pipeline_core/track_teams.py"
type: "rationale"
community: "Community 5"
location: "L114"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_5
---

# Return label string; for players optionally include ID and team (e.g. ID:3 T0-GK

## Connections
- [[HybridIDHealer]] - `uses` [INFERRED]
- [[TacticalRadar]] - `uses` [INFERRED]
- [[TeamClassifier]] - `uses` [INFERRED]
- [[get_detection_label()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_5