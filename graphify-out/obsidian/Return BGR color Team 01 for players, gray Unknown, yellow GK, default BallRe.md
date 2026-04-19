---
source_file: "backend/scripts/pipeline_core/track_teams.py"
type: "rationale"
community: "Community 5"
location: "L93"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_5
---

# Return BGR color: Team 0/1 for players, gray Unknown, yellow GK, default Ball/Re

## Connections
- [[HybridIDHealer]] - `uses` [INFERRED]
- [[TacticalRadar]] - `uses` [INFERRED]
- [[TeamClassifier]] - `uses` [INFERRED]
- [[get_detection_color()]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_5