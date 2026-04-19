---
type: community
cohesion: 0.29
members: 8
---

# llm_service.py (8+)

**Cohesion:** 0.29 - loosely connected
**Members:** 8 nodes

## Members
- [[Call Gemini with the assembled RAG prompt and return plain text coaching advice.]] - rationale - backend/llm_service.py
- [[Configure the Gemini client once per process.]] - rationale - backend/llm_service.py
- [[Google Gemini helpers for tactical coaching completions.]] - rationale - backend/llm_service.py
- [[Return True when ``GEMINI_API_KEY`` is present in the environment.]] - rationale - backend/llm_service.py
- [[_ensure_gemini_configured()]] - code - backend/llm_service.py
- [[gemini_is_configured()]] - code - backend/llm_service.py
- [[generate_coaching_advice()]] - code - backend/llm_service.py
- [[llm_service.py]] - code - backend/llm_service.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/llm_service.py_(8+)
SORT file.name ASC
```

## Connections to other communities
- 1 edge to [[_COMMUNITY_run_e2e_legacy.py (66+)]]
- 1 edge to [[_COMMUNITY_main.py (50+)]]

## Top bridge nodes
- [[gemini_is_configured()]] - degree 4, connects to 2 communities