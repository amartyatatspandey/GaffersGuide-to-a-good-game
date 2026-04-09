# Beta LLM Policy (Unified)

## Purpose

Unify structured-output behavior across cloud and local LLMs for both coaching advice and interactive chat.

## Policy

- Shared prompt contract is centralized in `backend/services/llm_policy.py`.
- Required output format: exactly 3 numbered tactical steps.
- Post-generation normalization is always applied:
  - split bullets/numbered lines
  - sentence fallback when models return paragraph text
  - deduplicate and cap at 3 steps

## Enforcement Points

- Chat flow in `backend/main.py` uses `build_structured_coaching_prompt(...)`.
- Advice payload formatting uses:
  - `normalize_instruction_steps(...)`
  - `format_numbered_steps(...)`

## Cost/Failure Behavior

- Routing remains in `backend/services/llm_router.py` with engine-specific errors.
- Output normalization is engine-agnostic and therefore consistent for Ollama/cloud responses.
