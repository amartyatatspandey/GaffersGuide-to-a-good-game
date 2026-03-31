from __future__ import annotations

import re


def build_structured_coaching_prompt(*, user_prompt: str, context: str = "") -> str:
    """Build a shared prompt contract for chat/advice responses."""
    return f"""You are an elite football coach and tactician.

{context}

User question:
{user_prompt}

Output requirements:
1. Provide exactly 3 numbered tactical steps.
2. Keep it under 150 words.
3. Keep each step concrete and action-oriented.
4. Do not output a paragraph block; output only numbered steps.
"""


def normalize_instruction_steps(text: str | None) -> list[str]:
    """Convert model output into up to 3 concise tactical steps."""
    if not isinstance(text, str):
        return []
    cleaned = text.strip()
    if not cleaned:
        return []

    bullet_prefix = re.compile(r"^\s*(?:\d+[.)]\s*|[-*]\s+)")
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    steps = [bullet_prefix.sub("", ln).strip() for ln in lines]
    steps = [s for s in steps if s]

    if len(steps) <= 1:
        chunks = re.split(r"(?<=[.!?])\s+", cleaned)
        steps = [c.strip() for c in chunks if c.strip()]

    deduped: list[str] = []
    for step in steps:
        normalized = " ".join(step.split())
        if normalized and normalized not in deduped:
            deduped.append(normalized)
        if len(deduped) >= 3:
            break
    return deduped


def format_numbered_steps(steps: list[str], fallback_text: str | None) -> str | None:
    """Preserve backward-compatible string response while guaranteeing readability."""
    if steps:
        return "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(steps)])
    if isinstance(fallback_text, str):
        out = fallback_text.strip()
        return out or None
    return None
