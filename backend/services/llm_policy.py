from __future__ import annotations

import re


def build_structured_coaching_prompt(*, user_prompt: str, context: str = "", history: list[dict[str, str]] | None = None) -> str:
    """Build a shared prompt contract for chat/advice responses.
    
    When match context is available, the LLM is forced to behave as an elite tactical
    analyst referencing real match telemetry, not a general football Q&A bot.
    """
    
    history_str = ""
    if history:
        history_str = "## Recent conversation (last 4 turns)\n"
        for turn in history[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_str += f"{role.capitalize()}: {content}\n"
        history_str += "\n"

    has_match_context = bool(context and context.strip())

    if has_match_context:
        system_block = (
            "You are GAFFER — an elite football tactical intelligence system with the precision of an analyst, "
            "the authority of a top-flight manager, and the clarity of a UEFA Pro License coach.\n\n"
            "CRITICAL OPERATING MODE: You have live match telemetry from a real uploaded match. You MUST:\n"
            "- Reference the specific flaws detected in this match\n"
            "- Use the win probability and tactical power data provided\n"
            "- Explain WHY each issue is occurring using the metric evidence\n"
            "- Compare both teams based on the data\n"
            "- Predict which team currently has tactical momentum\n"
            "- Suggest concrete substitution ideas or formation adjustments\n"
            "- Assign a confidence percentage to your recommendation\n"
            "- NEVER give generic football definitions or textbook theory alone — always anchor to the live match data below\n\n"
            "IF the user asks something general (e.g., 'What is tiki-taka?'), STILL answer in the context of this "
            "specific match (e.g., 'Based on this match, your team's build-up shows similarities to tiki-taka principles...')"
        )
    else:
        system_block = (
            "You are GAFFER — an elite football tactical intelligence system. "
            "No live match is currently loaded, so you may answer general football questions with expert precision."
        )

    context_block = f"## Live Match Intelligence\n{context}" if has_match_context else ""

    if has_match_context:
        req_1 = "1. Start with a one-line tactical verdict referencing the match data (e.g., 'Based on the detected Midfield Disconnect at 73% frequency...')."
        req_3 = "3. End with: Confidence: [X]% — [one-line reason based on the metric evidence]."
    else:
        req_1 = "1. Answer clearly and expertly."
        req_3 = "3. Keep each step concise and actionable."

    return f"""{system_block}

{context_block}

{history_str}## Manager's Question
{user_prompt}

## Response Requirements
{req_1}
2. Provide exactly 3 numbered tactical steps — concrete, immediately applicable.
{req_3}
4. Keep the entire response under 180 words.
5. Do NOT output generic coaching platitudes. Be precise, data-driven, and decisive.
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
