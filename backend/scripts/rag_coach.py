"""RAG-style synthesizer: map chunk-level tactical insights to philosophy and build LLM prompts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Ensure `import models` works regardless of cwd.
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models import ChunkTacticalInsight  # noqa: E402

DEFAULT_TRIGGERS_PATH = BACKEND_ROOT / "output" / "tactical_triggers.json"
DEFAULT_LIBRARY_PATH = BACKEND_ROOT / "data" / "tactical_library.json"
DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "output" / "final_llm_prompts.json"


class PhilosophyEntry(BaseModel):
    """One coaching philosophy row from `tactical_library.json`."""

    tags: list[str]
    author: str
    quote: str
    tactical_translation: str
    fc_role_recommendations: list[str] | None = None


class TacticalLibrary(BaseModel):
    philosophies: list[PhilosophyEntry]


class GeneratedPromptRecord(BaseModel):
    """One synthesized master prompt for a single flaw instance."""

    # Kept for downstream compatibility; chunk-level insights don't have a natural frame index.
    frame_idx: int
    team: Literal["team_0", "team_1"]
    flaw: str
    severity: str
    evidence: str
    frequency_pct: float
    matched_philosophy_author: str
    matched_quote_excerpt: str
    fc_role_recommendations: list[str] | None = None
    llm_prompt: str


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_philosophy(
    philosophies: list[PhilosophyEntry], flaw: str
) -> PhilosophyEntry | None:
    """Return the first philosophy whose `tags` contain the given flaw name."""
    for ph in philosophies:
        if flaw in ph.tags:
            return ph
    return None


def format_fc25_roles(ph: PhilosophyEntry) -> str:
    roles = ph.fc_role_recommendations
    if not roles:
        return "No explicit FC 25 player roles were listed for this philosophy in the library."
    return ", ".join(roles)


def build_master_llm_prompt(
    *,
    flaw: str,
    severity: str,
    frequency_pct: float,
    evidence: str,
    philosophy: PhilosophyEntry,
) -> str:
    """Assemble the instruction prompt to send to the downstream LLM."""
    quote = philosophy.quote.strip()
    if len(quote) > 400:
        quote = quote[:397] + "..."

    # Explicitly label the input as a chronic, chunk-wide macro-trend.
    return f"""You are an elite football coach and tactician. Your task is to synthesize mathematical match intelligence with a named coaching philosophy and FC 25 role guidance into clear touchline instructions.

## Mathematical ground truth (verified, chronic chunk-wide macro-trend)
- Flaw: {flaw}
- Severity: {severity}
- Chronic frequency: {frequency_pct:.1f}% of frames in this video chunk violated the rule
- Evidence: {evidence}

## Philosophy anchor
- Author: {philosophy.author}
- Core quote: "{quote}"

## FC 25 player roles (use if listed; otherwise infer carefully)
{format_fc25_roles(philosophy)}

## Tactical translation (from the knowledge base)
{philosophy.tactical_translation.strip()}

## Your output requirements
1. Act as an elite coach: be decisive, specific, and aligned with the philosophy above.
2. Produce exactly **3 numbered steps** the manager can apply immediately (formation behavior, pressing cues, or role focus — be concrete).
3. Keep the **entire response under 150 words** (count carefully).
4. Do not repeat the JSON verbatim; translate it into actionable coaching language.
"""


def process_insights(
    insights: list[ChunkTacticalInsight],
    library: TacticalLibrary,
) -> list[GeneratedPromptRecord]:
    """For every chunk insight, attach the first matching philosophy and build a prompt."""
    out: list[GeneratedPromptRecord] = []

    for insight in insights:
        team_key = cast(Literal["team_0", "team_1"], insight.team_id)
        ph = find_philosophy(library.philosophies, insight.flaw)

        if ph is None:
            LOGGER.warning(
                "No philosophy with tag %r for %s; using generic prompt.",
                insight.flaw,
                team_key,
            )
            ph = PhilosophyEntry(
                tags=[insight.flaw],
                author="Generic tactical response",
                quote="Compress lines, secure the ball, and re-establish connections between units.",
                tactical_translation=(
                    f"No library entry tagged '{insight.flaw}' was found. "
                    "Remedy the issue by restoring compactness between lines, "
                    "re-linking midfield to defense, and securing rest-defense numbers."
                ),
                fc_role_recommendations=None,
            )

        q = ph.quote.strip()
        excerpt = q if len(q) <= 200 else q[:197] + "..."
        prompt = build_master_llm_prompt(
            flaw=insight.flaw,
            severity=insight.severity,
            frequency_pct=insight.frequency_pct,
            evidence=insight.evidence,
            philosophy=ph,
        )
        out.append(
            GeneratedPromptRecord(
                frame_idx=0,
                team=team_key,
                flaw=insight.flaw,
                severity=insight.severity,
                evidence=insight.evidence,
                frequency_pct=insight.frequency_pct,
                matched_philosophy_author=ph.author,
                matched_quote_excerpt=excerpt,
                fc_role_recommendations=ph.fc_role_recommendations,
                llm_prompt=prompt,
            )
        )

    return out


def run(
    triggers_path: Path = DEFAULT_TRIGGERS_PATH,
    library_path: Path = DEFAULT_LIBRARY_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[GeneratedPromptRecord]:
    if not triggers_path.is_file():
        raise FileNotFoundError(
            f"Triggers file not found: {triggers_path}. "
            "Generate it with tactical_rule_engine.py or add a sample file."
        )
    if not library_path.is_file():
        raise FileNotFoundError(
            f"Tactical library not found: {library_path}. "
            "Run extract_tactical_library_from_pdfs.py first."
        )

    raw_insights = load_json(triggers_path)
    insights = [ChunkTacticalInsight.model_validate(row) for row in raw_insights]

    raw_lib = load_json(library_path)
    library = TacticalLibrary.model_validate(raw_lib)

    records = process_insights(insights, library)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in records], f, indent=2, ensure_ascii=False)

    LOGGER.info("Wrote %s master prompt(s) to %s", len(records), output_path)
    return records


run_rag_synthesizer = run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
