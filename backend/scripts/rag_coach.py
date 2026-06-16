"""RAG-style synthesizer: map chunk-level tactical insights to philosophy and build LLM prompts."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def convert_tactical_patterns(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. If 'tactical_patterns' exists, auto-map to 'philosophies'
            if "tactical_patterns" in data and "philosophies" not in data:
                data["philosophies"] = data.pop("tactical_patterns")
            
            # 2. Ensure all items in 'philosophies' conform to PhilosophyEntry
            if "philosophies" in data and isinstance(data["philosophies"], list):
                phils = []
                for tp in data["philosophies"]:
                    if "author" not in tp and "quote" not in tp:
                        # This is a legacy/ZSL tactical pattern
                        name = tp.get("name", "Unknown Pattern")
                        desc = tp.get("description", "")
                        tags = tp.get("tags", [name.lower()])
                        
                        phils.append({
                            "tags": tags,
                            "author": "Zero-Shot Model",
                            "quote": name,
                            "tactical_translation": desc,
                            "fc_role_recommendations": []
                        })
                    else:
                        phils.append(tp)
                data["philosophies"] = phils
        return data


class GeneratedPromptRecord(BaseModel):
    """One synthesized master prompt for a single flaw instance."""

    # Kept for downstream compatibility; chunk-level insights don't have a natural frame index.
    frame_idx: int
    team: Literal["team_0", "team_1", "global"]
    flaw: str
    severity: str
    evidence: str
    frequency_pct: float
    ball_data_quality: Literal["sufficient", "insufficient"] = "sufficient"
    matched_philosophy_author: str
    matched_quote_excerpt: str
    fc_role_recommendations: list[str] | None = None
    confidence_pct: float | None = None
    confidence_reason: str | None = None
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
    ball_data_quality: Literal["sufficient", "insufficient"],
    evidence: str,
    philosophy: PhilosophyEntry,
    confidence_pct: float | None = None,
    confidence_reason: str | None = None,
) -> str:
    """Assemble a structured elite analyst prompt for the downstream LLM."""
    quote = philosophy.quote.strip()
    if len(quote) > 300:
        quote = quote[:297] + "..."

    ball_guardrail = ""
    if ball_data_quality == "insufficient":
        ball_guardrail = (
            "\n⚠️ Ball Data Guardrail: ball visibility <50%. "
            "Do NOT discuss possession, passes, or turnovers. "
            "Focus only on structural shape, compactness, spacing, and pressing geometry."
        )

    confidence_block = ""
    if confidence_pct is not None:
        confidence_block = (
            f"\n## Detection Confidence\n"
            f"- Confidence: {confidence_pct:.0f}%\n"
            f"- Reason: {confidence_reason or 'Based on frequency and threshold margin.'}\n"
        )

    roles = format_fc25_roles(philosophy)

    return f"""You are GAFFER — an elite football tactical intelligence system. Analyze the match data below and deliver a precise coaching brief.

## Detected Tactical Flaw
- **Flaw**: {flaw}
- **Severity**: {severity}
- **Frequency**: {frequency_pct:.1f}% of video frames violated this rule
- **Ball Data Quality**: {ball_data_quality}
- **Evidence**: {evidence}{ball_guardrail}
{confidence_block}
## Philosophy Anchor
- **Coach**: {philosophy.author}
- **Principle**: "{quote}"
- **Tactical Translation**: {philosophy.tactical_translation.strip()[:400]}

## FC 25 Role Recommendations
{roles}

## Your Output (structured coaching brief)
Respond in this EXACT format — no deviations:

**Tactical Verdict** (1 sentence referencing the metric evidence above):
[Your verdict here]

**3 Immediate Coaching Steps**:
1. [Formation/pressing/role adjustment — be specific to the flaw evidence]
2. [Second step — reference which team and which zone is affected]
3. [Third step — include a substitution idea or role swap if applicable]

**Confidence**: {f"{confidence_pct:.0f}%" if confidence_pct else "[X]%"} — [one line explaining WHY you are this confident based on the frequency and metric margin]

Keep the entire response under 200 words. Be decisive. Reference the numbers.
"""



def process_insights(
    insights: list[ChunkTacticalInsight],
    library: TacticalLibrary,
) -> list[GeneratedPromptRecord]:
    """For every chunk insight, attach the first matching philosophy and build a prompt."""
    out: list[GeneratedPromptRecord] = []

    for insight in insights:
        team_key = cast(Literal["team_0", "team_1", "global"], insight.team_id)
        
        # Global Match Summary rows are informational cards — pass them through directly
        # without attempting a philosophy tag match (none will exist).
        if team_key == "global":
            summary_data = getattr(insight, "summary_data", None)
            out.append(
                GeneratedPromptRecord(
                    frame_idx=0,
                    team="global",
                    flaw=insight.flaw,
                    severity=insight.severity,
                    evidence=insight.evidence,
                    frequency_pct=insight.frequency_pct,
                    ball_data_quality=insight.ball_data_quality,
                    matched_philosophy_author="GAFFER Match Engine",
                    matched_quote_excerpt="Structured analysis based on spatial telemetry.",
                    fc_role_recommendations=[],
                    confidence_pct=insight.confidence_pct,
                    confidence_reason=insight.confidence_reason,
                    llm_prompt=(
                        f"Summarize the overall match tactical state in 3 concise coaching points:\n"
                        f"Evidence: {insight.evidence}\n"
                        f"Be specific, use win probability data, and make recommendations for both teams."
                    ),
                )
            )
            continue

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
            ball_data_quality=insight.ball_data_quality,
            evidence=insight.evidence,
            philosophy=ph,
            confidence_pct=insight.confidence_pct,
            confidence_reason=insight.confidence_reason,
        )
        out.append(
            GeneratedPromptRecord(
                frame_idx=0,
                team=team_key,
                flaw=insight.flaw,
                severity=insight.severity,
                evidence=insight.evidence,
                frequency_pct=insight.frequency_pct,
                ball_data_quality=insight.ball_data_quality,
                matched_philosophy_author=ph.author,
                matched_quote_excerpt=excerpt,
                fc_role_recommendations=ph.fc_role_recommendations,
                confidence_pct=insight.confidence_pct,
                confidence_reason=insight.confidence_reason,
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
