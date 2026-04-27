"""Extract and merge `philosophies` JSON from the two tactical knowledge-base PDFs."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz  # pymupdf

LOGGER = logging.getLogger(__name__)

KNOWN_FLAWS: list[str] = [
    "Midfield Disconnect",
    "Suicidal High Line",
    "Parked Bus",
    "Lethargic Press",
    "Over-Stretched Formation",
]

# Curated fallbacks when PDF text lacks explicit flaw strings (e.g. military / meta entries).
MANUAL_TAG_PREFIXES: list[tuple[str, str, list[str]]] = [
    (
        "Fernando Diniz / Relationism",
        "We play within the chaos",
        ["Midfield Disconnect"],
    ),
    ("Sun Tzu / Military Strategy", "All warfare", ["Midfield Disconnect"]),
    (
        "Niccolò Machiavelli / Military Strategy",
        "A commander should arrange",
        ["Midfield Disconnect"],
    ),
    (
        "Carlo Ancelotti / Functional Play",
        "Dominating the midfield",
        ["Midfield Disconnect"],
    ),
    ("Sean Dyche / Direct Play", "Keep them one side", ["Parked Bus"]),
    (
        "Johan Cruyff / Positional Play",
        "Playing football is very simple",
        ["Midfield Disconnect"],
    ),
    ("Helenio Herrera / Catenaccio", "If you play for yourself", ["Parked Bus"]),
    (
        "Antonio Conte / Pragmatism",
        "You have to be prepared",
        ["Over-Stretched Formation"],
    ),
    (
        "Sam Allardyce / Direct Play",
        "There is more focus on set pieces",
        ["Midfield Disconnect"],
    ),
]


def fix_multiline_strings(text: str) -> str:
    """Replace illegal JSON line breaks inside quoted strings with spaces."""

    out: list[str] = []
    i = 0
    in_str = False
    escape = False
    while i < len(text):
        c = text[i]
        if not in_str:
            if c == '"':
                in_str = True
            out.append(c)
        else:
            if escape:
                out.append(c)
                escape = False
            elif c == "\\":
                out.append(c)
                escape = True
            elif c == '"':
                in_str = False
                out.append(c)
            elif c in "\n\r\t":
                out.append(" ")
            else:
                out.append(c)
        i += 1
    return "".join(out)


def parse_philosophies_chunk(pdf_path: Path) -> list[dict[str, Any]]:
    """Parse the philosophies JSON block at the end of a knowledge-base PDF."""

    doc = fitz.open(pdf_path)
    t = "".join(page.get_text() for page in doc).replace("\u200b", "")
    doc.close()
    wc = t.find("Works cited")
    if wc < 0:
        wc = len(t)
    start = t.find('"philosophies"')
    if start < 0:
        raise ValueError(f"No philosophies JSON found in {pdf_path}")
    chunk = t[start:wc]
    chunk = chunk.replace('"philosophies":,', '"philosophies": [')
    chunk = chunk.replace('"tags":,', '"tags": [],')
    chunk = chunk.replace(
        '"fc_role_recommendations":,', '"fc_role_recommendations": null,'
    )
    chunk = re.sub(
        r'("philosophies":\s*\[)\s*\n\s*"author":',
        r'\1\n    {\n      "tags": [],\n      "author":',
        chunk,
        count=1,
    )
    chunk = "{" + chunk.strip()
    chunk = fix_multiline_strings(chunk)
    data = json.loads(chunk)
    return data["philosophies"]


def infer_tags(ph: dict[str, Any]) -> list[str]:
    raw = (ph.get("tactical_translation") or "") + " " + (ph.get("quote") or "")
    text = re.sub(r"\s+", " ", raw.lower())
    found: set[str] = set()
    for f in KNOWN_FLAWS:
        if f.lower() in text:
            found.add(f)
    if not found:
        if any(
            k in text
            for k in [
                "midfield disconnect",
                "disjointed midfield",
                "disconnected zone",
                "disconnected void",
                "flagged central gap",
                "gap in their compromised core",
                "massive gap",
                "vertical distance",
            ]
        ):
            found.add("Midfield Disconnect")
        if any(
            k in text
            for k in [
                "suicidal high line",
                "high defensive line",
                "halfway line",
                "recovery phase",
                "draw their defensive line even higher",
            ]
        ):
            found.add("Suicidal High Line")
        if any(
            k in text
            for k in [
                "parked bus",
                "low block",
                "18-yard",
                "catenaccio",
                "compressed block",
                "mid-block",
            ]
        ):
            found.add("Parked Bus")
        if any(
            k in text
            for k in [
                "lethargic press",
                "lethargic opponent",
                "lethargic",
                "loose press",
                "passive press",
                "static, lethargic",
            ]
        ):
            found.add("Lethargic Press")
        if any(
            k in text
            for k in [
                "over-stretched",
                "overstretched",
                "45-meter",
                "team length",
                "stretched formation",
                "expand the gaps",
            ]
        ):
            found.add("Over-Stretched Formation")
    return sorted(found)


def apply_manual_tags(ph: dict[str, Any]) -> None:
    if ph.get("tags"):
        return
    quote = (ph.get("quote") or "").strip()
    for author, prefix, tags in MANUAL_TAG_PREFIXES:
        if ph["author"] == author and quote.startswith(prefix):
            ph["tags"] = tags
            return


def merge_library(pdf_fc25: Path, pdf_coaching: Path) -> dict[str, Any]:
    p1 = parse_philosophies_chunk(pdf_fc25)
    p2 = parse_philosophies_chunk(pdf_coaching)
    merged = p1 + p2
    for ph in merged:
        existing = ph.get("tags") or []
        inferred = infer_tags(ph)
        if existing and any(isinstance(t, str) and t for t in existing):
            ph["tags"] = list(dict.fromkeys([t for t in existing if t] + inferred))
        else:
            ph["tags"] = inferred
        apply_manual_tags(ph)
    return {"philosophies": merged}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fc25",
        type=Path,
        required=True,
        help="Path to FC 25 Tactical AI Knowledge Base.pdf",
    )
    parser.add_argument(
        "--coaching",
        type=Path,
        required=True,
        help="Path to AI Coaching Engine Tactical Knowledge Base.pdf",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("backend/data/tactical_library.json"),
        help="Output JSON path (default: backend/data/tactical_library.json)",
    )
    args = parser.parse_args()
    data = merge_library(args.fc25, args.coaching)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    n = len(data["philosophies"])
    LOGGER.info("Wrote %s philosophies to %s", n, args.out)


if __name__ == "__main__":
    main()
