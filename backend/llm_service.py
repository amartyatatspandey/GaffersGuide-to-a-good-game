"""Google Gemini helpers for tactical coaching completions."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_ROOT / ".env")

_GEMINI_CONFIGURED: bool = False


def _ensure_gemini_configured() -> None:
    """Configure the Gemini client once per process."""

    global _GEMINI_CONFIGURED
    if _GEMINI_CONFIGURED:
        return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    _GEMINI_CONFIGURED = True


def generate_coaching_advice(prompt: str) -> str:
    """
    Call Gemini with the assembled RAG prompt and return plain text coaching advice.

    Args:
        prompt: Full user prompt (system-style instructions are embedded in the prompt).

    Returns:
        Model response text.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is missing.
        RuntimeError: If the model returns no usable text (e.g. safety block).
    """
    _ensure_gemini_configured()
    # Default: current Flash on the Gemini API (`gemini-1.5-flash` IDs often 404 on v1beta).
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.35, "max_output_tokens": 600},
    )
    try:
        text = response.text
    except (ValueError, AttributeError) as exc:
        LOGGER.warning("Gemini returned no extractable text: %s", exc)
        raise RuntimeError("Gemini response had no text (blocked or empty).") from exc
    return text.strip()


def gemini_is_configured() -> bool:
    """Return True when ``GEMINI_API_KEY`` is present in the environment."""

    return bool(os.getenv("GEMINI_API_KEY"))
