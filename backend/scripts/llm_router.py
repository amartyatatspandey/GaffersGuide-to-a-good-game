"""
Standardized LLM Routing Interface (Compatibility Layer).
Proxies to services/llm_router.py for system-wide consistency.
"""
from services.llm_router import (
    LLMEngine,
    QualityProfile,
    QUALITY_MODES,
    detect_intent,
    ensure_ollama_available,
    generate_coaching_text,
    get_tactical_advice,
    start_ollama_for_app_lifecycle,
    stop_ollama_for_app_lifecycle,
)

__all__ = [
    "LLMEngine",
    "QualityProfile",
    "QUALITY_MODES",
    "detect_intent",
    "ensure_ollama_available",
    "generate_coaching_text",
    "get_tactical_advice",
    "start_ollama_for_app_lifecycle",
    "stop_ollama_for_app_lifecycle",
]
