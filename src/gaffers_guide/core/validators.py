"""Validation utilities for core SDK contracts."""

from __future__ import annotations


def require_non_empty_string(value: str, *, field_name: str) -> str:
    """Return a trimmed non-empty string or raise ValueError."""
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(f"{field_name} must be a non-empty string")
    return trimmed
