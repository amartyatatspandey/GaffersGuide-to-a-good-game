"""Base exception hierarchy for gaffers-guide SDK."""

from __future__ import annotations


class GaffersGuideError(Exception):
    """Base exception for all gaffers-guide errors."""


class DetectionError(GaffersGuideError):
    """Raised when detection or tracking processing fails."""


class SpatialMappingError(GaffersGuideError):
    """Raised when homography or pitch mapping fails."""


class ConfigurationError(GaffersGuideError):
    """Raised when configuration is invalid or incomplete."""
