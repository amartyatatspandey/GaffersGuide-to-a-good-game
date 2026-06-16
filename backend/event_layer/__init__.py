"""
Gaffer's Guide — Event Intelligence Layer
==========================================

This package sits between raw tracking data and the LLM.

Architecture
------------
TrackingFrameArtifact[]
    └─► EventDetectionPipeline
            ├── MovementDetector
            ├── PositionalDetector
            ├── ThreatDetector
            ├── ShapeDetector
            └── TransitionDetector
    └─► EventRecord[]  ──► EventIndex (JSON)
            └─► ThreatAttributor  ──► PlayerThreatProfile[]
            └─► EvidenceRetriever ──► EvidenceBundle[]

Public API
----------
>>> from event_layer.pipeline import EventDetectionPipeline
>>> from event_layer.threat import ThreatAttributor
>>> from event_layer.retrieval import EvidenceRetriever
"""
from __future__ import annotations

__version__ = "1.0.0"
