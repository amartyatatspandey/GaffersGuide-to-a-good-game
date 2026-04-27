"""API tests: health and basic endpoints."""

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.main import app

client = TestClient(app)


def test_health_returns_200_and_ok() -> None:
    """GET /health returns 200 and JSON {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
