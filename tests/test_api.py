"""API tests: health and basic endpoints."""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_returns_200_and_ok() -> None:
    """GET /health returns 200 and JSON {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
