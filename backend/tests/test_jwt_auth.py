import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi import WebSocket
from fastapi.testclient import TestClient

# Mock environmental vars before imports
os.environ["API_KEY"] = "test-api-key-123"
os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret-abc"

from main import app, verify_ws_auth

# Disable startup/shutdown events
app.router.on_startup.clear()
app.router.on_shutdown.clear()

client = TestClient(app)

def test_public_endpoints():
    # Health check is public
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_key_auth_success():
    response = client.get("/api/v1/matches", headers={"x-api-key": "test-api-key-123", "x-test-force-auth": "1"})
    # Status should not be 401 (either 200 or 404 or mock-specific error, but not 401)
    assert response.status_code != 401

def test_api_key_auth_failure():
    response = client.get("/api/v1/matches", headers={"x-api-key": "wrong-key", "x-test-force-auth": "1"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing credentials"

def test_jwt_auth_failure_invalid_token():
    response = client.get("/api/v1/matches", headers={"Authorization": "Bearer invalid-token-string", "x-test-force-auth": "1"})
    assert response.status_code == 401
    assert "Invalid token" in response.json()["detail"]

@patch("jwt.decode")
def test_jwt_auth_success(mock_decode):
    # Mock jwt.decode to return a valid payload
    mock_decode.return_value = {
        "sub": "user-uuid-111",
        "email": "coach@test.com",
        "role": "authenticated"
    }
    
    response = client.get("/api/v1/matches", headers={"Authorization": "Bearer valid-token", "x-test-force-auth": "1"})
    assert response.status_code != 401
    mock_decode.assert_called_once_with(
        "valid-token",
        "test-jwt-secret-abc",
        algorithms=["HS256"],
        audience="authenticated"
    )

def test_verify_ws_auth_api_key():
    ws_mock = MagicMock(spec=WebSocket)
    ws_mock.query_params = {"api_key": "test-api-key-123", "x-test-force-auth": "1"}
    assert verify_ws_auth(ws_mock) is True

    ws_mock.query_params = {"api_key": "wrong-key", "x-test-force-auth": "1"}
    assert verify_ws_auth(ws_mock) is False

@patch("jwt.decode")
def test_verify_ws_auth_jwt(mock_decode):
    ws_mock = MagicMock(spec=WebSocket)
    ws_mock.query_params = {"token": "valid-token", "x-test-force-auth": "1"}
    
    mock_decode.return_value = {"sub": "user-123"}
    assert verify_ws_auth(ws_mock) is True
    
    mock_decode.side_effect = Exception("Invalid token")
    assert verify_ws_auth(ws_mock) is False
