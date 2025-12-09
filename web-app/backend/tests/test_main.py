"""Tests for main application endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client):
    """Test root endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "0.1.0"
    assert data["status"] == "operational"


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "llm-judge-auditor-backend"
    assert data["version"] == "0.1.0"


def test_request_id_middleware(client):
    """Test that request ID is added to response headers."""
    response = client.get("/")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options(
        "/",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        }
    )
    assert "access-control-allow-origin" in response.headers


def test_validation_error_handling(client):
    """Test validation error returns structured response."""
    # Try to create evaluation with invalid data
    response = client.post(
        "/api/v1/evaluations",
        json={"invalid": "data"},
        headers={"Authorization": "Bearer fake_token"}
    )
    
    # Should get validation error or auth error
    assert response.status_code in [401, 422]


def test_not_found_error(client):
    """Test 404 error handling."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
