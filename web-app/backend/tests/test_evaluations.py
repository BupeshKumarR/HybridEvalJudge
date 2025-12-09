"""Tests for evaluation endpoints."""

import pytest
from fastapi import status
from uuid import uuid4


@pytest.fixture
def evaluation_data():
    """Sample evaluation data."""
    return {
        "source_text": "The capital of France is Paris.",
        "candidate_output": "Paris is the capital city of France.",
        "config": {
            "judge_models": ["gpt-4", "claude-3"],
            "enable_retrieval": True,
            "aggregation_strategy": "weighted_average"
        }
    }


@pytest.fixture
def created_evaluation(client, auth_headers, evaluation_data):
    """Create a test evaluation session."""
    response = client.post(
        "/api/v1/evaluations",
        json=evaluation_data,
        headers=auth_headers
    )
    return response.json()


def test_create_evaluation(client, auth_headers, evaluation_data):
    """Test creating a new evaluation session."""
    response = client.post(
        "/api/v1/evaluations",
        json=evaluation_data,
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert "id" in data
    assert data["source_text"] == evaluation_data["source_text"]
    assert data["candidate_output"] == evaluation_data["candidate_output"]
    assert data["status"] == "pending"
    assert data["config"] is not None


def test_create_evaluation_no_auth(client, evaluation_data):
    """Test creating evaluation without authentication fails."""
    response = client.post("/api/v1/evaluations", json=evaluation_data)
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_create_evaluation_minimal(client, auth_headers):
    """Test creating evaluation with minimal data."""
    minimal_data = {
        "source_text": "Test source",
        "candidate_output": "Test output"
    }
    
    response = client.post(
        "/api/v1/evaluations",
        json=minimal_data,
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["config"] is None


def test_create_evaluation_empty_text(client, auth_headers):
    """Test creating evaluation with empty text fails."""
    invalid_data = {
        "source_text": "",
        "candidate_output": "Test output"
    }
    
    response = client.post(
        "/api/v1/evaluations",
        json=invalid_data,
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_evaluation(client, auth_headers, created_evaluation):
    """Test getting evaluation by ID."""
    session_id = created_evaluation["id"]
    
    response = client.get(
        f"/api/v1/evaluations/{session_id}",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == session_id
    assert "source_text" in data
    assert "candidate_output" in data


def test_get_evaluation_not_found(client, auth_headers):
    """Test getting nonexistent evaluation returns 404."""
    fake_id = str(uuid4())
    
    response = client.get(
        f"/api/v1/evaluations/{fake_id}",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_evaluation_no_auth(client, created_evaluation):
    """Test getting evaluation without authentication fails."""
    session_id = created_evaluation["id"]
    
    response = client.get(f"/api/v1/evaluations/{session_id}")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_list_evaluations(client, auth_headers, created_evaluation):
    """Test listing evaluations."""
    response = client.get("/api/v1/evaluations", headers=auth_headers)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "sessions" in data
    assert "total" in data
    assert "page" in data
    assert "limit" in data
    assert "has_more" in data
    assert len(data["sessions"]) > 0


def test_list_evaluations_pagination(client, auth_headers, created_evaluation):
    """Test evaluation list pagination."""
    response = client.get(
        "/api/v1/evaluations?page=1&limit=10",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["page"] == 1
    assert data["limit"] == 10


def test_list_evaluations_filtering(client, auth_headers, created_evaluation):
    """Test evaluation list filtering by status."""
    response = client.get(
        "/api/v1/evaluations?status_filter=pending",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    for session in data["sessions"]:
        assert session["status"] == "pending"


def test_list_evaluations_no_auth(client):
    """Test listing evaluations without authentication fails."""
    response = client.get("/api/v1/evaluations")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_export_evaluation_json(client, auth_headers, created_evaluation):
    """Test exporting evaluation as JSON."""
    session_id = created_evaluation["id"]
    
    response = client.get(
        f"/api/v1/evaluations/{session_id}/export?format=json",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "application/json"
    assert "attachment" in response.headers.get("content-disposition", "")


def test_export_evaluation_csv(client, auth_headers, created_evaluation):
    """Test exporting evaluation as CSV."""
    session_id = created_evaluation["id"]
    
    response = client.get(
        f"/api/v1/evaluations/{session_id}/export?format=csv",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_200_OK
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers.get("content-disposition", "")


def test_export_evaluation_pdf_not_implemented(client, auth_headers, created_evaluation):
    """Test PDF export returns not implemented."""
    session_id = created_evaluation["id"]
    
    response = client.get(
        f"/api/v1/evaluations/{session_id}/export?format=pdf",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED


def test_export_evaluation_invalid_format(client, auth_headers, created_evaluation):
    """Test export with invalid format returns error."""
    session_id = created_evaluation["id"]
    
    response = client.get(
        f"/api/v1/evaluations/{session_id}/export?format=invalid",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_delete_evaluation(client, auth_headers, created_evaluation):
    """Test deleting an evaluation."""
    session_id = created_evaluation["id"]
    
    response = client.delete(
        f"/api/v1/evaluations/{session_id}",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_204_NO_CONTENT
    
    # Verify it's deleted
    get_response = client.get(
        f"/api/v1/evaluations/{session_id}",
        headers=auth_headers
    )
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


def test_delete_evaluation_not_found(client, auth_headers):
    """Test deleting nonexistent evaluation returns 404."""
    fake_id = str(uuid4())
    
    response = client.delete(
        f"/api/v1/evaluations/{fake_id}",
        headers=auth_headers
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_delete_evaluation_no_auth(client, created_evaluation):
    """Test deleting evaluation without authentication fails."""
    session_id = created_evaluation["id"]
    
    response = client.delete(f"/api/v1/evaluations/{session_id}")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
