"""
Integration tests for the full evaluation pipeline.
Tests the complete flow from API request to database storage.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.models import EvaluationSession, JudgeResult, VerifierVerdict
from app.database import get_db
from tests.conftest import db_session, client, created_user, auth_headers


class TestFullEvaluationPipeline:
    """Test the complete evaluation pipeline end-to-end."""

    def test_evaluation_creates_database_records(
        self, client: TestClient, auth_headers: dict, db_session: Session
    ):
        """Test that evaluation creates all necessary database records."""
        # Create evaluation request
        evaluation_data = {
            "source_text": "The capital of France is Paris. It is known for the Eiffel Tower.",
            "candidate_output": "Paris is the capital of France and home to the famous Eiffel Tower.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        # Submit evaluation
        response = client.post(
            "/api/v1/evaluations",
            json=evaluation_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]

        # Verify session was created in database
        session = db_session.query(EvaluationSession).filter(
            EvaluationSession.id == session_id
        ).first()

        assert session is not None
        assert session.source_text == evaluation_data["source_text"]
        assert session.candidate_output == evaluation_data["candidate_output"]
        assert session.status in ["pending", "completed"]

    def test_evaluation_with_multiple_judges(
        self, client: TestClient, auth_headers: dict, db_session: Session
    ):
        """Test evaluation with multiple judge models."""
        evaluation_data = {
            "source_text": "Test source text for multiple judges evaluation.",
            "candidate_output": "Test candidate output for multiple judges.",
            "config": {
                "judge_models": ["gpt-4", "gpt-3.5-turbo", "claude-3"],
                "enable_retrieval": False,
                "aggregation_strategy": "median"
            }
        }

        response = client.post(
            "/api/v1/evaluations",
            json=evaluation_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]

        # Verify session configuration
        session = db_session.query(EvaluationSession).filter(
            EvaluationSession.id == session_id
        ).first()

        assert session is not None
        assert session.config["judge_models"] == evaluation_data["config"]["judge_models"]
        assert session.config["aggregation_strategy"] == "median"

    def test_evaluation_retrieval_and_storage(
        self, client: TestClient, auth_headers: dict, db_session: Session
    ):
        """Test that evaluation results are properly stored."""
        evaluation_data = {
            "source_text": "The Earth orbits around the Sun.",
            "candidate_output": "The Sun orbits around the Earth.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": True,
                "aggregation_strategy": "weighted_average"
            }
        }

        # Create evaluation
        response = client.post(
            "/api/v1/evaluations",
            json=evaluation_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        session_id = response.json()["session_id"]

        # Retrieve evaluation
        get_response = client.get(
            f"/api/v1/evaluations/{session_id}",
            headers=auth_headers
        )

        assert get_response.status_code == 200
        result = get_response.json()

        # Verify structure
        assert "session_id" in result
        assert "source_text" in result
        assert "candidate_output" in result
        assert "status" in result

    def test_evaluation_list_and_pagination(
        self, client: TestClient, auth_headers: dict, db_session: Session
    ):
        """Test listing evaluations with pagination."""
        # Create multiple evaluations
        for i in range(5):
            evaluation_data = {
                "source_text": f"Source text {i}",
                "candidate_output": f"Candidate output {i}",
                "config": {
                    "judge_models": ["gpt-4"],
                    "enable_retrieval": False,
                    "aggregation_strategy": "weighted_average"
                }
            }

            response = client.post(
                "/api/v1/evaluations",
                json=evaluation_data,
                headers=auth_headers
            )
            assert response.status_code == 200

        # List evaluations with pagination
        list_response = client.get(
            "/api/v1/evaluations?page=1&limit=3",
            headers=auth_headers
        )

        assert list_response.status_code == 200
        data = list_response.json()

        assert "sessions" in data
        assert "total" in data
        assert "page" in data
        assert len(data["sessions"]) <= 3

    def test_evaluation_error_handling(
        self, client: TestClient, auth_headers: dict
    ):
        """Test error handling in evaluation pipeline."""
        # Test with invalid data
        invalid_data = {
            "source_text": "",  # Empty source text
            "candidate_output": "Some output",
            "config": {
                "judge_models": [],  # No judges
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        response = client.post(
            "/api/v1/evaluations",
            json=invalid_data,
            headers=auth_headers
        )

        assert response.status_code == 422  # Validation error

    def test_evaluation_user_isolation(
        self, client: TestClient, db_session: Session
    ):
        """Test that users can only access their own evaluations."""
        # Create two users
        from app.auth import create_user, create_access_token

        user1 = create_user(db_session, "user1@test.com", "user1", "password123")
        user2 = create_user(db_session, "user2@test.com", "user2", "password123")

        token1 = create_access_token({"sub": user1.username})
        token2 = create_access_token({"sub": user2.username})

        headers1 = {"Authorization": f"Bearer {token1}"}
        headers2 = {"Authorization": f"Bearer {token2}"}

        # User 1 creates evaluation
        evaluation_data = {
            "source_text": "User 1 source text",
            "candidate_output": "User 1 output",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        response1 = client.post(
            "/api/v1/evaluations",
            json=evaluation_data,
            headers=headers1
        )

        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # User 2 tries to access User 1's evaluation
        response2 = client.get(
            f"/api/v1/evaluations/{session_id}",
            headers=headers2
        )

        # Should return 404 or 403
        assert response2.status_code in [403, 404]
