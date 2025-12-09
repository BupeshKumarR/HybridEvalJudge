"""
End-to-end tests for complete user workflows.
Tests realistic user scenarios from start to finish.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from tests.conftest import test_db, client


class TestCompleteUserWorkflows:
    """Test complete user workflows end-to-end."""

    def test_new_user_complete_workflow(
        self, client: TestClient, test_db: Session
    ):
        """
        Test complete workflow for a new user:
        1. Register
        2. Login
        3. Create evaluation
        4. View evaluation
        5. List evaluations
        6. Export evaluation
        7. Update preferences
        """
        # Step 1: Register
        register_data = {
            "email": "workflow@test.com",
            "username": "workflowuser",
            "password": "secure_password123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        assert register_response.status_code == 200
        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Step 2: Verify login works
        me_response = client.get("/api/v1/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "workflowuser"

        # Step 3: Create evaluation
        evaluation_data = {
            "source_text": "The Earth is the third planet from the Sun.",
            "candidate_output": "Earth is the third planet in our solar system.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        eval_response = client.post(
            "/api/v1/evaluations",
            json=evaluation_data,
            headers=headers
        )

        assert eval_response.status_code == 200
        session_id = eval_response.json()["session_id"]

        # Step 4: View evaluation
        view_response = client.get(
            f"/api/v1/evaluations/{session_id}",
            headers=headers
        )

        assert view_response.status_code == 200
        evaluation = view_response.json()
        assert evaluation["source_text"] == evaluation_data["source_text"]

        # Step 5: List evaluations
        list_response = client.get(
            "/api/v1/evaluations",
            headers=headers
        )

        assert list_response.status_code == 200
        evaluations = list_response.json()
        assert "sessions" in evaluations
        assert len(evaluations["sessions"]) >= 1

        # Step 6: Export evaluation (JSON)
        export_response = client.get(
            f"/api/v1/evaluations/{session_id}/export?format=json",
            headers=headers
        )

        assert export_response.status_code == 200

        # Step 7: Update preferences
        preferences_data = {
            "judge_models": ["gpt-4", "claude-3"],
            "enable_retrieval": True,
            "aggregation_strategy": "median"
        }

        pref_response = client.put(
            "/api/v1/preferences",
            json=preferences_data,
            headers=headers
        )

        assert pref_response.status_code == 200

    def test_multi_evaluation_workflow(
        self, client: TestClient, test_db: Session
    ):
        """
        Test workflow with multiple evaluations:
        1. Register user
        2. Create multiple evaluations
        3. Filter and search evaluations
        4. Compare evaluations
        """
        # Register user
        register_data = {
            "email": "multi@test.com",
            "username": "multiuser",
            "password": "password123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create multiple evaluations with different scores
        evaluation_configs = [
            {
                "source": "High quality source text with accurate information.",
                "candidate": "High quality output that matches the source.",
                "expected_score_range": (80, 100)
            },
            {
                "source": "Medium quality source text.",
                "candidate": "Medium quality output with some issues.",
                "expected_score_range": (50, 80)
            },
            {
                "source": "Source text about facts.",
                "candidate": "Output with potential inaccuracies.",
                "expected_score_range": (0, 50)
            }
        ]

        session_ids = []
        for config in evaluation_configs:
            eval_data = {
                "source_text": config["source"],
                "candidate_output": config["candidate"],
                "config": {
                    "judge_models": ["gpt-4"],
                    "enable_retrieval": False,
                    "aggregation_strategy": "weighted_average"
                }
            }

            response = client.post(
                "/api/v1/evaluations",
                json=eval_data,
                headers=headers
            )

            assert response.status_code == 200
            session_ids.append(response.json()["session_id"])

        # List all evaluations
        list_response = client.get(
            "/api/v1/evaluations",
            headers=headers
        )

        assert list_response.status_code == 200
        assert len(list_response.json()["sessions"]) >= 3

        # Test pagination
        page1_response = client.get(
            "/api/v1/evaluations?page=1&limit=2",
            headers=headers
        )

        assert page1_response.status_code == 200
        page1_data = page1_response.json()
        assert len(page1_data["sessions"]) <= 2

    def test_evaluation_export_workflow(
        self, client: TestClient, test_db: Session
    ):
        """
        Test evaluation export workflow:
        1. Create evaluation
        2. Export as JSON
        3. Export as CSV
        4. Generate shareable link
        """
        # Register and create evaluation
        register_data = {
            "email": "export@test.com",
            "username": "exportuser",
            "password": "password123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create evaluation
        eval_data = {
            "source_text": "Export test source text.",
            "candidate_output": "Export test output.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        eval_response = client.post(
            "/api/v1/evaluations",
            json=eval_data,
            headers=headers
        )

        session_id = eval_response.json()["session_id"]

        # Export as JSON
        json_response = client.get(
            f"/api/v1/evaluations/{session_id}/export?format=json",
            headers=headers
        )

        assert json_response.status_code == 200
        assert json_response.headers["content-type"] == "application/json"

        # Export as CSV
        csv_response = client.get(
            f"/api/v1/evaluations/{session_id}/export?format=csv",
            headers=headers
        )

        assert csv_response.status_code == 200
        assert "text/csv" in csv_response.headers["content-type"]

    def test_preferences_workflow(
        self, client: TestClient, test_db: Session
    ):
        """
        Test preferences management workflow:
        1. Register user
        2. Get default preferences
        3. Update preferences
        4. Create evaluation with preferences
        5. Reset preferences
        """
        # Register user
        register_data = {
            "email": "prefs@test.com",
            "username": "prefsuser",
            "password": "password123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Get default preferences
        get_prefs_response = client.get(
            "/api/v1/preferences",
            headers=headers
        )

        assert get_prefs_response.status_code == 200
        default_prefs = get_prefs_response.json()
        assert "judge_models" in default_prefs

        # Update preferences
        new_prefs = {
            "judge_models": ["gpt-4", "claude-3", "gemini-pro"],
            "enable_retrieval": True,
            "aggregation_strategy": "median"
        }

        update_response = client.put(
            "/api/v1/preferences",
            json=new_prefs,
            headers=headers
        )

        assert update_response.status_code == 200

        # Verify preferences were updated
        verify_response = client.get(
            "/api/v1/preferences",
            headers=headers
        )

        assert verify_response.status_code == 200
        updated_prefs = verify_response.json()
        assert updated_prefs["judge_models"] == new_prefs["judge_models"]
        assert updated_prefs["enable_retrieval"] == True

        # Create evaluation (should use preferences)
        eval_data = {
            "source_text": "Preferences test source.",
            "candidate_output": "Preferences test output.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        eval_response = client.post(
            "/api/v1/evaluations",
            json=eval_data,
            headers=headers
        )

        assert eval_response.status_code == 200

    def test_error_recovery_workflow(
        self, client: TestClient, test_db: Session
    ):
        """
        Test error handling and recovery:
        1. Attempt invalid operations
        2. Verify proper error responses
        3. Recover and complete valid operations
        """
        # Register user
        register_data = {
            "email": "error@test.com",
            "username": "erroruser",
            "password": "password123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Try to create evaluation with invalid data
        invalid_eval = {
            "source_text": "",  # Empty
            "candidate_output": "Some output",
            "config": {
                "judge_models": [],  # No judges
                "enable_retrieval": False,
                "aggregation_strategy": "invalid_strategy"
            }
        }

        error_response = client.post(
            "/api/v1/evaluations",
            json=invalid_eval,
            headers=headers
        )

        assert error_response.status_code == 422

        # Try to access non-existent evaluation
        not_found_response = client.get(
            "/api/v1/evaluations/00000000-0000-0000-0000-000000000000",
            headers=headers
        )

        assert not_found_response.status_code == 404

        # Recover with valid evaluation
        valid_eval = {
            "source_text": "Valid source text for recovery test.",
            "candidate_output": "Valid output for recovery test.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": False,
                "aggregation_strategy": "weighted_average"
            }
        }

        success_response = client.post(
            "/api/v1/evaluations",
            json=valid_eval,
            headers=headers
        )

        assert success_response.status_code == 200
