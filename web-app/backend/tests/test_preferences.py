"""
Tests for user preferences API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import User, UserPreference


def test_get_preferences_creates_defaults(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test that getting preferences creates defaults if they don't exist."""
    # Ensure no preferences exist
    db_session.query(UserPreference).filter(
        UserPreference.user_id == created_user.id
    ).delete()
    db_session.commit()
    
    # Get preferences
    response = client.get("/api/v1/preferences", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["user_id"] == str(created_user.id)
    assert data["default_judge_models"] == ["gpt-4", "claude-3"]
    assert data["default_retrieval_enabled"] is True
    assert data["default_aggregation_strategy"] == "weighted_average"
    assert data["theme"] == "light"
    assert data["notifications_enabled"] is True
    
    # Verify preferences were created in database
    prefs = db.query(UserPreference).filter(
        UserPreference.user_id == created_user.id
    ).first()
    assert prefs is not None


def test_get_existing_preferences(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test getting existing preferences."""
    # Create preferences
    prefs = UserPreference(
        user_id=created_user.id,
        default_judge_models=["gpt-3.5-turbo"],
        default_retrieval_enabled=False,
        default_aggregation_strategy="median",
        theme="dark",
        notifications_enabled=False,
    )
    db.add(prefs)
    db.commit()
    
    # Get preferences
    response = client.get("/api/v1/preferences", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["default_judge_models"] == ["gpt-3.5-turbo"]
    assert data["default_retrieval_enabled"] is False
    assert data["default_aggregation_strategy"] == "median"
    assert data["theme"] == "dark"
    assert data["notifications_enabled"] is False


def test_update_preferences(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test updating user preferences."""
    # Create initial preferences
    prefs = UserPreference(
        user_id=created_user.id,
        default_judge_models=["gpt-4"],
        default_retrieval_enabled=True,
        default_aggregation_strategy="weighted_average",
    )
    db.add(prefs)
    db.commit()
    
    # Update preferences
    update_data = {
        "default_judge_models": ["claude-3", "gemini-pro"],
        "default_retrieval_enabled": False,
        "default_aggregation_strategy": "majority_vote",
    }
    
    response = client.put(
        "/api/v1/preferences",
        json=update_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["default_judge_models"] == ["claude-3", "gemini-pro"]
    assert data["default_retrieval_enabled"] is False
    assert data["default_aggregation_strategy"] == "majority_vote"
    
    # Verify in database
    db.refresh(prefs)
    assert prefs.default_judge_models == ["claude-3", "gemini-pro"]
    assert prefs.default_retrieval_enabled is False
    assert prefs.default_aggregation_strategy == "majority_vote"


def test_update_preferences_creates_if_not_exist(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test that updating preferences creates them if they don't exist."""
    # Ensure no preferences exist
    db.query(UserPreference).filter(
        UserPreference.user_id == created_user.id
    ).delete()
    db.commit()
    
    # Update preferences
    update_data = {
        "default_judge_models": ["claude-3"],
        "default_retrieval_enabled": False,
    }
    
    response = client.put(
        "/api/v1/preferences",
        json=update_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["default_judge_models"] == ["claude-3"]
    assert data["default_retrieval_enabled"] is False
    
    # Verify preferences were created
    prefs = db.query(UserPreference).filter(
        UserPreference.user_id == created_user.id
    ).first()
    assert prefs is not None


def test_update_partial_preferences(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test updating only some preference fields."""
    # Create initial preferences
    prefs = UserPreference(
        user_id=created_user.id,
        default_judge_models=["gpt-4"],
        default_retrieval_enabled=True,
        default_aggregation_strategy="weighted_average",
        theme="light",
    )
    db.add(prefs)
    db.commit()
    
    # Update only judge models
    update_data = {
        "default_judge_models": ["claude-3", "gpt-4"],
    }
    
    response = client.put(
        "/api/v1/preferences",
        json=update_data,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Updated field
    assert data["default_judge_models"] == ["claude-3", "gpt-4"]
    
    # Unchanged fields
    assert data["default_retrieval_enabled"] is True
    assert data["default_aggregation_strategy"] == "weighted_average"
    assert data["theme"] == "light"


def test_reset_preferences(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test resetting preferences to defaults."""
    # Create custom preferences
    prefs = UserPreference(
        user_id=created_user.id,
        default_judge_models=["custom-model"],
        default_retrieval_enabled=False,
        default_aggregation_strategy="median",
        theme="dark",
        notifications_enabled=False,
    )
    db.add(prefs)
    db.commit()
    
    # Reset preferences
    response = client.post("/api/v1/preferences/reset", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check defaults
    assert data["default_judge_models"] == ["gpt-4", "claude-3"]
    assert data["default_retrieval_enabled"] is True
    assert data["default_aggregation_strategy"] == "weighted_average"
    assert data["theme"] == "light"
    assert data["notifications_enabled"] is True
    
    # Verify in database
    db.refresh(prefs)
    assert prefs.default_judge_models == ["gpt-4", "claude-3"]


def test_delete_preferences(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test deleting user preferences."""
    # Create preferences
    prefs = UserPreference(
        user_id=created_user.id,
        default_judge_models=["gpt-4"],
    )
    db.add(prefs)
    db.commit()
    
    # Delete preferences
    response = client.delete("/api/v1/preferences", headers=auth_headers)
    
    assert response.status_code == 200
    assert response.json()["message"] == "Preferences deleted successfully"
    
    # Verify deletion
    prefs = db.query(UserPreference).filter(
        UserPreference.user_id == created_user.id
    ).first()
    assert prefs is None


def test_preferences_require_authentication(client: TestClient):
    """Test that preferences endpoints require authentication."""
    # Get without auth
    response = client.get("/api/v1/preferences")
    assert response.status_code == 401
    
    # Update without auth
    response = client.put("/api/v1/preferences", json={})
    assert response.status_code == 401
    
    # Reset without auth
    response = client.post("/api/v1/preferences/reset")
    assert response.status_code == 401
    
    # Delete without auth
    response = client.delete("/api/v1/preferences")
    assert response.status_code == 401


def test_preferences_isolated_by_user(
    client: TestClient,
    created_user: User,
    auth_headers: dict,
    db_session: Session
):
    """Test that users can only access their own preferences."""
    # Create another user
    from app.auth import get_password_hash
    
    other_user = User(
        username="otheruser",
        email="other@example.com",
        password_hash=get_password_hash("password123")
    )
    db.add(other_user)
    db.commit()
    
    # Create preferences for other user
    other_prefs = UserPreference(
        user_id=other_user.id,
        default_judge_models=["other-model"],
    )
    db.add(other_prefs)
    db.commit()
    
    # Get preferences as test_user
    response = client.get("/api/v1/preferences", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    
    # Should get test_user's preferences (or defaults), not other_user's
    assert data["user_id"] == str(created_user.id)
    assert data["default_judge_models"] != ["other-model"]
