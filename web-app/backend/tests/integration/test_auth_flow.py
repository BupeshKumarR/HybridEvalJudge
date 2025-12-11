"""
Integration tests for authentication flow.
Tests the complete authentication lifecycle.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.main import app
from app.models import User
from tests.conftest import db_session, client


class TestAuthenticationFlow:
    """Test complete authentication workflows."""

    def test_complete_registration_and_login_flow(
        self, client: TestClient, db_session: Session
    ):
        """Test user registration followed by login."""
        # Register new user
        register_data = {
            "email": "newuser@test.com",
            "username": "newuser",
            "password": "securepass123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        assert register_response.status_code == 200
        register_result = register_response.json()
        assert "access_token" in register_result
        assert register_result["token_type"] == "bearer"

        # Login with same credentials
        login_data = {
            "username": "newuser",
            "password": "securepass123"
        }

        login_response = client.post(
            "/api/v1/auth/login",
            data=login_data  # OAuth2 form data
        )

        assert login_response.status_code == 200
        login_result = login_response.json()
        assert "access_token" in login_result
        assert login_result["token_type"] == "bearer"

    def test_authenticated_request_flow(
        self, client: TestClient, db_session: Session
    ):
        """Test making authenticated requests."""
        # Register user
        register_data = {
            "email": "authuser@test.com",
            "username": "authuser",
            "password": "password123"
        }

        register_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        token = register_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Make authenticated request
        me_response = client.get(
            "/api/v1/auth/me",
            headers=headers
        )

        assert me_response.status_code == 200
        user_data = me_response.json()
        assert user_data["username"] == "authuser"
        assert user_data["email"] == "authuser@test.com"

    def test_invalid_token_rejection(
        self, client: TestClient
    ):
        """Test that invalid tokens are rejected."""
        invalid_headers = {"Authorization": "Bearer invalid_token_here"}

        response = client.get(
            "/api/v1/auth/me",
            headers=invalid_headers
        )

        assert response.status_code == 401

    def test_missing_token_rejection(
        self, client: TestClient
    ):
        """Test that requests without tokens are rejected."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401

    def test_duplicate_registration_prevention(
        self, client: TestClient, db_session: Session
    ):
        """Test that duplicate usernames/emails are prevented."""
        # Register first user
        register_data = {
            "email": "duplicate@test.com",
            "username": "duplicate",
            "password": "password123"
        }

        first_response = client.post(
            "/api/v1/auth/register",
            json=register_data
        )

        assert first_response.status_code == 200

        # Try to register with same username
        duplicate_username = {
            "email": "different@test.com",
            "username": "duplicate",
            "password": "password123"
        }

        response = client.post(
            "/api/v1/auth/register",
            json=duplicate_username
        )

        assert response.status_code == 400

        # Try to register with same email
        duplicate_email = {
            "email": "duplicate@test.com",
            "username": "different",
            "password": "password123"
        }

        response = client.post(
            "/api/v1/auth/register",
            json=duplicate_email
        )

        assert response.status_code == 400

    def test_password_hashing(
        self, client: TestClient, db_session: Session
    ):
        """Test that passwords are properly hashed."""
        # Register user
        register_data = {
            "email": "hashtest@test.com",
            "username": "hashtest",
            "password": "plaintext_password"
        }

        client.post("/api/v1/auth/register", json=register_data)

        # Check database
        user = db_session.query(User).filter(
            User.username == "hashtest"
        ).first()

        assert user is not None
        # Password should be hashed, not plaintext
        assert user.password_hash != "plaintext_password"
        assert len(user.password_hash) > 20  # Hashed passwords are long

    def test_login_with_wrong_password(
        self, client: TestClient, db_session: Session
    ):
        """Test that login fails with wrong password."""
        # Register user
        register_data = {
            "email": "wrongpass@test.com",
            "username": "wrongpass",
            "password": "correct_password"
        }

        client.post("/api/v1/auth/register", json=register_data)

        # Try to login with wrong password
        login_data = {
            "username": "wrongpass",
            "password": "wrong_password"
        }

        response = client.post(
            "/api/v1/auth/login",
            data=login_data
        )

        assert response.status_code == 401
