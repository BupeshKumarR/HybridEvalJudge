"""Tests for authentication endpoints."""

import pytest
from fastapi import status


def test_register_user(client, test_user_data):
    """Test user registration."""
    response = client.post("/api/v1/auth/register", json=test_user_data)
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["username"] == test_user_data["username"]
    assert data["email"] == test_user_data["email"]
    assert "id" in data
    assert "password" not in data
    assert "password_hash" not in data


def test_register_duplicate_username(client, test_user_data, created_user):
    """Test registration with duplicate username fails."""
    response = client.post("/api/v1/auth/register", json=test_user_data)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    # Check for error message in either 'detail' or 'error' field
    error_msg = data.get("detail", data.get("error", "")).lower()
    assert "already registered" in error_msg


def test_register_duplicate_email(client, test_user_data, created_user):
    """Test registration with duplicate email fails."""
    new_user_data = test_user_data.copy()
    new_user_data["username"] = "differentuser"
    
    response = client.post("/api/v1/auth/register", json=new_user_data)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    # Check for error message in either 'detail' or 'error' field
    error_msg = data.get("detail", data.get("error", "")).lower()
    assert "already registered" in error_msg


def test_register_invalid_email(client, test_user_data):
    """Test registration with invalid email fails."""
    invalid_data = test_user_data.copy()
    invalid_data["email"] = "not-an-email"
    
    response = client.post("/api/v1/auth/register", json=invalid_data)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_register_short_password(client, test_user_data):
    """Test registration with short password fails."""
    invalid_data = test_user_data.copy()
    invalid_data["password"] = "short"
    
    response = client.post("/api/v1/auth/register", json=invalid_data)
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_login_success(client, test_user_data, created_user):
    """Test successful login."""
    response = client.post(
        "/api/v1/auth/login/json",
        json={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_oauth_form(client, test_user_data, created_user):
    """Test login with OAuth2 form data."""
    response = client.post(
        "/api/v1/auth/login",
        data={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
    )
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client, test_user_data, created_user):
    """Test login with wrong password fails."""
    response = client.post(
        "/api/v1/auth/login/json",
        json={
            "username": test_user_data["username"],
            "password": "wrongpassword"
        }
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_login_nonexistent_user(client):
    """Test login with nonexistent user fails."""
    response = client.post(
        "/api/v1/auth/login/json",
        json={
            "username": "nonexistent",
            "password": "password123"
        }
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_current_user(client, auth_headers):
    """Test getting current user information."""
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "username" in data
    assert "email" in data
    assert "id" in data


def test_get_current_user_no_token(client):
    """Test getting current user without token fails."""
    response = client.get("/api/v1/auth/me")
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_get_current_user_invalid_token(client):
    """Test getting current user with invalid token fails."""
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid_token"}
    )
    
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_logout(client, auth_headers):
    """Test logout endpoint."""
    response = client.post("/api/v1/auth/logout", headers=auth_headers)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "message" in data


def test_refresh_token(client, auth_headers):
    """Test token refresh."""
    response = client.post("/api/v1/auth/refresh", headers=auth_headers)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
