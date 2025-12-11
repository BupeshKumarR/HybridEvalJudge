"""
Tests for security features: rate limiting, CSRF protection, input sanitization, and security headers.
"""
import pytest
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock
import time

from app.security import (
    RateLimiter,
    CSRFProtection,
    InputSanitizer,
    check_rate_limit,
    rate_limiter,
    csrf_protection,
    input_sanitizer
)


class TestRateLimiter:
    """Tests for rate limiting functionality."""
    
    def test_rate_limiter_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter()
        key = "test_user_1"
        
        # Make requests within limit
        for _ in range(5):
            is_limited, retry_after = limiter.is_rate_limited(key, max_requests=10, window_seconds=60)
            assert not is_limited
            assert retry_after is None
    
    def test_rate_limiter_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        limiter = RateLimiter()
        key = "test_user_2"
        
        # Make requests up to limit
        for _ in range(10):
            is_limited, retry_after = limiter.is_rate_limited(key, max_requests=10, window_seconds=60)
            assert not is_limited
        
        # Next request should be limited
        is_limited, retry_after = limiter.is_rate_limited(key, max_requests=10, window_seconds=60)
        assert is_limited
        assert retry_after is not None
        assert retry_after > 0
    
    def test_rate_limiter_sliding_window(self):
        """Test that rate limiter uses sliding window correctly."""
        limiter = RateLimiter()
        key = "test_user_3"
        
        # Make requests
        for _ in range(5):
            limiter.is_rate_limited(key, max_requests=5, window_seconds=1)
        
        # Wait for window to pass
        time.sleep(1.1)
        
        # Should be able to make requests again
        is_limited, retry_after = limiter.is_rate_limited(key, max_requests=5, window_seconds=1)
        assert not is_limited
    
    def test_rate_limiter_cleanup(self):
        """Test that old entries are cleaned up."""
        limiter = RateLimiter()
        key = "test_user_4"
        
        # Add some requests
        limiter.is_rate_limited(key, max_requests=10, window_seconds=60)
        
        # Force cleanup
        limiter.last_cleanup = 0
        limiter._cleanup_old_entries()
        
        # Should still work
        is_limited, retry_after = limiter.is_rate_limited(key, max_requests=10, window_seconds=60)
        assert not is_limited


class TestCSRFProtection:
    """Tests for CSRF protection."""
    
    def test_generate_token(self):
        """Test CSRF token generation."""
        csrf = CSRFProtection()
        session_id = "test_session_123"
        
        token = csrf.generate_token(session_id)
        assert token is not None
        assert len(token) == 64  # SHA256 hex digest length
    
    def test_validate_token_success(self):
        """Test successful CSRF token validation."""
        csrf = CSRFProtection()
        session_id = "test_session_123"
        
        token = csrf.generate_token(session_id)
        is_valid = csrf.validate_token(token, session_id)
        assert is_valid
    
    def test_validate_token_failure(self):
        """Test failed CSRF token validation."""
        csrf = CSRFProtection()
        session_id = "test_session_123"
        
        token = csrf.generate_token(session_id)
        is_valid = csrf.validate_token(token, "different_session")
        assert not is_valid
    
    def test_validate_token_invalid_format(self):
        """Test validation with invalid token format."""
        csrf = CSRFProtection()
        session_id = "test_session_123"
        
        is_valid = csrf.validate_token("invalid_token", session_id)
        assert not is_valid


class TestInputSanitizer:
    """Tests for input sanitization."""
    
    def test_sanitize_html(self):
        """Test HTML sanitization."""
        text = "<script>alert('xss')</script>"
        sanitized = InputSanitizer.sanitize_html(text)
        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized
    
    def test_detect_sql_injection(self):
        """Test SQL injection detection."""
        # Should detect SQL injection
        assert InputSanitizer.detect_sql_injection("SELECT * FROM users WHERE id = 1 UNION SELECT * FROM passwords")
        assert InputSanitizer.detect_sql_injection("DROP TABLE users")
        assert InputSanitizer.detect_sql_injection("'; DELETE FROM users--")
        
        # Should not detect in normal text
        assert not InputSanitizer.detect_sql_injection("This is normal text")
        assert not InputSanitizer.detect_sql_injection("I want to select a book from the library")
    
    def test_detect_xss(self):
        """Test XSS detection."""
        # Should detect XSS
        assert InputSanitizer.detect_xss("<script>alert('xss')</script>")
        assert InputSanitizer.detect_xss("<img src=x onerror=alert('xss')>")
        assert InputSanitizer.detect_xss("javascript:alert('xss')")
        assert InputSanitizer.detect_xss("<iframe src='evil.com'></iframe>")
        
        # Should not detect in normal text
        assert not InputSanitizer.detect_xss("This is normal text")
        assert not InputSanitizer.detect_xss("<p>This is a paragraph</p>")
    
    def test_sanitize_text_success(self):
        """Test successful text sanitization."""
        text = "This is normal text with <b>HTML</b>"
        sanitized = InputSanitizer.sanitize_text(text)
        assert sanitized is not None
        assert "&lt;b&gt;" in sanitized
    
    def test_sanitize_text_max_length(self):
        """Test text sanitization with max length."""
        text = "a" * 10001
        
        with pytest.raises(HTTPException) as exc_info:
            InputSanitizer.sanitize_text(text, max_length=10000)
        
        assert exc_info.value.status_code == 400
        assert "exceeds maximum length" in exc_info.value.detail
    
    def test_sanitize_text_sql_injection(self):
        """Test that SQL injection is detected and rejected."""
        text = "'; DROP TABLE users--"
        
        with pytest.raises(HTTPException) as exc_info:
            InputSanitizer.sanitize_text(text)
        
        assert exc_info.value.status_code == 400
        assert "SQL injection" in exc_info.value.detail
    
    def test_sanitize_text_xss(self):
        """Test that XSS is detected and rejected."""
        text = "<script>alert('xss')</script>"
        
        with pytest.raises(HTTPException) as exc_info:
            InputSanitizer.sanitize_text(text)
        
        assert exc_info.value.status_code == 400
        assert "XSS" in exc_info.value.detail


class TestSecurityHeaders:
    """Tests for security headers middleware."""
    
    def test_security_headers_present(self, client: TestClient):
        """Test that security headers are present in responses."""
        response = client.get("/health")
        
        # Check for security headers
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Permissions-Policy" in response.headers
