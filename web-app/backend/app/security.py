"""
Security utilities for rate limiting, CSRF protection, input sanitization, and security headers.
"""
from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional, Callable
import time
import hashlib
import secrets
import re
import html
from collections import defaultdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Rate limiter with per-user and per-IP tracking.
    Uses sliding window algorithm for accurate rate limiting.
    """
    
    def __init__(self):
        # Store: {key: [(timestamp, count), ...]}
        self.requests: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 60  # Cleanup old entries every 60 seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Remove entries older than 1 hour to prevent memory bloat."""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
        
        current_time = time.time()
        cutoff_time = current_time - 3600  # 1 hour
        
        for key in list(self.requests.keys()):
            self.requests[key] = [
                (ts, count) for ts, count in self.requests[key]
                if ts > cutoff_time
            ]
            if not self.requests[key]:
                del self.requests[key]
        
        self.last_cleanup = current_time
    
    def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """
        Check if a key has exceeded the rate limit.
        
        Args:
            key: Unique identifier (user_id, IP address, etc.)
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_limited, retry_after_seconds)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter requests within the window
        recent_requests = [
            (ts, count) for ts, count in self.requests[key]
            if ts > cutoff_time
        ]
        
        # Count total requests in window
        total_requests = sum(count for _, count in recent_requests)
        
        if total_requests >= max_requests:
            # Calculate retry after time
            oldest_request = min(ts for ts, _ in recent_requests) if recent_requests else current_time
            retry_after = int(window_seconds - (current_time - oldest_request)) + 1
            return True, retry_after
        
        # Add current request
        self.requests[key] = recent_requests + [(current_time, 1)]
        return False, None


# Global rate limiter instance
rate_limiter = RateLimiter()


# Rate limit configurations
RATE_LIMITS = {
    "default": {"max_requests": 100, "window_seconds": 60},  # 100 req/min
    "auth": {"max_requests": 5, "window_seconds": 300},  # 5 req/5min for login
    "evaluation": {"max_requests": 20, "window_seconds": 60},  # 20 eval/min
    "export": {"max_requests": 10, "window_seconds": 60},  # 10 exports/min
}



def get_rate_limit_key(request: Request, user_id: Optional[str] = None) -> str:
    """
    Generate a rate limit key based on user ID or IP address.
    
    Args:
        request: FastAPI request object
        user_id: Optional user ID for authenticated requests
        
    Returns:
        Rate limit key string
    """
    if user_id:
        return f"user:{user_id}"
    
    # Get client IP (handle proxy headers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    return f"ip:{client_ip}"


async def check_rate_limit(
    request: Request,
    limit_type: str = "default",
    user_id: Optional[str] = None
):
    """
    Dependency to check rate limits for requests.
    
    Args:
        request: FastAPI request object
        limit_type: Type of rate limit to apply
        user_id: Optional user ID for authenticated requests
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    config = RATE_LIMITS.get(limit_type, RATE_LIMITS["default"])
    key = get_rate_limit_key(request, user_id)
    
    is_limited, retry_after = rate_limiter.is_rate_limited(
        key,
        config["max_requests"],
        config["window_seconds"]
    )
    
    if is_limited:
        logger.warning(f"Rate limit exceeded for {key} on {limit_type}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )



# ============================================================================
# CSRF Protection
# ============================================================================

class CSRFProtection:
    """
    CSRF token generation and validation.
    Uses double-submit cookie pattern.
    """
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
    
    def generate_token(self, session_id: str) -> str:
        """
        Generate a CSRF token for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            CSRF token string
        """
        # Create token from session ID and secret
        token_data = f"{session_id}:{self.secret_key}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        return token
    
    def validate_token(self, token: str, session_id: str) -> bool:
        """
        Validate a CSRF token.
        
        Args:
            token: CSRF token to validate
            session_id: Session identifier
            
        Returns:
            True if valid, False otherwise
        """
        expected_token = self.generate_token(session_id)
        return secrets.compare_digest(token, expected_token)


# Global CSRF protection instance
csrf_protection = CSRFProtection()


async def verify_csrf_token(request: Request):
    """
    Dependency to verify CSRF token for state-changing operations.
    
    Args:
        request: FastAPI request object
        
    Raises:
        HTTPException: If CSRF token is invalid or missing
    """
    # Skip CSRF check for safe methods
    if request.method in ["GET", "HEAD", "OPTIONS"]:
        return
    
    # Get token from header
    csrf_token = request.headers.get("X-CSRF-Token")
    if not csrf_token:
        logger.warning(f"Missing CSRF token for {request.method} {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF token missing"
        )
    
    # Get session ID from request state (set by auth middleware)
    session_id = getattr(request.state, "session_id", None)
    if not session_id:
        # Use request ID as fallback
        session_id = getattr(request.state, "request_id", "unknown")
    
    # Validate token
    if not csrf_protection.validate_token(csrf_token, session_id):
        logger.warning(f"Invalid CSRF token for {request.method} {request.url.path}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token"
        )



# ============================================================================
# Input Sanitization
# ============================================================================

class InputSanitizer:
    """
    Input sanitization utilities to prevent XSS and injection attacks.
    """
    
    # Patterns for detecting potentially malicious input
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bEXEC\b|\bEXECUTE\b)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
    ]
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """
        Escape HTML special characters to prevent XSS.
        
        Args:
            text: Input text
            
        Returns:
            Sanitized text with HTML entities escaped
        """
        if not text:
            return text
        return html.escape(text)
    
    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """
        Detect potential SQL injection attempts.
        
        Args:
            text: Input text to check
            
        Returns:
            True if potential SQL injection detected
        """
        if not text:
            return False
        
        text_upper = text.upper()
        for pattern in InputSanitizer.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def detect_xss(text: str) -> bool:
        """
        Detect potential XSS attempts.
        
        Args:
            text: Input text to check
            
        Returns:
            True if potential XSS detected
        """
        if not text:
            return False
        
        for pattern in InputSanitizer.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """
        Sanitize text input by removing potentially dangerous content.
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            HTTPException: If malicious content detected
        """
        if not text:
            return text
        
        # Check length
        if len(text) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input exceeds maximum length of {max_length} characters"
            )
        
        # Check for SQL injection
        if InputSanitizer.detect_sql_injection(text):
            logger.warning(f"Potential SQL injection detected: {text[:100]}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input: potential SQL injection detected"
            )
        
        # Check for XSS
        if InputSanitizer.detect_xss(text):
            logger.warning(f"Potential XSS detected: {text[:100]}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input: potential XSS detected"
            )
        
        # Escape HTML entities
        return InputSanitizer.sanitize_html(text)


# Global sanitizer instance
input_sanitizer = InputSanitizer()



# ============================================================================
# Security Headers Middleware
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to response.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain
            
        Returns:
            Response with security headers added
        """
        response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Enable XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Strict Transport Security (HTTPS only)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Adjust based on needs
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' ws: wss:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy (formerly Feature Policy)
        permissions = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        return response
