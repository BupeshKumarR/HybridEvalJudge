#!/usr/bin/env python3
"""
Script to verify security features are properly implemented.
"""
import sys

def check_imports():
    """Verify all security modules can be imported."""
    print("Checking security module imports...")
    try:
        from app.security import (
            RateLimiter,
            CSRFProtection,
            InputSanitizer,
            SecurityHeadersMiddleware,
            rate_limiter,
            csrf_protection,
            input_sanitizer,
            check_rate_limit
        )
        print("✓ Security module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import security modules: {e}")
        return False


def check_audit_log():
    """Verify audit log module can be imported."""
    print("\nChecking audit log module imports...")
    try:
        from app.audit_log import (
            AuditLog,
            AuditLogger,
            AuditEventType,
            AuditSeverity,
            AuditLogRetentionPolicy,
            audit_logger
        )
        print("✓ Audit log module imports successful")
        return True
    except ImportError as e:
        print(f"✗ Failed to import audit log modules: {e}")
        return False


def check_rate_limiter():
    """Test rate limiter functionality."""
    print("\nTesting rate limiter...")
    try:
        from app.security import RateLimiter
        
        limiter = RateLimiter()
        key = "test_key"
        
        # Test within limit
        is_limited, _ = limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
        if is_limited:
            print("✗ Rate limiter incorrectly limited first request")
            return False
        
        # Test over limit
        for _ in range(5):
            limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
        
        is_limited, retry_after = limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
        if not is_limited:
            print("✗ Rate limiter failed to limit after threshold")
            return False
        
        if retry_after is None or retry_after <= 0:
            print("✗ Rate limiter did not return valid retry_after")
            return False
        
        print("✓ Rate limiter working correctly")
        return True
    except Exception as e:
        print(f"✗ Rate limiter test failed: {e}")
        return False


def check_csrf():
    """Test CSRF protection functionality."""
    print("\nTesting CSRF protection...")
    try:
        from app.security import CSRFProtection
        
        csrf = CSRFProtection()
        session_id = "test_session"
        
        # Test token generation
        token = csrf.generate_token(session_id)
        if not token or len(token) != 64:
            print("✗ CSRF token generation failed")
            return False
        
        # Test valid token
        if not csrf.validate_token(token, session_id):
            print("✗ CSRF validation failed for valid token")
            return False
        
        # Test invalid token
        if csrf.validate_token(token, "different_session"):
            print("✗ CSRF validation passed for invalid token")
            return False
        
        print("✓ CSRF protection working correctly")
        return True
    except Exception as e:
        print(f"✗ CSRF test failed: {e}")
        return False


def check_input_sanitizer():
    """Test input sanitization functionality."""
    print("\nTesting input sanitizer...")
    try:
        from app.security import InputSanitizer
        
        # Test HTML sanitization
        html_text = "<script>alert('xss')</script>"
        sanitized = InputSanitizer.sanitize_html(html_text)
        if "<script>" in sanitized:
            print("✗ HTML sanitization failed")
            return False
        
        # Test SQL injection detection
        if not InputSanitizer.detect_sql_injection("SELECT * FROM users UNION SELECT * FROM passwords"):
            print("✗ SQL injection detection failed")
            return False
        
        # Test XSS detection
        if not InputSanitizer.detect_xss("<script>alert('xss')</script>"):
            print("✗ XSS detection failed")
            return False
        
        # Test normal text passes
        if InputSanitizer.detect_sql_injection("This is normal text"):
            print("✗ SQL injection false positive")
            return False
        
        if InputSanitizer.detect_xss("This is normal text"):
            print("✗ XSS false positive")
            return False
        
        print("✓ Input sanitizer working correctly")
        return True
    except Exception as e:
        print(f"✗ Input sanitizer test failed: {e}")
        return False


def check_integration():
    """Check that security features are integrated into routers."""
    print("\nChecking router integration...")
    try:
        from app.routers import auth, evaluations, preferences
        
        # Check imports in auth router
        auth_source = open("app/routers/auth.py").read()
        if "check_rate_limit" not in auth_source:
            print("✗ Rate limiting not integrated in auth router")
            return False
        if "audit_logger" not in auth_source:
            print("✗ Audit logging not integrated in auth router")
            return False
        
        # Check imports in evaluations router
        eval_source = open("app/routers/evaluations.py").read()
        if "check_rate_limit" not in eval_source:
            print("✗ Rate limiting not integrated in evaluations router")
            return False
        if "audit_logger" not in eval_source:
            print("✗ Audit logging not integrated in evaluations router")
            return False
        
        # Check imports in preferences router
        pref_source = open("app/routers/preferences.py").read()
        if "audit_logger" not in pref_source:
            print("✗ Audit logging not integrated in preferences router")
            return False
        
        print("✓ Security features integrated into routers")
        return True
    except Exception as e:
        print(f"✗ Integration check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Security Features Verification")
    print("=" * 60)
    
    checks = [
        check_imports,
        check_audit_log,
        check_rate_limiter,
        check_csrf,
        check_input_sanitizer,
        check_integration
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All security features verified successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
