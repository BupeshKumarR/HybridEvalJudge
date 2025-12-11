# Security Features

This document describes the security features implemented in the LLM Judge Auditor web application.

## Overview

The application implements multiple layers of security to protect against common web vulnerabilities and ensure data integrity.

## Security Features

### 1. Rate Limiting

**Purpose:** Prevent abuse and DoS attacks by limiting the number of requests per user/IP.

**Implementation:**
- Per-user and per-IP rate limiting using sliding window algorithm
- Different limits for different endpoint types:
  - Authentication endpoints: 5 requests per 5 minutes
  - Evaluation endpoints: 20 requests per minute
  - Export endpoints: 10 requests per minute
  - Default: 100 requests per minute

**Usage:**
```python
from app.security import check_rate_limit

@router.post("/endpoint")
async def endpoint(
    request: Request,
    _rate_limit: None = Depends(lambda r: check_rate_limit(r, "evaluation"))
):
    # Endpoint logic
    pass
```

**Configuration:**
Rate limits can be adjusted in `app/security.py`:
```python
RATE_LIMITS = {
    "default": {"max_requests": 100, "window_seconds": 60},
    "auth": {"max_requests": 5, "window_seconds": 300},
    "evaluation": {"max_requests": 20, "window_seconds": 60},
    "export": {"max_requests": 10, "window_seconds": 60},
}
```

### 2. CSRF Protection

**Purpose:** Prevent Cross-Site Request Forgery attacks on state-changing operations.

**Implementation:**
- Double-submit cookie pattern
- Token generation based on session ID and secret key
- Automatic validation for POST, PUT, DELETE, PATCH requests

**Usage:**
```python
from app.security import verify_csrf_token

@router.post("/endpoint")
async def endpoint(
    request: Request,
    _csrf: None = Depends(verify_csrf_token)
):
    # Endpoint logic
    pass
```

**Client-side:**
Include CSRF token in request headers:
```javascript
headers: {
    'X-CSRF-Token': csrfToken
}
```

### 3. Input Sanitization

**Purpose:** Prevent XSS and SQL injection attacks by validating and sanitizing user input.

**Implementation:**
- HTML entity escaping
- SQL injection pattern detection
- XSS pattern detection
- Maximum length validation

**Usage:**
```python
from app.security import input_sanitizer

# Sanitize text input
sanitized_text = input_sanitizer.sanitize_text(user_input, max_length=10000)
```

**Detected Patterns:**
- SQL injection: UNION SELECT, DROP TABLE, DELETE FROM, etc.
- XSS: `<script>`, `javascript:`, `onerror=`, `<iframe>`, etc.

### 4. Security Headers

**Purpose:** Protect against various client-side attacks through HTTP security headers.

**Implementation:**
Middleware automatically adds the following headers to all responses:

- **X-Frame-Options:** DENY (prevents clickjacking)
- **X-Content-Type-Options:** nosniff (prevents MIME sniffing)
- **X-XSS-Protection:** 1; mode=block (enables XSS filter)
- **Strict-Transport-Security:** max-age=31536000; includeSubDomains (enforces HTTPS)
- **Content-Security-Policy:** Restricts resource loading
- **Referrer-Policy:** strict-origin-when-cross-origin
- **Permissions-Policy:** Disables unnecessary browser features

**Configuration:**
Headers are configured in `app/security.py` in the `SecurityHeadersMiddleware` class.

### 5. Audit Logging

**Purpose:** Track security-critical events and user actions for compliance and forensics.

**Implementation:**
- Database-backed audit log with retention policies
- Automatic logging of authentication, evaluation, and configuration events
- Severity levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Retention periods based on severity (7 days to 2 years)

**Event Types:**
- Authentication: login, logout, registration, password changes
- Evaluations: create, complete, error, export
- Configuration: preference updates, config changes
- Security: rate limit violations, CSRF violations, invalid input

**Usage:**
```python
from app.audit_log import audit_logger, AuditEventType

# Log authentication event
audit_logger.log_authentication_event(
    event_type=AuditEventType.LOGIN_SUCCESS,
    username=username,
    ip_address=client_ip,
    user_agent=user_agent,
    success=True
)

# Log evaluation event
audit_logger.log_evaluation_event(
    event_type=AuditEventType.EVALUATION_CREATE,
    user_id=str(user_id),
    username=username,
    session_id=str(session_id),
    ip_address=client_ip
)

# Log configuration change
audit_logger.log_config_change(
    user_id=str(user_id),
    username=username,
    config_type="judge_models",
    old_value=old_models,
    new_value=new_models,
    ip_address=client_ip
)
```

**Querying Audit Logs:**
```python
from app.audit_log import AuditLogRetentionPolicy

# Get recent logs
logs = AuditLogRetentionPolicy.get_audit_logs(
    event_type=AuditEventType.LOGIN_FAILURE,
    start_date=datetime.utcnow() - timedelta(days=7),
    limit=100
)
```

**Retention Policy:**
- DEBUG: 7 days
- INFO: 90 days
- WARNING: 180 days
- ERROR: 365 days
- CRITICAL: 730 days (2 years)

**Cleanup:**
Run periodic cleanup to remove old logs:
```python
from app.audit_log import AuditLogRetentionPolicy

AuditLogRetentionPolicy.cleanup_old_logs()
```

## Database Migration

The audit logs table is created via Alembic migration:

```bash
# Run migration
alembic upgrade head
```

Migration file: `alembic/versions/04dac41cf0df_add_audit_logs_table.py`

## Testing

Security features are thoroughly tested:

```bash
# Run security tests
pytest tests/test_security.py -v

# Run audit log tests
pytest tests/test_audit_log.py -v
```

## Best Practices

1. **Always use rate limiting** on public-facing endpoints
2. **Sanitize all user input** before processing or storing
3. **Verify CSRF tokens** on state-changing operations
4. **Log security events** for audit trail and forensics
5. **Review audit logs regularly** for suspicious activity
6. **Keep security headers updated** based on latest recommendations
7. **Use HTTPS in production** to protect data in transit
8. **Rotate secrets regularly** (JWT secret, CSRF secret)

## Security Checklist

- [x] Rate limiting per user/IP
- [x] CSRF protection on state-changing operations
- [x] Input sanitization (XSS, SQL injection)
- [x] Security headers (CSP, HSTS, etc.)
- [x] Audit logging for critical events
- [x] Password hashing (bcrypt)
- [x] JWT authentication
- [x] HTTPS enforcement (via headers)
- [x] SQL injection prevention (ORM)
- [x] XSS prevention (input sanitization)

## Compliance

The security implementation helps meet requirements for:
- **GDPR:** Audit logging, data access tracking
- **SOC 2:** Security monitoring, access controls
- **HIPAA:** Audit trails, access logging (if handling health data)

## Monitoring

Monitor security metrics:
- Failed login attempts
- Rate limit violations
- CSRF violations
- Invalid input attempts
- Unusual access patterns

Set up alerts for:
- Multiple failed login attempts from same IP
- Repeated rate limit violations
- CSRF token validation failures
- SQL injection or XSS attempts

## Incident Response

In case of security incident:
1. Review audit logs for affected time period
2. Identify compromised accounts
3. Reset credentials if necessary
4. Block malicious IPs
5. Review and update security measures
6. Document incident and response

## Future Enhancements

Potential security improvements:
- [ ] Two-factor authentication (2FA)
- [ ] IP whitelisting/blacklisting
- [ ] Advanced bot detection
- [ ] Session management and revocation
- [ ] API key authentication
- [ ] OAuth2 integration
- [ ] Automated security scanning
- [ ] Intrusion detection system (IDS)
