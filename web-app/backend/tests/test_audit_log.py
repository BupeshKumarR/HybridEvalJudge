"""
Tests for audit logging functionality.
"""
import pytest
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.audit_log import (
    AuditLog,
    AuditLogger,
    AuditEventType,
    AuditSeverity,
    AuditLogRetentionPolicy
)


class TestAuditLog:
    """Tests for audit log model."""
    
    def test_create_audit_log(self, db_session: Session, created_user):
        """Test creating an audit log entry."""
        audit_entry = AuditLog(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id=str(created_user.id),
            username="testuser",
            ip_address="127.0.0.1",
            severity=AuditSeverity.INFO
        )
        
        db_session.add(audit_entry)
        db_session.commit()
        db_session.refresh(audit_entry)
        
        assert audit_entry.id is not None
        assert audit_entry.timestamp is not None
        assert audit_entry.event_type == AuditEventType.LOGIN_SUCCESS
        assert audit_entry.username == "testuser"


class TestAuditLogger:
    """Tests for audit logger service."""
    
    def test_log_event(self, db_session: Session, created_user):
        """Test logging a basic event."""
        AuditLogger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id=str(created_user.id),
            username="testuser",
            ip_address="127.0.0.1",
            severity=AuditSeverity.INFO
        )
        
        # Query the log
        logs = db_session.query(AuditLog).filter(
            AuditLog.username == "testuser"
        ).all()
        
        assert len(logs) > 0
        assert logs[0].event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_log_authentication_event(self, db_session: Session):
        """Test logging an authentication event."""
        AuditLogger.log_authentication_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            username="testuser",
            ip_address="127.0.0.1",
            user_agent="Mozilla/5.0",
            success=True,
            details={"method": "password"}
        )
        
        # Query the log
        logs = db_session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.LOGIN_SUCCESS
        ).all()
        
        assert len(logs) > 0
        assert logs[0].username == "testuser"
        assert logs[0].details["method"] == "password"
    
    def test_log_evaluation_event(self, db_session: Session, created_user):
        """Test logging an evaluation event."""
        AuditLogger.log_evaluation_event(
            event_type=AuditEventType.EVALUATION_CREATE,
            user_id=str(created_user.id),
            username="testuser",
            session_id="test-session-id",
            ip_address="127.0.0.1",
            details={"source_length": 100}
        )
        
        # Query the log
        logs = db_session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.EVALUATION_CREATE
        ).all()
        
        assert len(logs) > 0
        assert logs[0].details["session_id"] == "test-session-id"
    
    def test_log_security_event(self, db_session: Session):
        """Test logging a security event."""
        AuditLogger.log_security_event(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            ip_address="127.0.0.1",
            endpoint="/api/v1/evaluations",
            method="POST",
            details={"limit": 20}
        )
        
        # Query the log
        logs = db_session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.RATE_LIMIT_EXCEEDED
        ).all()
        
        assert len(logs) > 0
        assert logs[0].severity == AuditSeverity.WARNING
    
    def test_log_config_change(self, db_session: Session, created_user):
        """Test logging a configuration change."""
        AuditLogger.log_config_change(
            user_id=str(created_user.id),
            username="testuser",
            config_type="judge_models",
            old_value=["gpt-4"],
            new_value=["gpt-4", "claude-3"],
            ip_address="127.0.0.1"
        )
        
        # Query the log
        logs = db_session.query(AuditLog).filter(
            AuditLog.event_type == AuditEventType.CONFIG_UPDATE
        ).all()
        
        assert len(logs) > 0
        assert logs[0].details["config_type"] == "judge_models"


class TestAuditLogRetentionPolicy:
    """Tests for audit log retention policy."""
    
    def test_get_audit_logs(self, db_session: Session):
        """Test querying audit logs with filters."""
        # Create some test logs
        for i in range(5):
            audit_entry = AuditLog(
                event_type=AuditEventType.LOGIN_SUCCESS,
                username=f"user{i}",
                ip_address="127.0.0.1",
                severity=AuditSeverity.INFO
            )
            db.add(audit_entry)
        db.commit()
        
        # Query logs
        logs = AuditLogRetentionPolicy.get_audit_logs(
            event_type=AuditEventType.LOGIN_SUCCESS,
            limit=10
        )
        
        assert len(logs) >= 5
    
    def test_get_audit_logs_with_date_filter(self, db_session: Session):
        """Test querying audit logs with date filters."""
        # Create a log with specific timestamp
        old_log = AuditLog(
            event_type=AuditEventType.LOGIN_SUCCESS,
            username="olduser",
            ip_address="127.0.0.1",
            severity=AuditSeverity.INFO,
            timestamp=datetime.utcnow() - timedelta(days=10)
        )
        db.add(old_log)
        db.commit()
        
        # Query logs from last 5 days
        start_date = datetime.utcnow() - timedelta(days=5)
        logs = AuditLogRetentionPolicy.get_audit_logs(
            start_date=start_date,
            limit=100
        )
        
        # Old log should not be in results
        usernames = [log.username for log in logs]
        assert "olduser" not in usernames
    
    def test_cleanup_old_logs(self, db_session: Session):
        """Test cleanup of old audit logs."""
        # Create old logs with different severities
        old_debug_log = AuditLog(
            event_type=AuditEventType.DATA_ACCESS,
            username="debuguser",
            ip_address="127.0.0.1",
            severity=AuditSeverity.DEBUG,
            timestamp=datetime.utcnow() - timedelta(days=10)
        )
        db.add(old_debug_log)
        db.commit()
        
        # Run cleanup
        AuditLogRetentionPolicy.cleanup_old_logs()
        
        # Old debug log should be deleted (retention is 7 days)
        remaining_logs = db_session.query(AuditLog).filter(
            AuditLog.username == "debuguser"
        ).all()
        
        assert len(remaining_logs) == 0
