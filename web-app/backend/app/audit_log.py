"""
Audit logging for security-critical events and user actions.
"""
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid
import logging
import json

from .database import Base, get_db
from .models import GUID

logger = logging.getLogger(__name__)


# ============================================================================
# Audit Log Model
# ============================================================================

class AuditLog(Base):
    """
    Audit log table for tracking security-critical events.
    """
    __tablename__ = "audit_logs"
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    user_id = Column(GUID(), nullable=True, index=True)
    username = Column(String(255), nullable=True)
    ip_address = Column(String(45), nullable=True, index=True)  # IPv6 max length
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(100), nullable=True, index=True)
    endpoint = Column(String(255), nullable=True)
    method = Column(String(10), nullable=True)
    status_code = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    severity = Column(String(20), nullable=False, default="info", index=True)
    
    def __repr__(self):
        return f"<AuditLog {self.event_type} at {self.timestamp}>"



# ============================================================================
# Event Types
# ============================================================================

class AuditEventType:
    """Enumeration of audit event types."""
    
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    PASSWORD_CHANGE = "auth.password.change"
    REGISTRATION = "auth.registration"
    
    # Evaluation events
    EVALUATION_CREATE = "evaluation.create"
    EVALUATION_COMPLETE = "evaluation.complete"
    EVALUATION_ERROR = "evaluation.error"
    EVALUATION_EXPORT = "evaluation.export"
    
    # Configuration events
    CONFIG_UPDATE = "config.update"
    PREFERENCES_UPDATE = "preferences.update"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "security.rate_limit"
    CSRF_VIOLATION = "security.csrf_violation"
    INVALID_INPUT = "security.invalid_input"
    UNAUTHORIZED_ACCESS = "security.unauthorized"
    
    # Data events
    DATA_ACCESS = "data.access"
    DATA_EXPORT = "data.export"
    DATA_DELETE = "data.delete"


class AuditSeverity:
    """Enumeration of audit log severity levels."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"



# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """
    Service for logging audit events to database and application logs.
    """
    
    @staticmethod
    def log_event(
        event_type: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = AuditSeverity.INFO
    ):
        """
        Log an audit event to the database.
        
        Args:
            event_type: Type of event (use AuditEventType constants)
            user_id: User ID if applicable
            username: Username if applicable
            ip_address: Client IP address
            user_agent: Client user agent string
            request_id: Request ID for tracing
            endpoint: API endpoint
            method: HTTP method
            status_code: HTTP status code
            details: Additional event details as dictionary
            severity: Event severity level
        """
        try:
            # Create audit log entry
            audit_entry = AuditLog(
                event_type=event_type,
                user_id=user_id,
                username=username,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                details=details,
                severity=severity
            )
            
            # Save to database
            db = next(get_db())
            db.add(audit_entry)
            db.commit()
            
            # Also log to application logger
            log_message = f"AUDIT: {event_type}"
            if username:
                log_message += f" | User: {username}"
            if ip_address:
                log_message += f" | IP: {ip_address}"
            if details:
                log_message += f" | Details: {json.dumps(details)}"
            
            # Log at appropriate level
            if severity == AuditSeverity.CRITICAL:
                logger.critical(log_message)
            elif severity == AuditSeverity.ERROR:
                logger.error(log_message)
            elif severity == AuditSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception as e:
            # Don't let audit logging failures break the application
            logger.error(f"Failed to write audit log: {e}", exc_info=True)
    
    @staticmethod
    def log_authentication_event(
        event_type: str,
        username: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log an authentication-related event.
        
        Args:
            event_type: Type of auth event
            username: Username attempting authentication
            ip_address: Client IP address
            user_agent: Client user agent
            success: Whether authentication was successful
            details: Additional details
        """
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
        
        AuditLogger.log_event(
            event_type=event_type,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity=severity
        )
    
    @staticmethod
    def log_evaluation_event(
        event_type: str,
        user_id: str,
        username: str,
        session_id: str,
        ip_address: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log an evaluation-related event.
        
        Args:
            event_type: Type of evaluation event
            user_id: User ID
            username: Username
            session_id: Evaluation session ID
            ip_address: Client IP address
            details: Additional details
        """
        event_details = {"session_id": session_id}
        if details:
            event_details.update(details)
        
        AuditLogger.log_event(
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            details=event_details,
            severity=AuditSeverity.INFO
        )
    
    @staticmethod
    def log_security_event(
        event_type: str,
        ip_address: str,
        endpoint: str,
        method: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None
    ):
        """
        Log a security-related event.
        
        Args:
            event_type: Type of security event
            ip_address: Client IP address
            endpoint: API endpoint
            method: HTTP method
            details: Additional details
            user_id: User ID if applicable
            username: Username if applicable
        """
        AuditLogger.log_event(
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            details=details,
            severity=AuditSeverity.WARNING
        )
    
    @staticmethod
    def log_config_change(
        user_id: str,
        username: str,
        config_type: str,
        old_value: Any,
        new_value: Any,
        ip_address: str
    ):
        """
        Log a configuration change event.
        
        Args:
            user_id: User ID making the change
            username: Username making the change
            config_type: Type of configuration changed
            old_value: Previous value
            new_value: New value
            ip_address: Client IP address
        """
        details = {
            "config_type": config_type,
            "old_value": str(old_value),
            "new_value": str(new_value)
        }
        
        AuditLogger.log_event(
            event_type=AuditEventType.CONFIG_UPDATE,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            details=details,
            severity=AuditSeverity.INFO
        )



# ============================================================================
# Audit Log Retention Policy
# ============================================================================

class AuditLogRetentionPolicy:
    """
    Policy for managing audit log retention and cleanup.
    """
    
    # Retention periods by severity (in days)
    RETENTION_PERIODS = {
        AuditSeverity.DEBUG: 7,
        AuditSeverity.INFO: 90,
        AuditSeverity.WARNING: 180,
        AuditSeverity.ERROR: 365,
        AuditSeverity.CRITICAL: 730  # 2 years
    }
    
    @staticmethod
    def cleanup_old_logs():
        """
        Remove audit logs older than their retention period.
        Should be run periodically (e.g., daily cron job).
        """
        try:
            db = next(get_db())
            current_time = datetime.utcnow()
            
            for severity, days in AuditLogRetentionPolicy.RETENTION_PERIODS.items():
                cutoff_date = current_time - timedelta(days=days)
                
                # Delete logs older than retention period
                deleted = db.query(AuditLog).filter(
                    AuditLog.severity == severity,
                    AuditLog.timestamp < cutoff_date
                ).delete()
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} {severity} audit logs older than {days} days")
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to cleanup audit logs: {e}", exc_info=True)
    
    @staticmethod
    def get_audit_logs(
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """
        Query audit logs with filters.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            severity: Filter by severity
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of AuditLog entries
        """
        try:
            db = next(get_db())
            query = db.query(AuditLog)
            
            if event_type:
                query = query.filter(AuditLog.event_type == event_type)
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            if severity:
                query = query.filter(AuditLog.severity == severity)
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)
            
            # Order by timestamp descending (newest first)
            query = query.order_by(AuditLog.timestamp.desc())
            
            # Apply pagination
            query = query.limit(limit).offset(offset)
            
            return query.all()
            
        except Exception as e:
            logger.error(f"Failed to query audit logs: {e}", exc_info=True)
            return []


# Global audit logger instance
audit_logger = AuditLogger()
