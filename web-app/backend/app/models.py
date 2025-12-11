"""
SQLAlchemy ORM models for the LLM Judge Auditor web application.
"""
from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean, 
    ForeignKey, CheckConstraint, Index, TIMESTAMP, func, JSON
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import relationship, validates
from sqlalchemy import TypeDecorator, CHAR
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import re
import os

from .database import Base


# Database-agnostic UUID type
class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type when available, otherwise uses CHAR(36).
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(uuid.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


# Database-agnostic JSON type
def get_json_type():
    """Return appropriate JSON type based on database."""
    db_url = os.getenv("DATABASE_URL", "")
    if "postgresql" in db_url and os.getenv("USE_SQLITE", "false").lower() != "true":
        return JSONB
    return JSON


JSONType = get_json_type()


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    last_login = Column(TIMESTAMP, nullable=True)

    # Relationships
    evaluation_sessions = relationship(
        "EvaluationSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    preferences = relationship(
        "UserPreference",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    chat_sessions = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    # Constraints - removed PostgreSQL-specific regex constraint for SQLite compatibility
    # Email validation is handled by the @validates decorator
    __table_args__ = (
        CheckConstraint("length(username) >= 3", name="users_username_check"),
    )

    @validates('username')
    def validate_username(self, key, username):
        """Validate username length."""
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters long")
        return username

    @validates('email')
    def validate_email(self, key, email):
        """Validate email format."""
        email_pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")
        return email

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class EvaluationSession(Base):
    """Evaluation session model."""
    __tablename__ = "evaluation_sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    source_text = Column(Text, nullable=False)
    candidate_output = Column(Text, nullable=False)
    consensus_score = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    inter_judge_agreement = Column(Float, nullable=True)
    status = Column(String(50), default="pending", nullable=False, index=True)
    config = Column(JSONType, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    completed_at = Column(TIMESTAMP, nullable=True)

    # Relationships
    user = relationship("User", back_populates="evaluation_sessions")
    judge_results = relationship(
        "JudgeResult",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    verifier_verdicts = relationship(
        "VerifierVerdict",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    session_metadata = relationship(
        "SessionMetadata",
        back_populates="evaluation_session",
        uselist=False,
        cascade="all, delete-orphan"
    )
    claim_verdicts = relationship(
        "ClaimVerdict",
        back_populates="evaluation",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')",
            name="evaluation_sessions_status_check"
        ),
        CheckConstraint(
            "consensus_score IS NULL OR (consensus_score >= 0 AND consensus_score <= 100)",
            name="evaluation_sessions_consensus_score_check"
        ),
        CheckConstraint(
            "hallucination_score IS NULL OR (hallucination_score >= 0 AND hallucination_score <= 100)",
            name="evaluation_sessions_hallucination_score_check"
        ),
        CheckConstraint(
            "(confidence_interval_lower IS NULL AND confidence_interval_upper IS NULL) OR "
            "(confidence_interval_lower IS NOT NULL AND confidence_interval_upper IS NOT NULL AND "
            "confidence_interval_lower <= confidence_interval_upper)",
            name="evaluation_sessions_confidence_check"
        ),
        Index("idx_evaluation_sessions_user_created", "user_id", "created_at"),
        Index("idx_evaluation_sessions_consensus_score", "consensus_score"),
        Index("idx_evaluation_sessions_hallucination_score", "hallucination_score"),
        Index("idx_evaluation_sessions_user_status", "user_id", "status"),
    )

    @validates('status')
    def validate_status(self, key, status):
        """Validate status value."""
        valid_statuses = ['pending', 'in_progress', 'completed', 'failed', 'cancelled']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return status

    @validates('consensus_score', 'hallucination_score')
    def validate_score(self, key, score):
        """Validate score is between 0 and 100."""
        if score is not None and (score < 0 or score > 100):
            raise ValueError(f"{key} must be between 0 and 100")
        return score

    def __repr__(self):
        return f"<EvaluationSession(id={self.id}, status='{self.status}')>"


class JudgeResult(Base):
    """Judge result model."""
    __tablename__ = "judge_results"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        GUID(),
        ForeignKey("evaluation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    judge_name = Column(String(255), nullable=False, index=True)
    score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    session = relationship("EvaluationSession", back_populates="judge_results")
    flagged_issues = relationship(
        "FlaggedIssue",
        back_populates="judge_result",
        cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "score >= 0 AND score <= 100",
            name="judge_results_score_check"
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="judge_results_confidence_check"
        ),
        CheckConstraint(
            "response_time_ms IS NULL OR response_time_ms >= 0",
            name="judge_results_response_time_check"
        ),
        Index("idx_judge_results_session_judge", "session_id", "judge_name"),
    )

    @validates('score')
    def validate_score(self, key, score):
        """Validate score is between 0 and 100."""
        if score < 0 or score > 100:
            raise ValueError("Score must be between 0 and 100")
        return score

    @validates('confidence')
    def validate_confidence(self, key, confidence):
        """Validate confidence is between 0 and 1."""
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return confidence

    @validates('response_time_ms')
    def validate_response_time(self, key, response_time_ms):
        """Validate response time is non-negative."""
        if response_time_ms is not None and response_time_ms < 0:
            raise ValueError("Response time must be non-negative")
        return response_time_ms

    def __repr__(self):
        return f"<JudgeResult(id={self.id}, judge='{self.judge_name}', score={self.score})>"


class FlaggedIssue(Base):
    """Flagged issue model."""
    __tablename__ = "flagged_issues"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    judge_result_id = Column(
        GUID(),
        ForeignKey("judge_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    issue_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    evidence = Column(JSONType, nullable=True)
    text_span_start = Column(Integer, nullable=True)
    text_span_end = Column(Integer, nullable=True)

    # Relationships
    judge_result = relationship("JudgeResult", back_populates="flagged_issues")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "issue_type IN ('factual_error', 'hallucination', 'unsupported_claim', "
            "'temporal_inconsistency', 'numerical_error', 'bias')",
            name="flagged_issues_issue_type_check"
        ),
        CheckConstraint(
            "severity IN ('low', 'medium', 'high', 'critical')",
            name="flagged_issues_severity_check"
        ),
        CheckConstraint(
            "(text_span_start IS NULL AND text_span_end IS NULL) OR "
            "(text_span_start IS NOT NULL AND text_span_end IS NOT NULL AND "
            "text_span_start <= text_span_end)",
            name="flagged_issues_text_span_check"
        ),
        Index("idx_flagged_issues_type_severity", "issue_type", "severity"),
    )

    @validates('issue_type')
    def validate_issue_type(self, key, issue_type):
        """Validate issue type."""
        valid_types = [
            'factual_error', 'hallucination', 'unsupported_claim',
            'temporal_inconsistency', 'numerical_error', 'bias'
        ]
        if issue_type not in valid_types:
            raise ValueError(f"Issue type must be one of: {', '.join(valid_types)}")
        return issue_type

    @validates('severity')
    def validate_severity(self, key, severity):
        """Validate severity level."""
        valid_severities = ['low', 'medium', 'high', 'critical']
        if severity not in valid_severities:
            raise ValueError(f"Severity must be one of: {', '.join(valid_severities)}")
        return severity

    def __repr__(self):
        return f"<FlaggedIssue(id={self.id}, type='{self.issue_type}', severity='{self.severity}')>"


class VerifierVerdict(Base):
    """Verifier verdict model."""
    __tablename__ = "verifier_verdicts"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        GUID(),
        ForeignKey("evaluation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    claim_text = Column(Text, nullable=False)
    label = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    evidence = Column(JSONType, nullable=True)
    reasoning = Column(Text, nullable=True)

    # Relationships
    session = relationship("EvaluationSession", back_populates="verifier_verdicts")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "label IN ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO')",
            name="verifier_verdicts_label_check"
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="verifier_verdicts_confidence_check"
        ),
        Index("idx_verifier_verdicts_session_label", "session_id", "label"),
    )

    @validates('label')
    def validate_label(self, key, label):
        """Validate label value."""
        valid_labels = ['SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO']
        if label not in valid_labels:
            raise ValueError(f"Label must be one of: {', '.join(valid_labels)}")
        return label

    @validates('confidence')
    def validate_confidence(self, key, confidence):
        """Validate confidence is between 0 and 1."""
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return confidence

    def __repr__(self):
        return f"<VerifierVerdict(id={self.id}, label='{self.label}')>"


class ClaimVerdict(Base):
    """
    Claim verdict model for storing extracted claims with their verdicts.
    
    This model stores individual claims extracted from LLM responses,
    their classification type, verification verdict, and text span positions
    for highlighting in the UI.
    
    Requirements: 5.4, 5.5
    """
    __tablename__ = "claim_verdicts"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(
        GUID(),
        ForeignKey("evaluation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    claim_text = Column(Text, nullable=False)
    claim_type = Column(String(50), nullable=False, index=True)
    verdict = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    judge_name = Column(String(255), nullable=True, index=True)
    text_span_start = Column(Integer, nullable=False)
    text_span_end = Column(Integer, nullable=False)
    reasoning = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    # Relationships
    evaluation = relationship("EvaluationSession", back_populates="claim_verdicts")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "claim_type IN ('numerical', 'temporal', 'definitional', 'general')",
            name="claim_verdicts_claim_type_check"
        ),
        CheckConstraint(
            "verdict IN ('SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO')",
            name="claim_verdicts_verdict_check"
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="claim_verdicts_confidence_check"
        ),
        CheckConstraint(
            "text_span_start >= 0 AND text_span_end >= text_span_start",
            name="claim_verdicts_text_span_check"
        ),
        Index("idx_claim_verdicts_evaluation_verdict", "evaluation_id", "verdict"),
        Index("idx_claim_verdicts_claim_type", "claim_type"),
    )

    @validates('claim_type')
    def validate_claim_type(self, key, claim_type):
        """Validate claim type value."""
        valid_types = ['numerical', 'temporal', 'definitional', 'general']
        if claim_type not in valid_types:
            raise ValueError(f"Claim type must be one of: {', '.join(valid_types)}")
        return claim_type

    @validates('verdict')
    def validate_verdict(self, key, verdict):
        """Validate verdict value."""
        valid_verdicts = ['SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO']
        if verdict not in valid_verdicts:
            raise ValueError(f"Verdict must be one of: {', '.join(valid_verdicts)}")
        return verdict

    @validates('confidence')
    def validate_confidence(self, key, confidence):
        """Validate confidence is between 0 and 1."""
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return confidence

    @validates('text_span_start', 'text_span_end')
    def validate_text_span(self, key, value):
        """Validate text span values are non-negative."""
        if value < 0:
            raise ValueError(f"{key} must be non-negative")
        return value

    def __repr__(self):
        return f"<ClaimVerdict(id={self.id}, verdict='{self.verdict}', type='{self.claim_type}')>"


class SessionMetadata(Base):
    """Session metadata model for advanced analytics."""
    __tablename__ = "session_metadata"

    session_id = Column(
        GUID(),
        ForeignKey("evaluation_sessions.id", ondelete="CASCADE"),
        primary_key=True
    )
    total_judges = Column(Integer, nullable=False)
    judges_used = Column(JSONType, nullable=False)
    aggregation_strategy = Column(String(100), nullable=True, index=True)
    retrieval_enabled = Column(Boolean, default=False, index=True)
    num_retrieved_passages = Column(Integer, nullable=True)
    num_verifier_verdicts = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    variance = Column(Float, nullable=True)
    standard_deviation = Column(Float, nullable=True)

    # Relationships
    evaluation_session = relationship("EvaluationSession", back_populates="session_metadata")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "total_judges > 0",
            name="session_metadata_total_judges_check"
        ),
        CheckConstraint(
            "num_retrieved_passages IS NULL OR num_retrieved_passages >= 0",
            name="session_metadata_num_passages_check"
        ),
        CheckConstraint(
            "num_verifier_verdicts IS NULL OR num_verifier_verdicts >= 0",
            name="session_metadata_num_verdicts_check"
        ),
        CheckConstraint(
            "processing_time_ms IS NULL OR processing_time_ms >= 0",
            name="session_metadata_processing_time_check"
        ),
    )

    @validates('total_judges')
    def validate_total_judges(self, key, total_judges):
        """Validate total judges is positive."""
        if total_judges <= 0:
            raise ValueError("Total judges must be positive")
        return total_judges

    def __repr__(self):
        return f"<SessionMetadata(session_id={self.session_id}, total_judges={self.total_judges})>"


class UserPreference(Base):
    """User preference model for configuration management."""
    __tablename__ = "user_preferences"

    user_id = Column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    default_judge_models = Column(JSONType, nullable=True)
    default_retrieval_enabled = Column(Boolean, default=True)
    default_aggregation_strategy = Column(String(100), default="weighted_average")
    theme = Column(String(50), default="light")
    notifications_enabled = Column(Boolean, default=True)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id})>"


class ChatSession(Base):
    """Chat session model for conversational interactions."""
    __tablename__ = "chat_sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    ollama_model = Column(String(255), nullable=False, default="llama3.2")
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )

    # Constraints
    __table_args__ = (
        Index("idx_chat_sessions_user_created", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, model='{self.ollama_model}')>"


class ChatMessage(Base):
    """Chat message model for storing conversation messages."""
    __tablename__ = "chat_messages"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        GUID(),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    evaluation_id = Column(
        GUID(),
        ForeignKey("evaluation_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    created_at = Column(TIMESTAMP, default=datetime.utcnow, index=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    evaluation = relationship("EvaluationSession")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="chat_messages_role_check"
        ),
        Index("idx_chat_messages_session_created", "session_id", "created_at"),
    )

    @validates('role')
    def validate_role(self, key, role):
        """Validate role value."""
        valid_roles = ['user', 'assistant']
        if role not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(valid_roles)}")
        return role

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}', session_id={self.session_id})>"
