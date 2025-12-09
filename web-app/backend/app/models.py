"""
SQLAlchemy ORM models for the LLM Judge Auditor web application.
"""
from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean, 
    ForeignKey, CheckConstraint, Index, TIMESTAMP, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import re

from .database import Base


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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

    # Constraints
    __table_args__ = (
        CheckConstraint("char_length(username) >= 3", name="users_username_check"),
        CheckConstraint(
            "email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'",
            name="users_email_check"
        ),
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    source_text = Column(Text, nullable=False)
    candidate_output = Column(Text, nullable=False)
    consensus_score = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    inter_judge_agreement = Column(Float, nullable=True)
    status = Column(String(50), default="pending", nullable=False, index=True)
    config = Column(JSONB, nullable=True)
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    judge_result_id = Column(
        UUID(as_uuid=True),
        ForeignKey("judge_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    issue_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=False)
    evidence = Column(JSONB, nullable=True)
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

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    claim_text = Column(Text, nullable=False)
    label = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    evidence = Column(JSONB, nullable=True)
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


class SessionMetadata(Base):
    """Session metadata model for advanced analytics."""
    __tablename__ = "session_metadata"

    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_sessions.id", ondelete="CASCADE"),
        primary_key=True
    )
    total_judges = Column(Integer, nullable=False)
    judges_used = Column(JSONB, nullable=False)
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
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True
    )
    default_judge_models = Column(JSONB, nullable=True)
    default_retrieval_enabled = Column(Boolean, default=True)
    default_aggregation_strategy = Column(String(100), default="weighted_average")
    theme = Column(String(50), default="light")
    notifications_enabled = Column(Boolean, default=True)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id})>"
