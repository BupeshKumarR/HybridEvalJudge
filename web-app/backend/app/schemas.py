"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


# Enums
class EvaluationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IssueType(str, Enum):
    FACTUAL_ERROR = "factual_error"
    HALLUCINATION = "hallucination"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    NUMERICAL_ERROR = "numerical_error"
    BIAS = "bias"


class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerifierLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


# Base schemas
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=255)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserResponse(UserBase):
    id: UUID
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationConfig(BaseModel):
    judge_models: List[str] = Field(default_factory=list)
    enable_retrieval: bool = True
    aggregation_strategy: str = "weighted_average"


class EvaluationSessionCreate(BaseModel):
    source_text: str = Field(..., min_length=1)
    candidate_output: str = Field(..., min_length=1)
    config: Optional[EvaluationConfig] = None


class FlaggedIssueResponse(BaseModel):
    id: UUID
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    evidence: Optional[Dict[str, Any]] = None
    text_span_start: Optional[int] = None
    text_span_end: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class JudgeResultResponse(BaseModel):
    id: UUID
    judge_name: str
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    reasoning: Optional[str] = None
    response_time_ms: Optional[int] = None
    flagged_issues: List[FlaggedIssueResponse] = Field(default_factory=list)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VerifierVerdictResponse(BaseModel):
    id: UUID
    claim_text: str
    label: VerifierLabel
    confidence: float = Field(..., ge=0, le=1)
    evidence: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SessionMetadataResponse(BaseModel):
    total_judges: int = Field(..., gt=0)
    judges_used: List[str]
    aggregation_strategy: Optional[str] = None
    retrieval_enabled: bool = False
    num_retrieved_passages: Optional[int] = None
    num_verifier_verdicts: Optional[int] = None
    processing_time_ms: Optional[int] = None
    variance: Optional[float] = None
    standard_deviation: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class EvaluationSessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    source_text: str
    candidate_output: str
    consensus_score: Optional[float] = Field(None, ge=0, le=100)
    hallucination_score: Optional[float] = Field(None, ge=0, le=100)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    inter_judge_agreement: Optional[float] = None
    status: EvaluationStatus
    config: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    judge_results: List[JudgeResultResponse] = Field(default_factory=list)
    verifier_verdicts: List[VerifierVerdictResponse] = Field(default_factory=list)
    session_metadata: Optional[SessionMetadataResponse] = None

    model_config = ConfigDict(from_attributes=True)


class SessionSummary(BaseModel):
    id: UUID
    consensus_score: Optional[float] = None
    hallucination_score: Optional[float] = None
    status: EvaluationStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    source_preview: str
    candidate_preview: str
    num_judge_results: int = 0
    num_verifier_verdicts: int = 0
    num_flagged_issues: int = 0

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    total: int
    page: int
    limit: int
    has_more: bool


class UserPreferenceResponse(BaseModel):
    user_id: UUID
    default_judge_models: Optional[List[str]] = None
    default_retrieval_enabled: bool = True
    default_aggregation_strategy: str = "weighted_average"
    theme: str = "light"
    notifications_enabled: bool = True
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserPreferenceUpdate(BaseModel):
    default_judge_models: Optional[List[str]] = None
    default_retrieval_enabled: Optional[bool] = None
    default_aggregation_strategy: Optional[str] = None
    theme: Optional[str] = None
    notifications_enabled: Optional[bool] = None


# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[UUID] = None
    username: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


# WebSocket event schemas
class EvaluationProgressEvent(BaseModel):
    event: str = "evaluation_progress"
    data: Dict[str, Any]


class JudgeResultEvent(BaseModel):
    event: str = "judge_result"
    data: JudgeResultResponse


class EvaluationCompleteEvent(BaseModel):
    event: str = "evaluation_complete"
    data: EvaluationSessionResponse


class EvaluationErrorEvent(BaseModel):
    event: str = "evaluation_error"
    data: Dict[str, Any]
