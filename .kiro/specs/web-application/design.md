# Design Document: LLM Judge Auditor Web Application

## Overview

This design specifies a production-grade web application for the LLM Judge Auditor, featuring a React-based frontend with real-time streaming, a FastAPI backend with WebSocket support, and PostgreSQL for persistence. The application provides rich visualizations of judge decisions, confidence metrics, and hallucination detection using D3.js and Recharts.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Chat         │  │ Visualization│  │ History      │     │
│  │ Interface    │  │ Dashboard    │  │ Sidebar      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
                    ┌───────┴────────┐
                    │   API Gateway  │
                    │   (FastAPI)    │
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌──────▼──────┐
│   REST API     │  │   WebSocket    │  │  Auth       │
│   Endpoints    │  │   Server       │  │  Service    │
└───────┬────────┘  └───────┬────────┘  └──────┬──────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
        ┌───────▼────────┐      ┌──────▼──────┐
        │  Evaluation    │      │  Database   │
        │  Toolkit       │      │ (PostgreSQL)│
        │  (Python)      │      └─────────────┘
        └────────────────┘
```

### Technology Stack

**Frontend:**
- React 18 with TypeScript
- TailwindCSS for styling
- Recharts + D3.js for visualizations
- Socket.IO client for WebSocket
- React Query for data fetching
- Zustand for state management

**Backend:**
- FastAPI (Python 3.11+)
- SQLAlchemy ORM
- PostgreSQL 15+
- Socket.IO for WebSocket
- JWT for authentication
- Pydantic for validation

**Infrastructure:**
- Docker + Docker Compose
- Nginx as reverse proxy
- Redis for caching (optional)

## Components and Interfaces

### 1. Frontend Components

#### ChatInterface Component

```typescript
interface ChatMessage {
  id: string;
  type: 'user' | 'system' | 'evaluation';
  timestamp: Date;
  content: {
    sourceText?: string;
    candidateOutput?: string;
    results?: EvaluationResults;
  };
}

interface ChatInterfaceProps {
  sessionId: string;
  onSubmitEvaluation: (source: string, candidate: string) => void;
}

// Component renders:
// - Message history with auto-scroll
// - Input form for source/candidate text
// - Loading indicators during evaluation
// - Expandable result cards
```

#### VisualizationDashboard Component

```typescript
interface VisualizationDashboardProps {
  evaluationResults: EvaluationResults;
  showAdvancedMetrics: boolean;
}

// Sub-components:
// - JudgeComparisonChart: Bar chart of individual judge scores
// - ConfidenceGauge: Radial gauge showing confidence level
// - HallucinationMeter: Thermometer-style visualization
// - ScoreDistribution: Violin plot or box plot
// - InterJudgeAgreement: Heatmap of judge correlations
```

#### HistorySidebar Component

```typescript
interface HistorySession {
  id: string;
  timestamp: Date;
  sourcePreview: string;
  consensusScore: number;
  hallucinationScore: number;
}

interface HistorySidebarProps {
  sessions: HistorySession[];
  currentSessionId: string;
  onSelectSession: (id: string) => void;
  onLoadMore: () => void;
}
```

### 2. Backend API Endpoints

#### REST API

```python
# POST /api/v1/evaluations
# Create new evaluation session
Request:
{
  "source_text": str,
  "candidate_output": str,
  "config": {
    "judge_models": List[str],
    "enable_retrieval": bool,
    "aggregation_strategy": str
  }
}

Response:
{
  "session_id": str,
  "websocket_url": str,
  "status": "pending"
}

# GET /api/v1/evaluations/{session_id}
# Get evaluation results
Response:
{
  "session_id": str,
  "status": "completed" | "pending" | "failed",
  "results": EvaluationResults,
  "metrics": AdvancedMetrics
}

# GET /api/v1/evaluations
# List user's evaluation history
Query Params: page, limit, sort_by, filter_by_score
Response:
{
  "sessions": List[SessionSummary],
  "total": int,
  "page": int,
  "has_more": bool
}

# GET /api/v1/evaluations/{session_id}/export
# Export evaluation results
Query Params: format (json|csv|pdf)
Response: File download

# POST /api/v1/auth/login
# User authentication
Request: {"username": str, "password": str}
Response: {"access_token": str, "token_type": "bearer"}
```

#### WebSocket Events

```python
# Client -> Server
{
  "event": "start_evaluation",
  "data": {
    "session_id": str,
    "source_text": str,
    "candidate_output": str,
    "config": dict
  }
}

# Server -> Client (streaming updates)
{
  "event": "evaluation_progress",
  "data": {
    "stage": "retrieval" | "verification" | "judging" | "aggregation",
    "progress": float,  # 0-100
    "message": str
  }
}

{
  "event": "judge_result",
  "data": {
    "judge_name": str,
    "score": float,
    "confidence": float,
    "reasoning": str,
    "flagged_issues": List[Issue]
  }
}

{
  "event": "evaluation_complete",
  "data": {
    "consensus_score": float,
    "hallucination_score": float,
    "confidence_interval": [float, float],
    "inter_judge_agreement": float,
    "full_results": EvaluationResults
  }
}

{
  "event": "evaluation_error",
  "data": {
    "error_type": str,
    "message": str,
    "recovery_suggestions": List[str]
  }
}
```

### 3. Database Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Evaluation sessions
CREATE TABLE evaluation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    source_text TEXT NOT NULL,
    candidate_output TEXT NOT NULL,
    consensus_score FLOAT,
    hallucination_score FLOAT,
    confidence_interval_lower FLOAT,
    confidence_interval_upper FLOAT,
    inter_judge_agreement FLOAT,
    status VARCHAR(50) DEFAULT 'pending',
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    INDEX idx_user_created (user_id, created_at DESC),
    INDEX idx_consensus_score (consensus_score),
    INDEX idx_status (status)
);

-- Judge results
CREATE TABLE judge_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    judge_name VARCHAR(255) NOT NULL,
    score FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id)
);

-- Flagged issues
CREATE TABLE flagged_issues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    judge_result_id UUID REFERENCES judge_results(id) ON DELETE CASCADE,
    issue_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    evidence JSONB,
    text_span_start INTEGER,
    text_span_end INTEGER,
    INDEX idx_judge_result (judge_result_id),
    INDEX idx_issue_type (issue_type)
);

-- Verifier verdicts
CREATE TABLE verifier_verdicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    label VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    evidence JSONB,
    reasoning TEXT,
    INDEX idx_session (session_id)
);

-- Session metadata (for advanced analytics)
CREATE TABLE session_metadata (
    session_id UUID PRIMARY KEY REFERENCES evaluation_sessions(id) ON DELETE CASCADE,
    total_judges INTEGER,
    judges_used JSONB,
    aggregation_strategy VARCHAR(100),
    retrieval_enabled BOOLEAN,
    num_retrieved_passages INTEGER,
    num_verifier_verdicts INTEGER,
    processing_time_ms INTEGER,
    variance FLOAT,
    standard_deviation FLOAT
);
```

## Data Models

### Core Data Models

```python
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

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

class FlaggedIssue(BaseModel):
    type: IssueType
    severity: IssueSeverity
    description: str
    evidence: List[str]
    text_span: Optional[tuple[int, int]] = None

class JudgeResult(BaseModel):
    judge_name: str
    score: float  # 0-100
    confidence: float  # 0-1
    reasoning: str
    flagged_issues: List[FlaggedIssue]
    response_time_ms: int

class VerifierVerdict(BaseModel):
    claim_text: str
    label: str  # SUPPORTED, REFUTED, NOT_ENOUGH_INFO
    confidence: float
    evidence: List[str]
    reasoning: str

class ConfidenceMetrics(BaseModel):
    mean_confidence: float
    confidence_interval: tuple[float, float]
    confidence_level: float  # e.g., 0.95 for 95% CI
    is_low_confidence: bool

class InterJudgeAgreement(BaseModel):
    cohens_kappa: Optional[float]  # For 2 judges
    fleiss_kappa: Optional[float]  # For 3+ judges
    krippendorff_alpha: Optional[float]  # Alternative metric
    pairwise_correlations: Dict[str, Dict[str, float]]
    interpretation: str  # "poor", "fair", "moderate", "substantial", "perfect"

class HallucinationMetrics(BaseModel):
    overall_score: float  # 0-100
    breakdown_by_type: Dict[IssueType, float]
    affected_text_spans: List[tuple[int, int, str]]  # start, end, issue_type
    severity_distribution: Dict[IssueSeverity, int]

class EvaluationResults(BaseModel):
    session_id: str
    consensus_score: float
    judge_results: List[JudgeResult]
    verifier_verdicts: List[VerifierVerdict]
    confidence_metrics: ConfidenceMetrics
    inter_judge_agreement: InterJudgeAgreement
    hallucination_metrics: HallucinationMetrics
    variance: float
    standard_deviation: float
    processing_time_ms: int
    timestamp: datetime
```

## Hallucination Score Calculation

### Formula and Methodology

The hallucination score is a composite metric (0-100) calculated using multiple factors:

```python
def calculate_hallucination_score(
    judge_results: List[JudgeResult],
    verifier_verdicts: List[VerifierVerdict],
    consensus_score: float
) -> HallucinationMetrics:
    """
    Calculate comprehensive hallucination score.
    
    Components:
    1. Inverse of consensus score (40% weight)
    2. Verifier refutation rate (30% weight)
    3. Judge-flagged issues severity (20% weight)
    4. Confidence penalty (10% weight)
    """
    
    # Component 1: Inverse consensus (lower score = more hallucination)
    inverse_consensus = (100 - consensus_score) * 0.4
    
    # Component 2: Verifier refutation rate
    if verifier_verdicts:
        refuted_count = sum(1 for v in verifier_verdicts if v.label == "REFUTED")
        refutation_rate = (refuted_count / len(verifier_verdicts)) * 100
        verifier_component = refutation_rate * 0.3
    else:
        verifier_component = 0
    
    # Component 3: Weighted issue severity
    issue_weights = {
        IssueSeverity.LOW: 0.25,
        IssueSeverity.MEDIUM: 0.5,
        IssueSeverity.HIGH: 0.75,
        IssueSeverity.CRITICAL: 1.0
    }
    
    all_issues = [issue for jr in judge_results for issue in jr.flagged_issues]
    if all_issues:
        weighted_severity = sum(issue_weights[issue.severity] for issue in all_issues)
        max_possible = len(all_issues)
        severity_score = (weighted_severity / max_possible) * 100 * 0.2
    else:
        severity_score = 0
    
    # Component 4: Confidence penalty (low confidence suggests uncertainty/hallucination)
    mean_confidence = sum(jr.confidence for jr in judge_results) / len(judge_results)
    confidence_penalty = (1 - mean_confidence) * 100 * 0.1
    
    # Final score
    hallucination_score = (
        inverse_consensus +
        verifier_component +
        severity_score +
        confidence_penalty
    )
    
    # Calculate breakdown by type
    breakdown = {}
    for issue_type in IssueType:
        type_issues = [i for i in all_issues if i.type == issue_type]
        if type_issues:
            type_score = sum(issue_weights[i.severity] for i in type_issues)
            breakdown[issue_type] = (type_score / len(type_issues)) * 100
        else:
            breakdown[issue_type] = 0.0
    
    # Extract affected text spans
    affected_spans = [
        (issue.text_span[0], issue.text_span[1], issue.type.value)
        for issue in all_issues
        if issue.text_span
    ]
    
    # Severity distribution
    severity_dist = {
        severity: sum(1 for i in all_issues if i.severity == severity)
        for severity in IssueSeverity
    }
    
    return HallucinationMetrics(
        overall_score=min(100, max(0, hallucination_score)),
        breakdown_by_type=breakdown,
        affected_text_spans=affected_spans,
        severity_distribution=severity_dist
    )
```

## Confidence Metrics Calculation

### Statistical Methods

```python
import numpy as np
from scipy import stats

def calculate_confidence_metrics(
    judge_results: List[JudgeResult],
    confidence_level: float = 0.95
) -> ConfidenceMetrics:
    """
    Calculate confidence interval for consensus score.
    
    Uses bootstrap resampling for robust CI estimation.
    """
    scores = [jr.score for jr in judge_results]
    confidences = [jr.confidence for jr in judge_results]
    
    # Mean confidence
    mean_conf = np.mean(confidences)
    
    # Bootstrap confidence interval for consensus score
    n_bootstrap = 10000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentile-based CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    # Determine if confidence is low
    ci_width = ci_upper - ci_lower
    is_low_confidence = ci_width > 20 or mean_conf < 0.7
    
    return ConfidenceMetrics(
        mean_confidence=mean_conf,
        confidence_interval=(ci_lower, ci_upper),
        confidence_level=confidence_level,
        is_low_confidence=is_low_confidence
    )
```

## Inter-Judge Agreement Calculation

### Statistical Measures

```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

def calculate_inter_judge_agreement(
    judge_results: List[JudgeResult]
) -> InterJudgeAgreement:
    """
    Calculate inter-judge agreement using multiple metrics.
    """
    n_judges = len(judge_results)
    
    if n_judges < 2:
        return InterJudgeAgreement(
            cohens_kappa=None,
            fleiss_kappa=None,
            krippendorff_alpha=None,
            pairwise_correlations={},
            interpretation="insufficient_judges"
        )
    
    # Convert scores to categorical ratings for kappa calculation
    # Bins: 0-20 (poor), 20-40 (fair), 40-60 (moderate), 60-80 (good), 80-100 (excellent)
    def score_to_category(score):
        if score < 20: return 0
        elif score < 40: return 1
        elif score < 60: return 2
        elif score < 80: return 3
        else: return 4
    
    categories = [score_to_category(jr.score) for jr in judge_results]
    
    # Cohen's Kappa (for 2 judges)
    cohens_kappa = None
    if n_judges == 2:
        # For simplicity, treat as binary agreement on category
        cohens_kappa = 1.0 if categories[0] == categories[1] else 0.0
    
    # Fleiss' Kappa (for 3+ judges)
    fleiss_kappa = None
    if n_judges >= 3:
        fleiss_kappa = calculate_fleiss_kappa(categories)
    
    # Pairwise correlations
    scores = [jr.score for jr in judge_results]
    pairwise_corr = {}
    for i, jr1 in enumerate(judge_results):
        pairwise_corr[jr1.judge_name] = {}
        for j, jr2 in enumerate(judge_results):
            if i != j:
                # Pearson correlation (simplified for 2 points)
                corr = 1.0 - abs(jr1.score - jr2.score) / 100
                pairwise_corr[jr1.judge_name][jr2.judge_name] = corr
    
    # Interpretation
    kappa_value = fleiss_kappa if fleiss_kappa else cohens_kappa
    if kappa_value is None:
        interpretation = "insufficient_data"
    elif kappa_value < 0:
        interpretation = "poor"
    elif kappa_value < 0.20:
        interpretation = "slight"
    elif kappa_value < 0.40:
        interpretation = "fair"
    elif kappa_value < 0.60:
        interpretation = "moderate"
    elif kappa_value < 0.80:
        interpretation = "substantial"
    else:
        interpretation = "almost_perfect"
    
    return InterJudgeAgreement(
        cohens_kappa=cohens_kappa,
        fleiss_kappa=fleiss_kappa,
        krippendorff_alpha=None,  # Can be added if needed
        pairwise_correlations=pairwise_corr,
        interpretation=interpretation
    )

def calculate_fleiss_kappa(categories: List[int]) -> float:
    """
    Simplified Fleiss' Kappa for multiple raters.
    """
    # This is a simplified version
    # Full implementation would require matrix of ratings
    n = len(categories)
    if n < 3:
        return None
    
    # Calculate proportion of agreement
    mode_category = max(set(categories), key=categories.count)
    agreement = categories.count(mode_category) / n
    
    # Expected agreement (random chance)
    p_e = 1 / 5  # 5 categories
    
    # Kappa = (P_o - P_e) / (1 - P_e)
    kappa = (agreement - p_e) / (1 - p_e)
    
    return kappa
```

## Visualization Specifications

### 1. Judge Comparison Chart

**Type:** Horizontal Bar Chart with Error Bars

**Library:** Recharts

**Data Structure:**
```typescript
interface JudgeChartData {
  judgeName: string;
  score: number;
  confidence: number;
  confidenceLower: number;
  confidenceUpper: number;
  color: string;  // Based on score: green/yellow/red
}
```

**Visual Features:**
- Bars colored by score (green: 80-100, yellow: 50-79, red: 0-49)
- Error bars showing confidence intervals
- Hover tooltip with detailed reasoning
- Click to expand full judge details

### 2. Confidence Gauge

**Type:** Radial Gauge / Speedometer

**Library:** D3.js custom component

**Visual Features:**
- Arc from 0-100% confidence
- Color gradient: red (low) → yellow (medium) → green (high)
- Needle pointing to current confidence
- Threshold markers at 70% and 90%

### 3. Hallucination Thermometer

**Type:** Vertical Thermometer / Progress Bar

**Library:** Custom React component

**Visual Features:**
- Vertical bar from 0-100
- Color gradient: green (0) → yellow (50) → red (100)
- Threshold markers for severity levels
- Animated fill on load
- Breakdown pie chart on hover

### 4. Score Distribution

**Type:** Violin Plot or Box Plot

**Library:** Recharts or Plotly.js

**Visual Features:**
- Shows distribution of judge scores
- Median line and quartiles
- Outlier detection
- Comparison with historical distributions

### 5. Inter-Judge Agreement Heatmap

**Type:** Correlation Matrix Heatmap

**Library:** D3.js

**Visual Features:**
- Grid of judge pairs
- Color intensity based on correlation
- Diagonal shows perfect agreement (1.0)
- Hover shows exact correlation value

### 6. Hallucination Breakdown

**Type:** Stacked Bar Chart or Pie Chart

**Library:** Recharts

**Visual Features:**
- Shows proportion of each hallucination type
- Color-coded by severity
- Interactive legend to filter types
- Drill-down to see specific instances

## Error Handling

### Frontend Error Handling

```typescript
// Error boundary for React components
class EvaluationErrorBoundary extends React.Component {
  componentDidCatch(error, errorInfo) {
    // Log to error tracking service (e.g., Sentry)
    logErrorToService(error, errorInfo);
    
    // Show user-friendly error message
    this.setState({ hasError: true, error });
  }
}

// WebSocket error handling
socket.on('error', (error) => {
  if (error.type === 'rate_limit') {
    showNotification('Rate limit reached. Please wait before retrying.');
  } else if (error.type === 'authentication') {
    redirectToLogin();
  } else {
    showNotification('An error occurred. Please try again.');
  }
});
```

### Backend Error Handling

```python
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id
        }
    )

# Specific error handlers
class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    pass

class JudgeTimeoutError(EvaluationError):
    """Raised when a judge times out."""
    pass

class InsufficientJudgesError(EvaluationError):
    """Raised when not enough judges are available."""
    pass
```

## Testing Strategy

### Frontend Testing

**Unit Tests (Jest + React Testing Library):**
- Component rendering
- User interactions
- State management
- Utility functions

**Integration Tests:**
- API integration
- WebSocket communication
- End-to-end user flows

**Visual Regression Tests (Chromatic):**
- Component visual consistency
- Responsive design
- Chart rendering

### Backend Testing

**Unit Tests (pytest):**
- API endpoint logic
- Metric calculations
- Database operations
- WebSocket handlers

**Integration Tests:**
- Full evaluation pipeline
- Database transactions
- Authentication flow

**Load Tests (Locust):**
- Concurrent evaluations
- WebSocket scalability
- Database performance

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                │
└────────────┬────────────────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼─────┐ ┌────▼──────┐
│  Frontend │ │  Backend  │
│  (React)  │ │ (FastAPI) │
│  Container│ │ Container │
└───────────┘ └─────┬─────┘
                    │
              ┌─────┴─────┐
              │           │
        ┌─────▼─────┐ ┌──▼────────┐
        │PostgreSQL │ │   Redis   │
        │ Container │ │ Container │
        └───────────┘ └───────────┘
```

**Docker Compose Configuration:**
- Frontend: Nginx serving React build
- Backend: Gunicorn + Uvicorn workers
- Database: PostgreSQL with persistent volume
- Cache: Redis for session storage
- Reverse Proxy: Nginx for SSL termination

## Performance Optimization

### Frontend Optimizations

1. **Code Splitting:** Lazy load routes and heavy components
2. **Memoization:** Use React.memo for expensive components
3. **Virtual Scrolling:** For long history lists
4. **Debouncing:** For search and filter inputs
5. **Canvas Rendering:** For complex visualizations

### Backend Optimizations

1. **Connection Pooling:** PostgreSQL connection pool
2. **Caching:** Redis for frequently accessed data
3. **Async Processing:** Background tasks for heavy computations
4. **Database Indexing:** Optimize query performance
5. **Response Compression:** Gzip compression for API responses

## Security Considerations

1. **Authentication:** JWT with refresh tokens
2. **Authorization:** Role-based access control
3. **Input Validation:** Pydantic models for all inputs
4. **SQL Injection Prevention:** SQLAlchemy ORM
5. **XSS Prevention:** React's built-in escaping
6. **CSRF Protection:** CSRF tokens for state-changing operations
7. **Rate Limiting:** Per-user and per-IP rate limits
8. **HTTPS Only:** Force SSL in production
9. **Secrets Management:** Environment variables, never in code
10. **Audit Logging:** Log all evaluation requests and user actions

