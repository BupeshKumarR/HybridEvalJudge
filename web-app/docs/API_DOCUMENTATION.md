# API Documentation

## Overview

The LLM Judge Auditor Web Application provides a RESTful API for managing evaluations, user authentication, and session history. Real-time updates are delivered via WebSocket connections.

**Base URL:** `http://localhost:8000/api/v1`

**WebSocket URL:** `ws://localhost:8000/ws`

## Table of Contents

- [Authentication](#authentication)
- [REST API Endpoints](#rest-api-endpoints)
  - [Authentication Endpoints](#authentication-endpoints)
  - [Evaluation Endpoints](#evaluation-endpoints)
  - [Preferences Endpoints](#preferences-endpoints)
- [WebSocket Events](#websocket-events)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Authentication

The API uses JWT (JSON Web Token) for authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your_jwt_token>
```

Tokens expire after 24 hours. Refresh tokens are not currently implemented.

---

## REST API Endpoints

### Authentication Endpoints

#### POST /api/v1/auth/register

Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

**Response (201 Created):**
```json
{
  "id": "uuid",
  "username": "string",
  "email": "string",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request` - Invalid input data
- `409 Conflict` - Username or email already exists

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "SecurePass123!"
  }'
```

---

#### POST /api/v1/auth/login

Authenticate and receive a JWT token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string"
  }
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid credentials

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "SecurePass123!"
  }'
```

---

#### GET /api/v1/auth/me

Get current authenticated user information.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "id": "uuid",
  "username": "string",
  "email": "string",
  "created_at": "2024-01-15T10:30:00Z",
  "last_login": "2024-01-20T14:22:00Z"
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid or expired token

---

### Evaluation Endpoints

#### POST /api/v1/evaluations

Create a new evaluation session.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "source_text": "string",
  "candidate_output": "string",
  "config": {
    "judge_models": ["gpt-4", "claude-3"],
    "enable_retrieval": true,
    "aggregation_strategy": "weighted_average"
  }
}
```

**Response (201 Created):**
```json
{
  "session_id": "uuid",
  "status": "pending",
  "websocket_url": "ws://localhost:8000/ws?session_id=uuid",
  "created_at": "2024-01-20T15:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request` - Invalid input data
- `401 Unauthorized` - Missing or invalid token
- `422 Unprocessable Entity` - Validation error

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/evaluations \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "The capital of France is Paris.",
    "candidate_output": "Paris is the capital and largest city of France.",
    "config": {
      "judge_models": ["gpt-4"],
      "enable_retrieval": true,
      "aggregation_strategy": "mean"
    }
  }'
```

---

#### GET /api/v1/evaluations/{session_id}

Get evaluation results for a specific session.

**Headers:**
```
Authorization: Bearer <token>
```

**Path Parameters:**
- `session_id` (uuid) - The evaluation session ID

**Response (200 OK):**
```json
{
  "session_id": "uuid",
  "user_id": "uuid",
  "source_text": "string",
  "candidate_output": "string",
  "status": "completed",
  "consensus_score": 85.5,
  "hallucination_score": 12.3,
  "confidence_interval_lower": 82.1,
  "confidence_interval_upper": 88.9,
  "inter_judge_agreement": 0.78,
  "judge_results": [
    {
      "judge_name": "gpt-4",
      "score": 87.0,
      "confidence": 0.92,
      "reasoning": "The output accurately reflects the source...",
      "flagged_issues": []
    }
  ],
  "verifier_verdicts": [
    {
      "claim_text": "Paris is the capital of France",
      "label": "SUPPORTED",
      "confidence": 0.98,
      "evidence": ["Wikipedia: Paris"],
      "reasoning": "Claim is factually correct"
    }
  ],
  "metrics": {
    "variance": 2.5,
    "standard_deviation": 1.58,
    "processing_time_ms": 3450
  },
  "created_at": "2024-01-20T15:30:00Z",
  "completed_at": "2024-01-20T15:30:45Z"
}
```

**Error Responses:**
- `401 Unauthorized` - Missing or invalid token
- `404 Not Found` - Session not found
- `403 Forbidden` - Session belongs to another user

---

#### GET /api/v1/evaluations

List evaluation sessions for the authenticated user.

**Headers:**
```
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (integer, default: 1) - Page number
- `limit` (integer, default: 20, max: 100) - Items per page
- `sort_by` (string, default: "created_at") - Sort field (created_at, consensus_score)
- `order` (string, default: "desc") - Sort order (asc, desc)
- `min_score` (float, optional) - Filter by minimum consensus score
- `max_score` (float, optional) - Filter by maximum consensus score
- `search` (string, optional) - Search in source text or candidate output

**Response (200 OK):**
```json
{
  "sessions": [
    {
      "id": "uuid",
      "source_text_preview": "The capital of France...",
      "candidate_output_preview": "Paris is the capital...",
      "consensus_score": 85.5,
      "hallucination_score": 12.3,
      "status": "completed",
      "created_at": "2024-01-20T15:30:00Z"
    }
  ],
  "total": 42,
  "page": 1,
  "limit": 20,
  "has_more": true
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/api/v1/evaluations?page=1&limit=10&sort_by=consensus_score&order=desc" \
  -H "Authorization: Bearer <token>"
```

---

#### GET /api/v1/evaluations/{session_id}/export

Export evaluation results in various formats.

**Headers:**
```
Authorization: Bearer <token>
```

**Path Parameters:**
- `session_id` (uuid) - The evaluation session ID

**Query Parameters:**
- `format` (string, required) - Export format: `json`, `csv`, or `pdf`

**Response:**
- Content-Type varies based on format
- `application/json` for JSON
- `text/csv` for CSV
- `application/pdf` for PDF

**Example:**
```bash
# Export as JSON
curl -X GET "http://localhost:8000/api/v1/evaluations/{session_id}/export?format=json" \
  -H "Authorization: Bearer <token>" \
  -o evaluation.json

# Export as PDF
curl -X GET "http://localhost:8000/api/v1/evaluations/{session_id}/export?format=pdf" \
  -H "Authorization: Bearer <token>" \
  -o evaluation.pdf
```

---

#### DELETE /api/v1/evaluations/{session_id}

Delete an evaluation session.

**Headers:**
```
Authorization: Bearer <token>
```

**Path Parameters:**
- `session_id` (uuid) - The evaluation session ID

**Response (204 No Content)**

**Error Responses:**
- `401 Unauthorized` - Missing or invalid token
- `404 Not Found` - Session not found
- `403 Forbidden` - Session belongs to another user

---

### Preferences Endpoints

#### GET /api/v1/preferences

Get user preferences.

**Headers:**
```
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "user_id": "uuid",
  "default_judge_models": ["gpt-4", "claude-3"],
  "default_retrieval_enabled": true,
  "default_aggregation_strategy": "weighted_average",
  "theme": "dark",
  "notifications_enabled": true
}
```

---

#### PUT /api/v1/preferences

Update user preferences.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "default_judge_models": ["gpt-4"],
  "default_retrieval_enabled": false,
  "default_aggregation_strategy": "mean",
  "theme": "light"
}
```

**Response (200 OK):**
```json
{
  "user_id": "uuid",
  "default_judge_models": ["gpt-4"],
  "default_retrieval_enabled": false,
  "default_aggregation_strategy": "mean",
  "theme": "light",
  "notifications_enabled": true
}
```

---

## WebSocket Events

### Connection

Connect to the WebSocket server with authentication:

```javascript
const socket = io('ws://localhost:8000/ws', {
  auth: {
    token: 'your_jwt_token'
  }
});
```

### Client → Server Events

#### start_evaluation

Start a new evaluation with real-time streaming.

**Payload:**
```json
{
  "session_id": "uuid",
  "source_text": "string",
  "candidate_output": "string",
  "config": {
    "judge_models": ["gpt-4"],
    "enable_retrieval": true,
    "aggregation_strategy": "mean"
  }
}
```

**Example:**
```javascript
socket.emit('start_evaluation', {
  session_id: sessionId,
  source_text: 'The capital of France is Paris.',
  candidate_output: 'Paris is the capital of France.',
  config: {
    judge_models: ['gpt-4'],
    enable_retrieval: true,
    aggregation_strategy: 'mean'
  }
});
```

---

### Server → Client Events

#### evaluation_progress

Sent periodically during evaluation to show progress.

**Payload:**
```json
{
  "session_id": "uuid",
  "stage": "retrieval",
  "progress": 25.0,
  "message": "Retrieving relevant passages..."
}
```

**Stages:**
- `retrieval` - Fetching relevant documents
- `verification` - Verifying claims
- `judging` - Running judge models
- `aggregation` - Computing final scores

**Example Handler:**
```javascript
socket.on('evaluation_progress', (data) => {
  console.log(`${data.stage}: ${data.progress}% - ${data.message}`);
  updateProgressBar(data.progress);
});
```

---

#### judge_result

Sent when an individual judge completes evaluation.

**Payload:**
```json
{
  "session_id": "uuid",
  "judge_name": "gpt-4",
  "score": 87.0,
  "confidence": 0.92,
  "reasoning": "The output accurately reflects the source information...",
  "flagged_issues": [
    {
      "type": "minor_inconsistency",
      "severity": "low",
      "description": "Minor wording difference",
      "text_span": [10, 25]
    }
  ],
  "response_time_ms": 1250
}
```

**Example Handler:**
```javascript
socket.on('judge_result', (data) => {
  console.log(`${data.judge_name} scored: ${data.score}`);
  addJudgeResultToUI(data);
});
```

---

#### verifier_verdict

Sent when a claim is verified.

**Payload:**
```json
{
  "session_id": "uuid",
  "claim_text": "Paris is the capital of France",
  "label": "SUPPORTED",
  "confidence": 0.98,
  "evidence": ["Wikipedia: Paris is the capital and most populous city of France"],
  "reasoning": "The claim is factually correct and well-supported"
}
```

**Labels:**
- `SUPPORTED` - Claim is verified as true
- `REFUTED` - Claim is verified as false
- `NOT_ENOUGH_INFO` - Insufficient evidence to verify

---

#### evaluation_complete

Sent when the entire evaluation is finished.

**Payload:**
```json
{
  "session_id": "uuid",
  "consensus_score": 85.5,
  "hallucination_score": 12.3,
  "confidence_interval": [82.1, 88.9],
  "inter_judge_agreement": 0.78,
  "metrics": {
    "variance": 2.5,
    "standard_deviation": 1.58,
    "processing_time_ms": 3450
  },
  "full_results": {
    // Complete evaluation results object
  }
}
```

**Example Handler:**
```javascript
socket.on('evaluation_complete', (data) => {
  console.log('Evaluation complete!');
  displayFinalResults(data);
});
```

---

#### evaluation_error

Sent when an error occurs during evaluation.

**Payload:**
```json
{
  "session_id": "uuid",
  "error_type": "judge_timeout",
  "message": "Judge 'gpt-4' timed out after 30 seconds",
  "recovery_suggestions": [
    "Try again with a shorter input",
    "Use a different judge model",
    "Check your API key configuration"
  ]
}
```

**Error Types:**
- `judge_timeout` - Judge model took too long
- `judge_error` - Judge model returned an error
- `retrieval_error` - Failed to retrieve documents
- `verification_error` - Verifier failed
- `authentication_error` - Invalid or expired token
- `rate_limit_error` - Too many requests

**Example Handler:**
```javascript
socket.on('evaluation_error', (data) => {
  console.error(`Error: ${data.message}`);
  showErrorNotification(data.message, data.recovery_suggestions);
});
```

---

## Data Models

### EvaluationSession

```typescript
interface EvaluationSession {
  id: string;
  user_id: string;
  source_text: string;
  candidate_output: string;
  consensus_score: number | null;
  hallucination_score: number | null;
  confidence_interval_lower: number | null;
  confidence_interval_upper: number | null;
  inter_judge_agreement: number | null;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  config: EvaluationConfig;
  created_at: string;
  completed_at: string | null;
}
```

### EvaluationConfig

```typescript
interface EvaluationConfig {
  judge_models: string[];
  enable_retrieval: boolean;
  aggregation_strategy: 'mean' | 'median' | 'weighted_average';
}
```

### JudgeResult

```typescript
interface JudgeResult {
  id: string;
  session_id: string;
  judge_name: string;
  score: number;
  confidence: number;
  reasoning: string;
  flagged_issues: FlaggedIssue[];
  response_time_ms: number;
  created_at: string;
}
```

### FlaggedIssue

```typescript
interface FlaggedIssue {
  id: string;
  type: 'factual_error' | 'hallucination' | 'unsupported_claim' | 
        'temporal_inconsistency' | 'numerical_error' | 'bias';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  evidence: any;
  text_span_start: number | null;
  text_span_end: number | null;
}
```

### VerifierVerdict

```typescript
interface VerifierVerdict {
  id: string;
  session_id: string;
  claim_text: string;
  label: 'SUPPORTED' | 'REFUTED' | 'NOT_ENOUGH_INFO';
  confidence: number;
  evidence: any;
  reasoning: string;
}
```

---

## Error Handling

All API errors follow a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    // Additional error context
  },
  "request_id": "uuid"
}
```

### Common HTTP Status Codes

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `204 No Content` - Request succeeded with no response body
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict (e.g., duplicate username)
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service temporarily unavailable

### Error Examples

**Validation Error (422):**
```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": {
    "source_text": ["Field is required"],
    "config.judge_models": ["Must contain at least one judge model"]
  },
  "request_id": "abc-123"
}
```

**Authentication Error (401):**
```json
{
  "error": "authentication_error",
  "message": "Invalid or expired token",
  "details": {
    "token_expired": true,
    "expired_at": "2024-01-20T10:00:00Z"
  },
  "request_id": "def-456"
}
```

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Authenticated requests:** 100 requests per minute per user
- **Unauthenticated requests:** 20 requests per minute per IP
- **Evaluation creation:** 10 evaluations per minute per user

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705756800
```

When rate limit is exceeded, the API returns `429 Too Many Requests`:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please try again later.",
  "details": {
    "retry_after": 60
  },
  "request_id": "ghi-789"
}
```

---

## Best Practices

1. **Always handle errors gracefully** - Check status codes and handle error responses
2. **Use WebSocket for real-time updates** - Don't poll the REST API for evaluation status
3. **Implement exponential backoff** - When retrying failed requests
4. **Store tokens securely** - Never expose JWT tokens in client-side code
5. **Validate input before sending** - Reduce unnecessary API calls
6. **Use pagination** - When fetching large lists of evaluations
7. **Close WebSocket connections** - When no longer needed to free resources

---

## Support

For API issues or questions:
- GitHub Issues: https://github.com/your-org/llm-judge-auditor/issues
- Documentation: https://docs.llm-judge-auditor.com
- Email: support@llm-judge-auditor.com
