# WebSocket Events Documentation

## Overview

The LLM Judge Auditor uses WebSocket connections for real-time streaming of evaluation progress and results. This document describes all WebSocket events, their payloads, and usage examples.

**WebSocket URL:** `ws://localhost:8000/ws`

## Table of Contents

- [Connection](#connection)
- [Authentication](#authentication)
- [Client to Server Events](#client-to-server-events)
- [Server to Client Events](#server-to-client-events)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Code Examples](#code-examples)

---

## Connection

### Establishing Connection

Connect to the WebSocket server using Socket.IO client:

```javascript
import { io } from 'socket.io-client';

const socket = io('ws://localhost:8000/ws', {
  auth: {
    token: 'your_jwt_token_here'
  },
  transports: ['websocket'],
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionAttempts: 5
});
```

### Connection Events

```javascript
socket.on('connect', () => {
  console.log('Connected to WebSocket server');
  console.log('Socket ID:', socket.id);
});

socket.on('disconnect', (reason) => {
  console.log('Disconnected:', reason);
  if (reason === 'io server disconnect') {
    // Server disconnected, manually reconnect
    socket.connect();
  }
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error.message);
});
```

---

## Authentication

Authentication is handled during connection establishment via the `auth` parameter. The JWT token must be valid and not expired.

### Authentication Errors

If authentication fails, the connection will be rejected:

```javascript
socket.on('connect_error', (error) => {
  if (error.message === 'Authentication failed') {
    // Redirect to login or refresh token
    console.error('Invalid or expired token');
  }
});
```

---

## Client to Server Events

### start_evaluation

Initiates a new evaluation with real-time streaming updates.

**Event Name:** `start_evaluation`

**Payload:**
```typescript
interface StartEvaluationPayload {
  session_id: string;
  source_text: string;
  candidate_output: string;
  config: {
    judge_models: string[];
    enable_retrieval: boolean;
    aggregation_strategy: 'mean' | 'median' | 'weighted_average';
  };
}
```

**Example:**
```javascript
socket.emit('start_evaluation', {
  session_id: '550e8400-e29b-41d4-a716-446655440000',
  source_text: 'The capital of France is Paris.',
  candidate_output: 'Paris is the capital and largest city of France.',
  config: {
    judge_models: ['gpt-4', 'claude-3'],
    enable_retrieval: true,
    aggregation_strategy: 'mean'
  }
});
```

**Response:** The server will emit a series of events (`evaluation_progress`, `judge_result`, etc.) as the evaluation progresses.

---

### cancel_evaluation

Cancels an ongoing evaluation.

**Event Name:** `cancel_evaluation`

**Payload:**
```typescript
interface CancelEvaluationPayload {
  session_id: string;
}
```

**Example:**
```javascript
socket.emit('cancel_evaluation', {
  session_id: '550e8400-e29b-41d4-a716-446655440000'
});
```

---

## Server to Client Events

### evaluation_progress

Sent periodically during evaluation to indicate progress.

**Event Name:** `evaluation_progress`

**Payload:**
```typescript
interface EvaluationProgressPayload {
  session_id: string;
  stage: 'retrieval' | 'verification' | 'judging' | 'aggregation';
  progress: number; // 0-100
  message: string;
  timestamp: string;
}
```

**Stages:**
- `retrieval` - Fetching relevant documents from knowledge base
- `verification` - Verifying claims against evidence
- `judging` - Running judge models to evaluate output
- `aggregation` - Computing final consensus scores

**Example:**
```javascript
socket.on('evaluation_progress', (data) => {
  console.log(`[${data.stage}] ${data.progress}%: ${data.message}`);
  
  // Update UI progress bar
  updateProgressBar(data.progress);
  updateStatusMessage(data.message);
});
```

**Sample Payloads:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "stage": "retrieval",
  "progress": 15.0,
  "message": "Retrieving relevant passages from knowledge base...",
  "timestamp": "2024-01-20T15:30:10Z"
}

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "stage": "judging",
  "progress": 60.0,
  "message": "Running judge model: gpt-4...",
  "timestamp": "2024-01-20T15:30:25Z"
}
```

---

### judge_result

Sent when an individual judge completes its evaluation.

**Event Name:** `judge_result`

**Payload:**
```typescript
interface JudgeResultPayload {
  session_id: string;
  judge_name: string;
  score: number; // 0-100
  confidence: number; // 0-1
  reasoning: string;
  flagged_issues: FlaggedIssue[];
  response_time_ms: number;
  timestamp: string;
}

interface FlaggedIssue {
  type: 'factual_error' | 'hallucination' | 'unsupported_claim' | 
        'temporal_inconsistency' | 'numerical_error' | 'bias';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  text_span?: [number, number]; // [start, end] indices
}
```

**Example:**
```javascript
socket.on('judge_result', (data) => {
  console.log(`Judge ${data.judge_name} scored: ${data.score}`);
  console.log(`Confidence: ${data.confidence}`);
  console.log(`Reasoning: ${data.reasoning}`);
  
  // Add judge result to UI
  addJudgeCard({
    name: data.judge_name,
    score: data.score,
    confidence: data.confidence,
    reasoning: data.reasoning,
    issues: data.flagged_issues
  });
});
```

**Sample Payload:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "judge_name": "gpt-4",
  "score": 87.5,
  "confidence": 0.92,
  "reasoning": "The candidate output accurately reflects the source information. Paris is correctly identified as the capital of France. The additional detail about it being the largest city is factually correct and adds value.",
  "flagged_issues": [
    {
      "type": "minor_inconsistency",
      "severity": "low",
      "description": "Minor wording difference between source and output",
      "text_span": [0, 25]
    }
  ],
  "response_time_ms": 1250,
  "timestamp": "2024-01-20T15:30:28Z"
}
```

---

### verifier_verdict

Sent when a claim is verified against evidence.

**Event Name:** `verifier_verdict`

**Payload:**
```typescript
interface VerifierVerdictPayload {
  session_id: string;
  claim_text: string;
  label: 'SUPPORTED' | 'REFUTED' | 'NOT_ENOUGH_INFO';
  confidence: number; // 0-1
  evidence: string[];
  reasoning: string;
  timestamp: string;
}
```

**Labels:**
- `SUPPORTED` - Claim is verified as factually correct
- `REFUTED` - Claim is verified as factually incorrect
- `NOT_ENOUGH_INFO` - Insufficient evidence to verify claim

**Example:**
```javascript
socket.on('verifier_verdict', (data) => {
  const icon = data.label === 'SUPPORTED' ? '✓' : 
               data.label === 'REFUTED' ? '✗' : '?';
  
  console.log(`${icon} ${data.claim_text}`);
  console.log(`Verdict: ${data.label} (${data.confidence})`);
  
  // Update UI with verification result
  addVerificationResult({
    claim: data.claim_text,
    verdict: data.label,
    confidence: data.confidence,
    evidence: data.evidence
  });
});
```

**Sample Payload:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "claim_text": "Paris is the capital of France",
  "label": "SUPPORTED",
  "confidence": 0.98,
  "evidence": [
    "Wikipedia: Paris is the capital and most populous city of France.",
    "Britannica: Paris, city and capital of France."
  ],
  "reasoning": "The claim is factually correct and well-supported by multiple authoritative sources.",
  "timestamp": "2024-01-20T15:30:20Z"
}
```

---

### evaluation_complete

Sent when the entire evaluation process is finished.

**Event Name:** `evaluation_complete`

**Payload:**
```typescript
interface EvaluationCompletePayload {
  session_id: string;
  consensus_score: number;
  hallucination_score: number;
  confidence_interval: [number, number];
  inter_judge_agreement: number;
  metrics: {
    variance: number;
    standard_deviation: number;
    processing_time_ms: number;
  };
  full_results: EvaluationDetail; // Complete evaluation object
  timestamp: string;
}
```

**Example:**
```javascript
socket.on('evaluation_complete', (data) => {
  console.log('Evaluation complete!');
  console.log(`Consensus Score: ${data.consensus_score}`);
  console.log(`Hallucination Score: ${data.hallucination_score}`);
  console.log(`Confidence Interval: [${data.confidence_interval[0]}, ${data.confidence_interval[1]}]`);
  console.log(`Inter-Judge Agreement: ${data.inter_judge_agreement}`);
  
  // Display final results
  displayFinalResults(data);
  
  // Enable export buttons
  enableExportOptions(data.session_id);
});
```

**Sample Payload:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
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
    // Complete evaluation object with all details
  },
  "timestamp": "2024-01-20T15:30:45Z"
}
```

---

### evaluation_error

Sent when an error occurs during evaluation.

**Event Name:** `evaluation_error`

**Payload:**
```typescript
interface EvaluationErrorPayload {
  session_id: string;
  error_type: 'judge_timeout' | 'judge_error' | 'retrieval_error' | 
              'verification_error' | 'authentication_error' | 'rate_limit_error';
  message: string;
  details?: any;
  recovery_suggestions: string[];
  timestamp: string;
}
```

**Error Types:**
- `judge_timeout` - Judge model exceeded timeout limit
- `judge_error` - Judge model returned an error
- `retrieval_error` - Failed to retrieve documents
- `verification_error` - Verifier failed to process claims
- `authentication_error` - Invalid or expired authentication
- `rate_limit_error` - Too many requests

**Example:**
```javascript
socket.on('evaluation_error', (data) => {
  console.error(`Error: ${data.message}`);
  console.error(`Type: ${data.error_type}`);
  
  // Show error notification with recovery suggestions
  showErrorNotification({
    title: 'Evaluation Failed',
    message: data.message,
    suggestions: data.recovery_suggestions
  });
  
  // Update UI to show error state
  setEvaluationStatus('failed');
});
```

**Sample Payloads:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "error_type": "judge_timeout",
  "message": "Judge 'gpt-4' timed out after 30 seconds",
  "details": {
    "judge_name": "gpt-4",
    "timeout_seconds": 30
  },
  "recovery_suggestions": [
    "Try again with a shorter input text",
    "Use a different judge model",
    "Check your API key configuration"
  ],
  "timestamp": "2024-01-20T15:31:00Z"
}

{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "error_type": "rate_limit_error",
  "message": "Rate limit exceeded. Please wait before retrying.",
  "details": {
    "retry_after": 60
  },
  "recovery_suggestions": [
    "Wait 60 seconds before retrying",
    "Reduce the number of concurrent evaluations"
  ],
  "timestamp": "2024-01-20T15:31:15Z"
}
```

---

## Error Handling

### Connection Errors

```javascript
socket.on('connect_error', (error) => {
  console.error('Connection error:', error.message);
  
  if (error.message === 'Authentication failed') {
    // Redirect to login
    window.location.href = '/login';
  } else {
    // Show reconnection message
    showNotification('Connection lost. Attempting to reconnect...');
  }
});
```

### Reconnection Logic

```javascript
socket.on('reconnect', (attemptNumber) => {
  console.log(`Reconnected after ${attemptNumber} attempts`);
  showNotification('Connection restored');
  
  // Optionally resume evaluation
  if (currentSessionId) {
    socket.emit('resume_evaluation', { session_id: currentSessionId });
  }
});

socket.on('reconnect_failed', () => {
  console.error('Failed to reconnect');
  showNotification('Unable to reconnect. Please refresh the page.');
});
```

---

## Best Practices

### 1. Always Handle Disconnections

```javascript
socket.on('disconnect', (reason) => {
  if (reason === 'io server disconnect') {
    // Server disconnected, manually reconnect
    socket.connect();
  }
  // Update UI to show disconnected state
  setConnectionStatus('disconnected');
});
```

### 2. Implement Timeout Handling

```javascript
let evaluationTimeout;

socket.emit('start_evaluation', evaluationData);

// Set timeout for evaluation
evaluationTimeout = setTimeout(() => {
  console.error('Evaluation timed out');
  showNotification('Evaluation is taking longer than expected');
}, 60000); // 60 seconds

socket.on('evaluation_complete', (data) => {
  clearTimeout(evaluationTimeout);
  // Handle completion
});
```

### 3. Clean Up Event Listeners

```javascript
useEffect(() => {
  const handleProgress = (data) => {
    updateProgress(data);
  };
  
  socket.on('evaluation_progress', handleProgress);
  
  // Cleanup on unmount
  return () => {
    socket.off('evaluation_progress', handleProgress);
  };
}, []);
```

### 4. Handle Multiple Concurrent Evaluations

```javascript
const activeEvaluations = new Map();

socket.on('judge_result', (data) => {
  const evaluation = activeEvaluations.get(data.session_id);
  if (evaluation) {
    evaluation.addJudgeResult(data);
  }
});

socket.on('evaluation_complete', (data) => {
  const evaluation = activeEvaluations.get(data.session_id);
  if (evaluation) {
    evaluation.complete(data);
    activeEvaluations.delete(data.session_id);
  }
});
```

### 5. Implement Exponential Backoff for Retries

```javascript
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

socket.on('connect_error', () => {
  reconnectAttempts++;
  
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error('Max reconnection attempts reached');
    showNotification('Unable to connect. Please check your connection.');
    return;
  }
  
  const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
  console.log(`Retrying in ${delay}ms...`);
  
  setTimeout(() => {
    socket.connect();
  }, delay);
});

socket.on('connect', () => {
  reconnectAttempts = 0; // Reset on successful connection
});
```

---

## Code Examples

### React Hook for WebSocket

```typescript
import { useEffect, useState } from 'react';
import { io, Socket } from 'socket.io-client';

interface UseWebSocketOptions {
  token: string;
  onProgress?: (data: any) => void;
  onJudgeResult?: (data: any) => void;
  onComplete?: (data: any) => void;
  onError?: (data: any) => void;
}

export function useWebSocket(options: UseWebSocketOptions) {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const newSocket = io('ws://localhost:8000/ws', {
      auth: { token: options.token },
      transports: ['websocket']
    });

    newSocket.on('connect', () => {
      console.log('Connected');
      setConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected');
      setConnected(false);
    });

    if (options.onProgress) {
      newSocket.on('evaluation_progress', options.onProgress);
    }

    if (options.onJudgeResult) {
      newSocket.on('judge_result', options.onJudgeResult);
    }

    if (options.onComplete) {
      newSocket.on('evaluation_complete', options.onComplete);
    }

    if (options.onError) {
      newSocket.on('evaluation_error', options.onError);
    }

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [options.token]);

  const startEvaluation = (data: any) => {
    if (socket && connected) {
      socket.emit('start_evaluation', data);
    }
  };

  return { socket, connected, startEvaluation };
}
```

### Usage in Component

```typescript
function EvaluationPage() {
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  
  const { connected, startEvaluation } = useWebSocket({
    token: getAuthToken(),
    onProgress: (data) => {
      setProgress(data.progress);
    },
    onJudgeResult: (data) => {
      console.log('Judge result:', data);
    },
    onComplete: (data) => {
      setResults(data);
    },
    onError: (data) => {
      alert(`Error: ${data.message}`);
    }
  });

  const handleSubmit = (sourceText: string, candidateOutput: string) => {
    startEvaluation({
      session_id: generateUUID(),
      source_text: sourceText,
      candidate_output: candidateOutput,
      config: {
        judge_models: ['gpt-4'],
        enable_retrieval: true,
        aggregation_strategy: 'mean'
      }
    });
  };

  return (
    <div>
      <ConnectionStatus connected={connected} />
      <ProgressBar value={progress} />
      {results && <ResultsDisplay results={results} />}
    </div>
  );
}
```

---

## Testing WebSocket Events

### Using Socket.IO Client for Testing

```javascript
const io = require('socket.io-client');

const socket = io('ws://localhost:8000/ws', {
  auth: { token: 'test_token' }
});

socket.on('connect', () => {
  console.log('Connected for testing');
  
  // Start test evaluation
  socket.emit('start_evaluation', {
    session_id: 'test-session-123',
    source_text: 'Test source',
    candidate_output: 'Test output',
    config: {
      judge_models: ['gpt-4'],
      enable_retrieval: false,
      aggregation_strategy: 'mean'
    }
  });
});

socket.on('evaluation_progress', (data) => {
  console.log('Progress:', data);
});

socket.on('evaluation_complete', (data) => {
  console.log('Complete:', data);
  socket.close();
});
```

---

## Support

For WebSocket issues:
- Check connection status in browser DevTools (Network tab)
- Verify JWT token is valid and not expired
- Ensure WebSocket port (8000) is not blocked by firewall
- Review server logs for connection errors

For additional help:
- GitHub Issues: https://github.com/your-org/llm-judge-auditor/issues
- Documentation: https://docs.llm-judge-auditor.com
