# WebSocket Implementation Summary

## Overview

This document summarizes the WebSocket implementation for real-time evaluation streaming in the LLM Judge Auditor web application.

## Components Implemented

### 1. WebSocket Server (`app/websocket.py`)

**Features:**
- Socket.IO server with async mode for FastAPI integration
- JWT-based authentication for WebSocket connections
- Room management for evaluation sessions
- Connection tracking and cleanup
- Event handlers for:
  - `connect`: Authenticate and establish connection
  - `disconnect`: Clean up connections and rooms
  - `join_session`: Join an evaluation session room
  - `leave_session`: Leave an evaluation session room
  - `start_evaluation`: Start evaluation processing via WebSocket
  - `ping`: Health check

**Helper Functions:**
- `authenticate_socket()`: Authenticate WebSocket connections using JWT
- `verify_session_access()`: Verify user has access to evaluation session
- `emit_evaluation_progress()`: Stream progress updates
- `emit_judge_result()`: Stream individual judge results
- `emit_evaluation_complete()`: Send completion event
- `emit_evaluation_error()`: Send error events with recovery suggestions
- `get_active_connections_count()`: Get number of active connections

### 2. Evaluation Service (`app/services/evaluation_service.py`)

**Features:**
- Asynchronous evaluation processing with streaming updates
- Multi-stage evaluation pipeline:
  1. Retrieval (optional)
  2. Claim verification
  3. Judge evaluation (parallel processing)
  4. Metrics aggregation
- Real-time progress streaming at each stage
- Comprehensive error handling with graceful degradation
- Database persistence of all results

**Key Methods:**
- `process_evaluation()`: Main evaluation orchestration
- `_simulate_retrieval()`: Retrieval simulation (placeholder for production)
- `_simulate_verification()`: Claim verification simulation
- `_simulate_judge_evaluation()`: Judge evaluation simulation
- `_calculate_metrics()`: Calculate consensus and confidence metrics
- `_calculate_hallucination_score()`: Calculate hallucination score using composite formula

**Metrics Calculated:**
- Consensus score (weighted average of judge scores)
- Hallucination score (composite of multiple factors)
- Confidence intervals (bootstrap method)
- Inter-judge agreement (variance-based)
- Statistical measures (variance, standard deviation)

### 3. API Integration

**New Endpoint:**
- `POST /api/v1/evaluations/{session_id}/start`: Start evaluation processing

**WebSocket Events (Server → Client):**
- `evaluation_progress`: Progress updates with stage, percentage, and message
- `judge_result`: Individual judge results as they complete
- `evaluation_complete`: Final results with all metrics
- `evaluation_error`: Error events with recovery suggestions
- `session_joined`: Confirmation of joining session room
- `session_left`: Confirmation of leaving session room
- `evaluation_started`: Confirmation that evaluation has started

**WebSocket Events (Client → Server):**
- `join_session`: Join an evaluation session room
- `leave_session`: Leave an evaluation session room
- `start_evaluation`: Start evaluation processing
- `ping`: Health check

### 4. Main Application Integration

**Changes to `app/main.py`:**
- Imported Socket.IO server and ASGI app
- Mounted Socket.IO app at `/ws` endpoint
- WebSocket accessible at `ws://localhost:8000/ws/socket.io`

## Authentication Flow

1. Client obtains JWT token via `/api/v1/auth/login`
2. Client connects to WebSocket with token in auth data:
   ```javascript
   const socket = io('http://localhost:8000/ws', {
     auth: { token: 'JWT_TOKEN_HERE' }
   });
   ```
3. Server validates token and establishes connection
4. Client joins evaluation session room
5. Client can start evaluation or receive updates

## Evaluation Flow

1. **Create Session**: Client creates evaluation via REST API
2. **Connect WebSocket**: Client connects and joins session room
3. **Start Evaluation**: Client triggers evaluation start
4. **Streaming Updates**: Server streams progress and results:
   - Retrieval progress (if enabled)
   - Verification progress
   - Individual judge results (as they complete)
   - Aggregation progress
   - Final completion event
5. **Error Handling**: Any errors are streamed with recovery suggestions
6. **Disconnection**: Graceful handling of client disconnections

## Testing

### Test Files Created:
- `tests/test_websocket.py`: WebSocket functionality tests
- `tests/test_evaluation_service.py`: Evaluation service tests

### Test Coverage:
- WebSocket authentication (valid/invalid/expired tokens)
- Session access verification
- Event emitters (progress, results, completion, errors)
- Active connection tracking
- Evaluation service methods (retrieval, verification, judge evaluation)
- Metrics calculation (consensus, hallucination, confidence)

### Known Test Issues:
- Some tests fail due to bcrypt password hashing compatibility issues in test fixtures
- Core WebSocket and evaluation service functionality tests pass successfully

## Configuration

### CORS Settings:
WebSocket server allows connections from:
- `http://localhost:3000` (React dev server)
- `http://localhost:80` (Production)
- `http://localhost:8000` (Backend)
- `http://127.0.0.1:*` (Alternative localhost)

### Dependencies Added:
- `python-socketio==5.10.0`: Socket.IO server
- `python-engineio==4.8.0`: Engine.IO (Socket.IO dependency)

## Usage Example

### Backend (Already Implemented):
```python
# Evaluation automatically streams updates via WebSocket
# when started through the API or WebSocket event
```

### Frontend (To Be Implemented):
```javascript
import io from 'socket.io-client';

// Connect with authentication
const socket = io('http://localhost:8000/ws', {
  auth: { token: accessToken }
});

// Join session room
socket.emit('join_session', { session_id: sessionId });

// Listen for progress updates
socket.on('evaluation_progress', (data) => {
  console.log(`Stage: ${data.stage}, Progress: ${data.progress}%`);
  console.log(`Message: ${data.message}`);
});

// Listen for judge results
socket.on('judge_result', (data) => {
  console.log(`Judge ${data.judge_name}: ${data.score}`);
});

// Listen for completion
socket.on('evaluation_complete', (data) => {
  console.log('Evaluation complete!', data);
});

// Listen for errors
socket.on('evaluation_error', (data) => {
  console.error('Error:', data.message);
  console.log('Suggestions:', data.recovery_suggestions);
});

// Start evaluation
socket.emit('start_evaluation', { session_id: sessionId });
```

## Next Steps

1. **Frontend Implementation** (Task 6.4): Implement Socket.IO client in React
2. **Integration Testing**: Test full WebSocket flow with frontend
3. **Production Integration**: Replace simulation methods with actual LLM Judge Auditor components
4. **Performance Testing**: Test with multiple concurrent evaluations
5. **Error Recovery**: Implement reconnection logic and state recovery

## Security Considerations

- JWT authentication required for all WebSocket connections
- Session access verification before joining rooms
- User can only access their own evaluation sessions
- Automatic cleanup of disconnected clients
- Rate limiting should be added for production

## Performance Considerations

- Async/await used throughout for non-blocking operations
- Background task execution for long-running evaluations
- Efficient room-based broadcasting (only to interested clients)
- Connection tracking for monitoring and cleanup
- Database connection pooling (already configured)

## Compliance with Requirements

### Requirement 2.1 ✅
"WHEN an evaluation starts THEN the system SHALL establish a WebSocket connection for streaming updates"
- Implemented: WebSocket server with authentication and room management

### Requirement 2.2 ✅
"WHEN each judge completes evaluation THEN the system SHALL stream the individual result immediately"
- Implemented: `emit_judge_result()` called after each judge completes

### Requirement 2.3 ✅
"WHEN the verifier processes claims THEN the system SHALL stream verification progress with claim-by-claim updates"
- Implemented: Progress updates during verification stage

### Requirement 2.4 ✅
"WHEN aggregation completes THEN the system SHALL stream the final consensus score"
- Implemented: `emit_evaluation_complete()` with all final metrics

### Requirement 2.5 ✅
"WHEN any error occurs THEN the system SHALL stream error details with recovery suggestions"
- Implemented: `emit_evaluation_error()` with error type, message, and suggestions
