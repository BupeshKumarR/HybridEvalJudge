# Integration Guide

This document describes how all components of the LLM Judge Auditor Web Application are integrated and how to verify the integration.

## Architecture Overview

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

## Component Integration Points

### 1. Frontend ↔ Backend API

**Integration Method:** REST API over HTTP/HTTPS

**Key Files:**
- Frontend: `web-app/frontend/src/api/client.ts`
- Backend: `web-app/backend/app/main.py`

**Configuration:**
- Frontend API URL: `REACT_APP_API_BASE_URL` (default: `http://localhost:8000/api/v1`)
- Backend CORS: Configured in `app/main.py` to allow frontend origin

**Endpoints:**
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/evaluations` - Create evaluation
- `GET /api/v1/evaluations/{id}` - Get evaluation results
- `GET /api/v1/evaluations` - List evaluation history
- `GET /api/v1/evaluations/{id}/export` - Export results
- `GET /api/v1/preferences` - Get user preferences
- `PUT /api/v1/preferences` - Update user preferences

**Authentication:**
- JWT tokens stored in frontend auth store
- Tokens sent in `Authorization: Bearer <token>` header
- Token refresh handled automatically by axios interceptors

### 2. Frontend ↔ Backend WebSocket

**Integration Method:** Socket.IO over WebSocket

**Key Files:**
- Frontend: `web-app/frontend/src/hooks/useWebSocket.ts`
- Backend: `web-app/backend/app/websocket.py`

**Configuration:**
- Frontend WS URL: `REACT_APP_WS_URL` (default: `http://localhost:8000`)
- WebSocket path: `/ws/socket.io/`

**Events:**
- Client → Server: `start_evaluation`
- Server → Client: `evaluation_progress`, `judge_result`, `evaluation_complete`, `evaluation_error`

**Connection Flow:**
1. Frontend establishes WebSocket connection on app load
2. Authentication token sent during connection handshake
3. Server validates token and creates authenticated session
4. Real-time events streamed during evaluation

### 3. Backend ↔ Database

**Integration Method:** SQLAlchemy ORM with PostgreSQL

**Key Files:**
- Models: `web-app/backend/app/models.py`
- Database config: `web-app/backend/app/database.py`
- Migrations: `web-app/backend/alembic/versions/`

**Configuration:**
- Database URL: `DATABASE_URL` environment variable
- Connection pooling: Configured in `database.py`

**Tables:**
- `users` - User accounts
- `evaluation_sessions` - Evaluation records
- `judge_results` - Individual judge verdicts
- `flagged_issues` - Issues detected by judges
- `verifier_verdicts` - Claim verification results
- `session_metadata` - Additional evaluation metadata
- `user_preferences` - User configuration
- `audit_logs` - Security audit trail

### 4. Backend ↔ Redis Cache

**Integration Method:** Redis client with connection pooling

**Key Files:**
- Cache config: `web-app/backend/app/cache.py`

**Configuration:**
- Redis URL: `REDIS_URL` environment variable
- Cache TTL: Configurable per cache key

**Cached Data:**
- Session data
- User preferences
- Frequently accessed evaluation results

### 5. Backend ↔ LLM Judge Auditor Core

**Integration Method:** Python package import

**Key Files:**
- Service: `web-app/backend/app/services/evaluation_service.py`
- Core package: `src/llm_judge_auditor/`

**Integration:**
- Backend imports `llm_judge_auditor` package
- Evaluation service wraps core functionality
- Streaming evaluator used for real-time updates

## User Flow Integration

### Flow 1: User Registration and Login

```
User → Frontend (LoginPage)
  → POST /api/v1/auth/register
    → Backend validates input
      → Creates user in database
        → Returns JWT token
          → Frontend stores token in auth store
            → Redirects to ChatPage
```

### Flow 2: Submit Evaluation

```
User → Frontend (ChatInputForm)
  → POST /api/v1/evaluations
    → Backend creates session
      → Returns session_id
        → Frontend connects WebSocket
          → Backend starts evaluation
            → Streams progress events
              → Frontend updates UI in real-time
                → Backend saves results to database
                  → Frontend displays final results
```

### Flow 3: View History

```
User → Frontend (HistorySidebar)
  → GET /api/v1/evaluations
    → Backend queries database
      → Returns paginated sessions
        → Frontend displays list
          → User clicks session
            → GET /api/v1/evaluations/{id}
              → Backend retrieves full results
                → Frontend restores session
```

### Flow 4: Export Results

```
User → Frontend (ExportMenu)
  → GET /api/v1/evaluations/{id}/export?format=pdf
    → Backend generates PDF
      → Returns file download
        → Frontend triggers browser download
```

## Testing Integration

### Manual Testing Checklist

- [ ] User can register a new account
- [ ] User can log in with credentials
- [ ] User can submit an evaluation
- [ ] Real-time progress updates appear during evaluation
- [ ] Evaluation results display correctly
- [ ] Visualizations render properly
- [ ] User can view evaluation history
- [ ] User can restore previous sessions
- [ ] User can export results (JSON, CSV, PDF)
- [ ] User preferences persist across sessions
- [ ] WebSocket reconnects after disconnection
- [ ] Error messages display appropriately
- [ ] Loading states show during async operations

### Automated Testing

Run the integration test script:

```bash
cd web-app
./scripts/test-integration.sh
```

This script tests:
- Backend health endpoints
- Database connectivity
- Redis cache connectivity
- User registration and login
- Evaluation creation and retrieval
- History listing
- Preferences management
- Export functionality
- Frontend accessibility
- CORS configuration
- Error handling

### End-to-End Testing

Run the E2E test suite:

```bash
cd web-app/backend
pytest tests/e2e/test_user_workflows.py -v
```

## Common Integration Issues

### Issue 1: CORS Errors

**Symptom:** Browser console shows CORS policy errors

**Solution:**
1. Check `CORS_ORIGINS` environment variable in backend
2. Ensure frontend URL is in allowed origins list
3. Verify CORS middleware is configured in `app/main.py`

### Issue 2: WebSocket Connection Fails

**Symptom:** "WebSocket connection failed" message in frontend

**Solution:**
1. Check `REACT_APP_WS_URL` environment variable
2. Verify backend WebSocket server is running
3. Check firewall/proxy settings
4. Ensure Socket.IO versions match between frontend and backend

### Issue 3: Authentication Token Expired

**Symptom:** 401 Unauthorized errors after some time

**Solution:**
1. Check JWT token expiration time in backend
2. Implement token refresh mechanism
3. Clear browser storage and re-login

### Issue 4: Database Connection Pool Exhausted

**Symptom:** "Too many connections" errors

**Solution:**
1. Increase connection pool size in `database.py`
2. Check for connection leaks (unclosed sessions)
3. Implement connection pooling best practices

### Issue 5: Redis Cache Unavailable

**Symptom:** Cache-related warnings in logs

**Solution:**
1. Verify Redis container is running
2. Check `REDIS_URL` environment variable
3. Application should gracefully degrade without cache

## Environment Configuration

### Development Environment

**Frontend (.env):**
```
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=http://localhost:8000
NODE_ENV=development
```

**Backend (.env):**
```
DATABASE_URL=postgresql://llm_judge_user:changeme@localhost:5432/llm_judge_auditor
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=dev-secret-key-change-in-production
ENVIRONMENT=development
LOG_LEVEL=info
CORS_ORIGINS=http://localhost:3000,http://localhost:80
```

### Production Environment

**Frontend:**
```
REACT_APP_API_BASE_URL=https://api.yourdomain.com/api/v1
REACT_APP_WS_URL=https://api.yourdomain.com
NODE_ENV=production
```

**Backend:**
```
DATABASE_URL=postgresql://user:password@db-host:5432/dbname
REDIS_URL=redis://redis-host:6379/0
SECRET_KEY=<strong-random-secret>
ENVIRONMENT=production
LOG_LEVEL=warning
CORS_ORIGINS=https://yourdomain.com
```

## Deployment Integration

### Docker Compose

All services are integrated via Docker Compose:

```bash
cd web-app
docker-compose up -d
```

Services:
- `postgres` - Database (port 5432)
- `redis` - Cache (port 6379)
- `backend` - API server (port 8000)
- `frontend` - React app (port 3000)
- `nginx` - Reverse proxy (port 80/443)

### Service Dependencies

```
nginx → frontend → backend → postgres
                          → redis
```

Health checks ensure services start in correct order.

## Monitoring Integration

### Health Endpoints

- Backend: `http://localhost:8000/health`
- Detailed: `http://localhost:8000/health/detailed`
- Metrics: `http://localhost:8000/metrics`

### Logging

- Backend logs: JSON format with request IDs
- Frontend logs: Browser console (development)
- Audit logs: Database table for security events

### Error Tracking

- Backend: Sentry integration (optional)
- Frontend: Error boundaries with logging

## Security Integration

### Authentication Flow

1. User submits credentials
2. Backend validates against database
3. JWT token generated with user claims
4. Token stored in frontend auth store
5. Token sent with all API requests
6. Backend validates token on each request

### Authorization

- Role-based access control (RBAC)
- User can only access their own evaluations
- Admin users have elevated permissions

### Security Headers

- CSP (Content Security Policy)
- HSTS (HTTP Strict Transport Security)
- X-Frame-Options
- X-Content-Type-Options

### Rate Limiting

- Per-user rate limits
- Per-IP rate limits
- Configurable thresholds

## Performance Optimization

### Frontend

- Code splitting by route
- React.memo for expensive components
- Virtual scrolling for long lists
- Debounced search inputs
- Lazy loading of visualizations

### Backend

- Database connection pooling
- Redis caching layer
- Query optimization with indexes
- Response compression (gzip)
- Async request handling

### Database

- Proper indexing on frequently queried columns
- Connection pooling
- Query optimization
- Periodic cleanup of old data

## Troubleshooting

### Check Service Status

```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# Database
docker exec llm-judge-postgres pg_isready

# Redis
docker exec llm-judge-redis redis-cli ping
```

### View Logs

```bash
# Backend logs
docker logs llm-judge-backend

# Frontend logs
docker logs llm-judge-frontend

# Database logs
docker logs llm-judge-postgres

# All services
docker-compose logs -f
```

### Reset Environment

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Rebuild and restart
docker-compose up --build -d
```

## Next Steps

1. Run integration tests: `./scripts/test-integration.sh`
2. Perform manual testing of all user flows
3. Review logs for any errors or warnings
4. Test with production-like data volumes
5. Conduct security audit
6. Perform load testing
7. Document any custom configurations
8. Create runbook for common issues
