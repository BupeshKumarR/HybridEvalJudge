# Performance Optimization Guide

This document provides guidelines and best practices for optimizing the performance of the LLM Judge Auditor Web Application.

## Performance Requirements

Based on the requirements document:

1. **Frontend Load Time**: < 2 seconds
2. **API Acknowledgment**: < 500ms
3. **WebSocket Updates**: < 100ms
4. **History Pagination**: 20 sessions per page
5. **Efficient Rendering**: Canvas-based charts for complex visualizations

## Performance Testing

### Running Performance Tests

```bash
# Run comprehensive performance tests
cd web-app
./scripts/performance-test.sh

# Run load tests with Locust
cd web-app/backend
pip install locust
locust -f locustfile.py --host=http://localhost:8000

# Access Locust web UI at http://localhost:8089
```

### Load Testing Scenarios

1. **Normal Load**: 10-50 concurrent users
2. **Peak Load**: 100-200 concurrent users
3. **Stress Test**: 500+ concurrent users
4. **Spike Test**: Sudden increase from 10 to 200 users

## Frontend Optimizations

### 1. Code Splitting

**Implementation:**
```typescript
// Lazy load routes
const ChatPage = lazy(() => import('../pages/ChatPage'));
const HistoryPage = lazy(() => import('../pages/HistoryPage'));
const SettingsPage = lazy(() => import('../pages/SettingsPage'));
```

**Benefits:**
- Reduces initial bundle size
- Faster initial page load
- Better caching strategy

### 2. Component Memoization

**Implementation:**
```typescript
// Memoize expensive components
const MemoizedVisualization = React.memo(VisualizationDashboard, (prevProps, nextProps) => {
  return prevProps.evaluationResults === nextProps.evaluationResults;
});
```

**When to use:**
- Components with expensive render logic
- Components that receive the same props frequently
- Pure components without side effects

### 3. Virtual Scrolling

**Implementation:**
```typescript
// Use react-window for long lists
import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={600}
  itemCount={sessions.length}
  itemSize={80}
  width="100%"
>
  {({ index, style }) => (
    <div style={style}>
      <SessionItem session={sessions[index]} />
    </div>
  )}
</FixedSizeList>
```

**Benefits:**
- Only renders visible items
- Handles thousands of items efficiently
- Smooth scrolling performance

### 4. Debouncing and Throttling

**Implementation:**
```typescript
// Debounce search input
const debouncedSearch = useMemo(
  () => debounce((query: string) => {
    performSearch(query);
  }, 300),
  []
);

// Throttle scroll events
const throttledScroll = useMemo(
  () => throttle(() => {
    handleScroll();
  }, 100),
  []
);
```

**Use cases:**
- Search inputs
- Scroll events
- Window resize handlers
- API calls triggered by user input

### 5. Image and Asset Optimization

**Best practices:**
- Use WebP format for images
- Implement lazy loading for images
- Use appropriate image sizes
- Compress assets during build
- Use CDN for static assets

### 6. Bundle Size Optimization

**Techniques:**
```json
// package.json - analyze bundle
{
  "scripts": {
    "analyze": "source-map-explorer 'build/static/js/*.js'"
  }
}
```

**Actions:**
- Remove unused dependencies
- Use tree-shaking
- Import only needed modules
- Use dynamic imports for large libraries

## Backend Optimizations

### 1. Database Connection Pooling

**Configuration:**
```python
# database.py
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Number of connections to maintain
    max_overflow=10,       # Additional connections when pool is full
    pool_timeout=30,       # Timeout for getting connection
    pool_recycle=3600,     # Recycle connections after 1 hour
    pool_pre_ping=True     # Verify connections before use
)
```

**Benefits:**
- Reuses database connections
- Reduces connection overhead
- Handles concurrent requests efficiently

### 2. Query Optimization

**Best practices:**
```python
# Use indexes
class EvaluationSession(Base):
    __tablename__ = "evaluation_sessions"
    
    # Add indexes for frequently queried columns
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),
        Index('idx_consensus_score', 'consensus_score'),
        Index('idx_status', 'status'),
    )

# Use select_related to avoid N+1 queries
sessions = db.query(EvaluationSession)\
    .options(joinedload(EvaluationSession.judge_results))\
    .filter(EvaluationSession.user_id == user_id)\
    .all()

# Use pagination
sessions = db.query(EvaluationSession)\
    .offset((page - 1) * limit)\
    .limit(limit)\
    .all()
```

### 3. Caching Strategy

**Implementation:**
```python
# cache.py
import redis
from functools import wraps

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

def cache_result(ttl=300):
    """Cache decorator with TTL"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl=600)
async def get_user_preferences(user_id: str):
    return db.query(UserPreferences).filter_by(user_id=user_id).first()
```

**What to cache:**
- User preferences
- Frequently accessed evaluation results
- Aggregated statistics
- Session metadata

**What NOT to cache:**
- Real-time data
- User-specific sensitive data
- Frequently changing data

### 4. Async Processing

**Implementation:**
```python
from fastapi import BackgroundTasks

@app.post("/api/v1/evaluations")
async def create_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    # Create session immediately
    session = create_session(request)
    
    # Process evaluation in background
    background_tasks.add_task(
        process_evaluation,
        session.id,
        request
    )
    
    return {"session_id": session.id}
```

**Benefits:**
- Faster API response times
- Better user experience
- Handles long-running tasks

### 5. Response Compression

**Configuration:**
```python
# main.py
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,  # Only compress responses > 1KB
    compresslevel=6     # Balance between speed and compression
)
```

**Benefits:**
- Reduces bandwidth usage
- Faster data transfer
- Lower hosting costs

### 6. Database Query Batching

**Implementation:**
```python
# Instead of multiple queries
for session_id in session_ids:
    session = db.query(EvaluationSession).get(session_id)
    process(session)

# Use batch query
sessions = db.query(EvaluationSession)\
    .filter(EvaluationSession.id.in_(session_ids))\
    .all()

for session in sessions:
    process(session)
```

## Database Optimizations

### 1. Indexing Strategy

```sql
-- Composite indexes for common queries
CREATE INDEX idx_user_created ON evaluation_sessions(user_id, created_at DESC);
CREATE INDEX idx_status_created ON evaluation_sessions(status, created_at DESC);

-- Partial indexes for specific conditions
CREATE INDEX idx_completed_sessions ON evaluation_sessions(created_at DESC)
WHERE status = 'completed';

-- Full-text search indexes
CREATE INDEX idx_source_text_fts ON evaluation_sessions
USING gin(to_tsvector('english', source_text));
```

### 2. Query Performance

**Use EXPLAIN ANALYZE:**
```sql
EXPLAIN ANALYZE
SELECT * FROM evaluation_sessions
WHERE user_id = 'user123'
ORDER BY created_at DESC
LIMIT 20;
```

**Optimize based on results:**
- Add missing indexes
- Rewrite inefficient queries
- Use appropriate JOIN types
- Avoid SELECT *

### 3. Connection Management

**Best practices:**
- Use connection pooling
- Close connections properly
- Set appropriate timeouts
- Monitor connection usage

### 4. Data Archival

**Strategy:**
```python
# Archive old evaluations
def archive_old_evaluations():
    cutoff_date = datetime.now() - timedelta(days=90)
    
    # Move to archive table
    old_sessions = db.query(EvaluationSession)\
        .filter(EvaluationSession.created_at < cutoff_date)\
        .all()
    
    for session in old_sessions:
        archive_session(session)
        db.delete(session)
    
    db.commit()
```

## WebSocket Optimizations

### 1. Connection Management

**Implementation:**
```python
# Limit connections per user
MAX_CONNECTIONS_PER_USER = 5

@sio.on('connect')
async def handle_connect(sid, environ, auth):
    user_id = get_user_from_token(auth['token'])
    
    # Check connection limit
    user_connections = get_user_connections(user_id)
    if len(user_connections) >= MAX_CONNECTIONS_PER_USER:
        return False  # Reject connection
    
    # Store connection
    store_connection(user_id, sid)
    return True
```

### 2. Message Batching

**Implementation:**
```python
# Batch progress updates
progress_buffer = []
last_send_time = time.time()

def send_progress(data):
    progress_buffer.append(data)
    
    # Send every 100ms or when buffer is full
    if time.time() - last_send_time > 0.1 or len(progress_buffer) >= 10:
        sio.emit('evaluation_progress', progress_buffer)
        progress_buffer.clear()
        last_send_time = time.time()
```

### 3. Compression

**Enable compression:**
```python
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    compression_threshold=1024,  # Compress messages > 1KB
    compression_method='gzip'
)
```

## Monitoring and Profiling

### 1. Application Metrics

**Track key metrics:**
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate
- Active connections
- Database query time
- Cache hit rate

### 2. Performance Monitoring

**Tools:**
- Backend: Python profilers (cProfile, py-spy)
- Frontend: Chrome DevTools, Lighthouse
- Database: pg_stat_statements
- Infrastructure: Prometheus + Grafana

### 3. Logging

**Best practices:**
```python
# Structured logging with timing
import logging
import time

logger = logging.getLogger(__name__)

def timed_operation(operation_name):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"{operation_name} completed",
                    extra={
                        "duration_ms": duration * 1000,
                        "operation": operation_name
                    }
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{operation_name} failed",
                    extra={
                        "duration_ms": duration * 1000,
                        "operation": operation_name,
                        "error": str(e)
                    }
                )
                raise
        return wrapper
    return decorator
```

## Performance Checklist

### Frontend
- [ ] Code splitting implemented
- [ ] Components memoized where appropriate
- [ ] Virtual scrolling for long lists
- [ ] Debouncing on search inputs
- [ ] Images optimized and lazy loaded
- [ ] Bundle size analyzed and optimized
- [ ] Service worker for caching
- [ ] Lighthouse score > 90

### Backend
- [ ] Database connection pooling configured
- [ ] Queries optimized with indexes
- [ ] Caching strategy implemented
- [ ] Response compression enabled
- [ ] Async processing for long tasks
- [ ] Rate limiting configured
- [ ] Health checks implemented
- [ ] Monitoring and logging in place

### Database
- [ ] Indexes on frequently queried columns
- [ ] Query performance analyzed
- [ ] Connection limits configured
- [ ] Backup and archival strategy
- [ ] Vacuum and analyze scheduled

### Infrastructure
- [ ] CDN for static assets
- [ ] Load balancer configured
- [ ] Auto-scaling enabled
- [ ] Monitoring alerts set up
- [ ] Backup strategy in place

## Performance Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Frontend Load Time | < 2s | TBD | ⏳ |
| API Response Time | < 500ms | TBD | ⏳ |
| WebSocket Latency | < 100ms | TBD | ⏳ |
| Database Query Time | < 100ms | TBD | ⏳ |
| Cache Hit Rate | > 80% | TBD | ⏳ |
| Error Rate | < 1% | TBD | ⏳ |
| Concurrent Users | 100+ | TBD | ⏳ |

## Troubleshooting Performance Issues

### Slow API Responses

1. Check database query performance
2. Review cache hit rates
3. Analyze slow query logs
4. Check for N+1 query problems
5. Review connection pool usage

### High Memory Usage

1. Check for memory leaks
2. Review cache size limits
3. Analyze object retention
4. Check for large result sets
5. Review connection pool size

### Slow Frontend Load

1. Analyze bundle size
2. Check for render blocking resources
3. Review network waterfall
4. Check for unnecessary re-renders
5. Analyze third-party scripts

### Database Performance

1. Run EXPLAIN ANALYZE on slow queries
2. Check index usage
3. Review connection pool metrics
4. Analyze lock contention
5. Check for missing indexes

## Continuous Optimization

1. **Regular Performance Testing**: Run tests weekly
2. **Monitor Metrics**: Track key performance indicators
3. **Profile Regularly**: Use profiling tools to find bottlenecks
4. **Review Logs**: Analyze slow queries and errors
5. **Update Dependencies**: Keep libraries up to date
6. **Optimize Incrementally**: Make small, measurable improvements
7. **Document Changes**: Track performance improvements

## Resources

- [Web.dev Performance](https://web.dev/performance/)
- [FastAPI Performance](https://fastapi.tiangolo.com/deployment/concepts/)
- [PostgreSQL Performance](https://www.postgresql.org/docs/current/performance-tips.html)
- [React Performance](https://react.dev/learn/render-and-commit)
- [Locust Documentation](https://docs.locust.io/)
