# Frontend Architecture

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         App Component                           │
│                   (QueryClientProvider)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AppRoutes                                │
│                    (BrowserRouter)                              │
├─────────────────────────────────────────────────────────────────┤
│  /login          →  LoginPage                                   │
│  /               →  PrivateRoute → ChatPage                     │
│  /history        →  PrivateRoute → HistoryPage                  │
└─────────────────────────────────────────────────────────────────┘
```

## Layout Structure

```
┌──────────────────────────────────────────────────────────────────┐
│                          Navbar                                  │
│  [☰] LLM Judge Auditor        Chat | History | User | Logout    │
└──────────────────────────────────────────────────────────────────┘
┌──────────────┬───────────────────────────────────────────────────┐
│              │                                                   │
│   Sidebar    │              Main Content                         │
│   (History)  │              (Page Component)                     │
│              │                                                   │
│  Session 1   │                                                   │
│  Session 2   │                                                   │
│  Session 3   │                                                   │
│  ...         │                                                   │
│              │                                                   │
│  [Load More] │                                                   │
│              │                                                   │
└──────────────┴───────────────────────────────────────────────────┘
```

## State Management Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Zustand Stores                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  authStore   │  │ evalStore    │  │ historyStore │        │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤        │
│  │ • user       │  │ • sessionId  │  │ • sessions   │        │
│  │ • token      │  │ • messages   │  │ • page       │        │
│  │ • isAuth     │  │ • isEval     │  │ • hasMore    │        │
│  │              │  │ • config     │  │ • isLoading  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
                    React Components
```

## Data Flow

### REST API Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Component  │────▶│  React Query │────▶│  API Client  │
│              │     │   Hook       │     │   (Axios)    │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                    ▲                     │
       │                    │                     │
       │                    │                     ▼
       │                    │              ┌──────────────┐
       │                    └──────────────│   Backend    │
       │                                   │   REST API   │
       └───────────────────────────────────│              │
                  (cached data)            └──────────────┘
```

### WebSocket Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Component  │────▶│  useWebSocket│────▶│  WebSocket   │
│              │     │    Hook      │     │   Service    │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                    ▲                     │
       │                    │                     │
       │                    │                     ▼
       │                    │              ┌──────────────┐
       │                    │              │   Backend    │
       │                    │              │  Socket.IO   │
       └────────────────────┴──────────────│   Server     │
              (real-time events)           └──────────────┘
```

## Authentication Flow

```
┌─────────────┐
│   Login     │
│   Page      │
└──────┬──────┘
       │
       │ 1. Submit credentials
       ▼
┌─────────────┐
│  authApi    │
│  .login()   │
└──────┬──────┘
       │
       │ 2. POST /auth/login
       ▼
┌─────────────┐
│   Backend   │
│   API       │
└──────┬──────┘
       │
       │ 3. Return { user, token }
       ▼
┌─────────────┐
│  authStore  │
│  .login()   │
└──────┬──────┘
       │
       │ 4. Store user & token
       │    (persisted to localStorage)
       ▼
┌─────────────┐
│  Navigate   │
│  to /       │
└─────────────┘
```

## Evaluation Flow

```
┌─────────────┐
│   Chat      │
│   Page      │
└──────┬──────┘
       │
       │ 1. Submit evaluation
       ▼
┌─────────────┐
│createEval   │
│ mutation    │
└──────┬──────┘
       │
       │ 2. POST /evaluations
       ▼
┌─────────────┐
│   Backend   │
│   API       │
└──────┬──────┘
       │
       │ 3. Return { session_id, ws_url }
       ▼
┌─────────────┐
│  WebSocket  │
│  connect    │
└──────┬──────┘
       │
       │ 4. Emit 'start_evaluation'
       ▼
┌─────────────┐
│   Backend   │
│  processes  │
└──────┬──────┘
       │
       │ 5. Stream events:
       │    • evaluation_progress
       │    • judge_result
       │    • evaluation_complete
       ▼
┌─────────────┐
│  Component  │
│  updates    │
└─────────────┘
```

## API Client Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Axios Instance                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Request Interceptor                                        │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 1. Get token from authStore                        │   │
│  │ 2. Add Authorization header                        │   │
│  │ 3. Add request ID                                  │   │
│  └────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│                    HTTP Request                             │
│                          │                                  │
│                          ▼                                  │
│                    Backend API                              │
│                          │                                  │
│                          ▼                                  │
│                    HTTP Response                            │
│                          │                                  │
│  Response Interceptor    ▼                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 1. Check status code                               │   │
│  │ 2. Handle 401 → logout & redirect                  │   │
│  │ 3. Handle 429 → log rate limit                     │   │
│  │ 4. Handle 500 → log server error                   │   │
│  │ 5. Handle network error → retry with backoff       │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## WebSocket Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   WebSocket Service                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Connection Management                                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • connect()      - Establish connection            │   │
│  │ • disconnect()   - Close connection                │   │
│  │ • isConnected()  - Check status                    │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Event Management                                           │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • emit()         - Send event to server            │   │
│  │ • on()           - Listen to event                 │   │
│  │ • off()          - Remove listener                 │   │
│  │ • clearHandlers()- Remove all listeners            │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Reconnection Logic                                         │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • Max attempts: 5                                  │   │
│  │ • Exponential backoff: 1s, 2s, 4s, 8s, 16s       │   │
│  │ • Auto-reconnect on disconnect                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Responsive Design Breakpoints

```
Mobile First Approach:

┌─────────────────────────────────────────────────────────────┐
│  Default (< 768px)                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  [☰] LLM Judge Auditor              [⋮]             │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │                                                      │  │
│  │              Full Width Content                      │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Tablet (768px - 1024px)                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  [☰] LLM Judge Auditor    Chat | History | User     │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │                                                      │  │
│  │              Full Width Content                      │  │
│  │              (Sidebar as overlay)                    │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Desktop (> 1024px)                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LLM Judge Auditor    Chat | History | User | Logout│  │
│  ├────────────┬─────────────────────────────────────────┤  │
│  │  Sidebar   │                                         │  │
│  │  (Fixed)   │         Main Content                    │  │
│  │            │                                         │  │
│  └────────────┴─────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Error Handling                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Network Errors                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • Retry with exponential backoff                   │   │
│  │ • Show user-friendly message                       │   │
│  │ • Log to console for debugging                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Authentication Errors (401)                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • Clear auth state                                 │   │
│  │ • Redirect to login                                │   │
│  │ • Show session expired message                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Authorization Errors (403)                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • Show access denied message                       │   │
│  │ • Log error details                                │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Rate Limit Errors (429)                                    │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • Show rate limit message                          │   │
│  │ • Display retry-after time                         │   │
│  │ • Disable submit button temporarily                │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Server Errors (500)                                        │
│  ┌────────────────────────────────────────────────────┐   │
│  │ • Show generic error message                       │   │
│  │ • Log error with request ID                        │   │
│  │ • Offer retry option                               │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Performance Optimizations

### React Query Caching

```
┌─────────────────────────────────────────────────────────────┐
│                   React Query Cache                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: 'currentUser'                                       │
│  ├─ Stale Time: 5 minutes                                  │
│  ├─ Cache Time: 10 minutes                                 │
│  └─ Refetch: On window focus (disabled)                    │
│                                                             │
│  Query: 'evaluationHistory'                                 │
│  ├─ Stale Time: 1 minute                                   │
│  ├─ Cache Time: 5 minutes                                  │
│  └─ Keep Previous Data: true (for pagination)              │
│                                                             │
│  Query: 'evaluation/:id'                                    │
│  ├─ Stale Time: 30 seconds                                 │
│  ├─ Retry: 3 times with exponential backoff                │
│  └─ Refetch: On window focus (disabled)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Code Splitting (Future)

```
Route-based splitting:
├─ /login       → LoginPage.chunk.js
├─ /            → ChatPage.chunk.js
└─ /history     → HistoryPage.chunk.js

Component-based splitting:
├─ Visualizations → Charts.chunk.js
├─ Export         → Export.chunk.js
└─ Settings       → Settings.chunk.js
```

## Security Considerations

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Measures                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✓ JWT tokens stored in memory (Zustand with persistence)  │
│  ✓ Automatic token injection via interceptors              │
│  ✓ Protected routes with authentication check              │
│  ✓ HTTPS only in production                                │
│  ✓ XSS prevention via React's built-in escaping            │
│  ✓ CSRF protection via SameSite cookies (backend)          │
│  ✓ Input validation on all forms                           │
│  ✓ Rate limiting handled by backend                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
