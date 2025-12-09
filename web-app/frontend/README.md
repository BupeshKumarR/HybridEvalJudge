# LLM Judge Auditor - Frontend

React-based frontend for the LLM Judge Auditor web application.

## Features

- 🎨 Modern UI with TailwindCSS
- 🔐 JWT-based authentication
- 🔄 Real-time evaluation streaming via WebSocket
- 📊 Interactive visualizations (coming soon)
- 📱 Fully responsive design
- 🎯 Type-safe with TypeScript
- ⚡ Optimized data fetching with React Query

## Tech Stack

- **Framework:** React 18 with TypeScript
- **Routing:** React Router v6
- **State Management:** Zustand
- **Data Fetching:** React Query + Axios
- **Real-time:** Socket.IO Client
- **Styling:** TailwindCSS
- **Build Tool:** Create React App

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend API running (see `../backend/README.md`)

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Update .env with your API URL
# REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
# REACT_APP_WS_URL=http://localhost:8000
```

### Development

```bash
# Start development server
npm start

# The app will open at http://localhost:3000
```

### Building

```bash
# Create production build
npm run build

# The build will be in the `build/` directory
```

### Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage

# Type checking
npx tsc --noEmit
```

## Project Structure

```
src/
├── api/                    # API client and endpoints
│   ├── client.ts          # Axios instance with interceptors
│   ├── auth.ts            # Authentication endpoints
│   ├── evaluations.ts     # Evaluation endpoints
│   └── types.ts           # API type definitions
├── components/            # React components
│   └── layout/           # Layout components
│       ├── MainLayout.tsx
│       ├── Navbar.tsx
│       ├── Sidebar.tsx
│       └── MobileMenu.tsx
├── hooks/                 # Custom React hooks
│   ├── useAuth.ts        # Authentication hooks
│   ├── useEvaluations.ts # Evaluation hooks
│   └── useWebSocket.ts   # WebSocket hooks
├── pages/                 # Page components
│   ├── ChatPage.tsx
│   ├── LoginPage.tsx
│   └── HistoryPage.tsx
├── routes/                # Routing configuration
│   └── index.tsx
├── services/              # Services
│   └── websocket.ts      # WebSocket service
├── store/                 # Zustand stores
│   ├── authStore.ts
│   ├── evaluationStore.ts
│   └── historyStore.ts
├── App.tsx               # Root component
└── index.tsx             # Entry point
```

## State Management

### Zustand Stores

1. **authStore** - User authentication
   - User profile
   - JWT token
   - Login/logout actions

2. **evaluationStore** - Current evaluation session
   - Session ID
   - Messages
   - Evaluation status
   - Configuration

3. **historyStore** - Evaluation history
   - Past sessions
   - Pagination state
   - Loading state

## API Integration

### REST API

All API calls go through the configured Axios client with:
- Automatic JWT token injection
- Error handling and retry logic
- Request/response interceptors

### WebSocket

Real-time communication for:
- Evaluation progress updates
- Streaming judge results
- Error notifications

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_API_BASE_URL` | Backend API base URL | `http://localhost:8000/api/v1` |
| `REACT_APP_WS_URL` | WebSocket server URL | `http://localhost:8000` |
| `NODE_ENV` | Environment | `development` |

## Available Scripts

- `npm start` - Start development server
- `npm build` - Create production build
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App (⚠️ irreversible)

## Styling

TailwindCSS is configured with a custom color palette:

```javascript
colors: {
  primary: {
    50: '#f0f9ff',
    // ... through to
    900: '#0c4a6e',
  }
}
```

## Contributing

1. Follow the existing code structure
2. Use TypeScript for all new files
3. Add tests for new features
4. Run `npx tsc --noEmit` before committing
5. Follow the component naming conventions

## Next Steps

- [ ] Implement chat interface (Task 7)
- [ ] Add visualization components (Task 8)
- [ ] Create statistics dashboard (Task 9)
- [ ] Complete history functionality (Task 10)
- [ ] Add export features (Task 11)
- [ ] Implement configuration UI (Task 12)

## Troubleshooting

### Port already in use

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Module not found errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### TypeScript errors

```bash
# Check for type errors
npx tsc --noEmit
```

## License

See the main project LICENSE file.
