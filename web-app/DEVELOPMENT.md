# Development Guide

This guide covers development workflows, best practices, and troubleshooting for the LLM Judge Auditor Web Application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Project Structure](#project-structure)
4. [Backend Development](#backend-development)
5. [Frontend Development](#frontend-development)
6. [Testing](#testing)
7. [Database Management](#database-management)
8. [Debugging](#debugging)
9. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Git
- (Optional) Node.js 18+ for local frontend development
- (Optional) Python 3.11+ for local backend development

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd web-app

# Run setup script
./scripts/setup-dev.sh

# Or manually:
cp .env.example .env
# Edit .env with your configuration
make setup
```

## Development Workflow

### Starting Development

```bash
# Start all services with hot reload
make up

# View logs
make logs

# Or view specific service logs
make logs-backend
make logs-frontend
```

### Making Changes

1. **Backend Changes**: Edit files in `backend/app/`. Changes will auto-reload.
2. **Frontend Changes**: Edit files in `frontend/src/`. Changes will auto-reload.
3. **Database Changes**: Create migrations in `backend/migrations/`.

### Testing Changes

```bash
# Run all tests
make test

# Run specific tests
make test-backend
make test-frontend

# Run with coverage
docker-compose exec backend pytest --cov=app
docker-compose exec frontend npm test -- --coverage
```

### Committing Changes

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Commit
git add .
git commit -m "Your commit message"
git push
```

## Project Structure

### Backend Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   ├── v1/           # API version 1
│   │   │   ├── endpoints/
│   │   │   │   ├── auth.py
│   │   │   │   ├── evaluations.py
│   │   │   │   └── users.py
│   │   │   └── router.py
│   │   └── deps.py       # Dependencies
│   ├── core/             # Core configuration
│   │   ├── config.py     # Settings
│   │   ├── security.py   # Auth utilities
│   │   └── logging.py    # Logging config
│   ├── db/               # Database
│   │   ├── base.py       # Base model
│   │   ├── session.py    # DB session
│   │   └── models/       # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   │   ├── evaluation.py
│   │   ├── user.py
│   │   └── token.py
│   ├── services/         # Business logic
│   │   ├── evaluation.py
│   │   ├── metrics.py
│   │   └── websocket.py
│   └── main.py           # FastAPI app
├── migrations/           # Alembic migrations
├── tests/                # Tests
└── requirements.txt
```

### Frontend Structure

```
frontend/
├── public/               # Static files
├── src/
│   ├── components/       # Reusable components
│   │   ├── chat/
│   │   ├── visualizations/
│   │   └── common/
│   ├── pages/            # Page components
│   │   ├── Home.tsx
│   │   ├── Evaluation.tsx
│   │   └── History.tsx
│   ├── services/         # API clients
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── hooks/            # Custom hooks
│   │   ├── useEvaluation.ts
│   │   └── useWebSocket.ts
│   ├── store/            # Zustand store
│   │   └── evaluationStore.ts
│   ├── types/            # TypeScript types
│   │   └── evaluation.ts
│   ├── utils/            # Utilities
│   ├── App.tsx
│   └── index.tsx
└── package.json
```

## Backend Development

### Adding a New API Endpoint

1. Create endpoint in `backend/app/api/v1/endpoints/`:

```python
from fastapi import APIRouter, Depends
from app.schemas.evaluation import EvaluationCreate, EvaluationResponse

router = APIRouter()

@router.post("/evaluations", response_model=EvaluationResponse)
async def create_evaluation(
    evaluation: EvaluationCreate,
    db: Session = Depends(get_db)
):
    # Implementation
    pass
```

2. Register router in `backend/app/api/v1/router.py`:

```python
from app.api.v1.endpoints import evaluations

api_router = APIRouter()
api_router.include_router(evaluations.router, tags=["evaluations"])
```

### Adding a Database Model

1. Create model in `backend/app/db/models/`:

```python
from sqlalchemy import Column, String, Float
from app.db.base import Base

class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(String, primary_key=True)
    source_text = Column(String, nullable=False)
    consensus_score = Column(Float)
```

2. Create migration:

```bash
docker-compose exec backend alembic revision --autogenerate -m "Add evaluation table"
docker-compose exec backend alembic upgrade head
```

### Adding a Service

Create service in `backend/app/services/`:

```python
from app.db.models.evaluation import Evaluation

class EvaluationService:
    def __init__(self, db: Session):
        self.db = db
    
    async def create_evaluation(self, data: dict) -> Evaluation:
        # Implementation
        pass
```

## Frontend Development

### Adding a New Component

1. Create component in `frontend/src/components/`:

```typescript
import React from 'react';

interface MyComponentProps {
  title: string;
}

export const MyComponent: React.FC<MyComponentProps> = ({ title }) => {
  return (
    <div className="p-4">
      <h2>{title}</h2>
    </div>
  );
};
```

2. Add tests in `frontend/src/components/__tests__/`:

```typescript
import { render, screen } from '@testing-library/react';
import { MyComponent } from '../MyComponent';

test('renders component', () => {
  render(<MyComponent title="Test" />);
  expect(screen.getByText('Test')).toBeInTheDocument();
});
```

### Adding a New Page

1. Create page in `frontend/src/pages/`:

```typescript
import React from 'react';
import { MyComponent } from '../components/MyComponent';

export const MyPage: React.FC = () => {
  return (
    <div>
      <MyComponent title="My Page" />
    </div>
  );
};
```

2. Add route in `frontend/src/App.tsx`:

```typescript
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MyPage } from './pages/MyPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/my-page" element={<MyPage />} />
      </Routes>
    </BrowserRouter>
  );
}
```

### Using the API Client

```typescript
import { apiClient } from '../services/api';

const fetchEvaluations = async () => {
  const response = await apiClient.get('/evaluations');
  return response.data;
};
```

## Testing

### Backend Testing

```bash
# Run all tests
docker-compose exec backend pytest

# Run specific test file
docker-compose exec backend pytest tests/test_api.py

# Run with coverage
docker-compose exec backend pytest --cov=app --cov-report=html

# Run specific test
docker-compose exec backend pytest tests/test_api.py::test_create_evaluation
```

### Frontend Testing

```bash
# Run all tests
docker-compose exec frontend npm test

# Run with coverage
docker-compose exec frontend npm test -- --coverage

# Run specific test file
docker-compose exec frontend npm test -- MyComponent.test.tsx
```

## Database Management

### Creating Migrations

```bash
# Auto-generate migration
docker-compose exec backend alembic revision --autogenerate -m "Description"

# Create empty migration
docker-compose exec backend alembic revision -m "Description"
```

### Running Migrations

```bash
# Upgrade to latest
make db-migrate

# Or manually
docker-compose exec backend alembic upgrade head

# Downgrade one version
docker-compose exec backend alembic downgrade -1

# Reset database
make db-reset
```

### Accessing Database

```bash
# Using psql
docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor

# Using pgAdmin or other GUI tools
# Host: localhost
# Port: 5432
# Database: llm_judge_auditor
# User: llm_judge_user
# Password: (from .env)
```

## Debugging

### Backend Debugging

1. Add breakpoint in code:

```python
import pdb; pdb.set_trace()
```

2. Attach to container:

```bash
docker attach llm-judge-backend
```

### Frontend Debugging

Use browser DevTools:
- Chrome: F12 or Cmd+Option+I (Mac)
- Firefox: F12 or Cmd+Option+I (Mac)

### Viewing Logs

```bash
# All logs
make logs

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

## Best Practices

### Backend

1. **Use type hints**: Always use Python type hints
2. **Async/await**: Use async functions for I/O operations
3. **Error handling**: Use proper exception handling
4. **Validation**: Use Pydantic models for validation
5. **Testing**: Write tests for all endpoints

### Frontend

1. **TypeScript**: Use TypeScript for type safety
2. **Components**: Keep components small and focused
3. **Hooks**: Use custom hooks for reusable logic
4. **State management**: Use Zustand for global state
5. **Testing**: Write tests for components and hooks

### Git Workflow

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes and commit regularly
3. Run tests before pushing
4. Create pull request
5. Address review comments
6. Merge when approved

### Code Style

- **Backend**: Follow PEP 8, use Black for formatting
- **Frontend**: Follow Airbnb style guide, use Prettier
- **Commits**: Use conventional commits format

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using port
lsof -i :3000
# Kill process
kill -9 <PID>
```

**Database connection error:**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres
# Restart PostgreSQL
docker-compose restart postgres
```

**Frontend build error:**
```bash
# Clear cache and rebuild
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Backend import error:**
```bash
# Rebuild backend
docker-compose build backend
docker-compose up -d backend
```

For more help, check the main README.md or open an issue.
