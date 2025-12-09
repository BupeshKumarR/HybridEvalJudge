# LLM Judge Auditor Web Application

A production-grade web application for evaluating LLM outputs with comprehensive visualization of judge decisions, confidence metrics, and hallucination detection.

## Architecture

This is a monorepo containing:
- **Frontend**: React 18 + TypeScript + TailwindCSS
- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **Infrastructure**: Docker + Docker Compose + Nginx

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

## Quick Start

### 1. Clone and Setup

```bash
# Copy environment configuration
cp .env.example .env

# Edit .env with your configuration
nano .env

# Build and start all services
make setup
```

### 2. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Development

### Using Docker (Recommended)

```bash
# Start all services with hot reload
make up

# View logs
make logs

# Run tests
make test

# Stop services
make down
```

### Local Development (Without Docker)

#### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## Project Structure

```
web-app/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core configuration
│   │   ├── db/               # Database models and session
│   │   ├── services/         # Business logic
│   │   ├── schemas/          # Pydantic models
│   │   └── main.py           # FastAPI application
│   ├── migrations/           # Alembic migrations
│   ├── tests/                # Backend tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── public/               # Static files
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API clients
│   │   ├── hooks/            # Custom hooks
│   │   ├── store/            # Zustand store
│   │   ├── types/            # TypeScript types
│   │   └── App.tsx           # Main app component
│   ├── Dockerfile
│   └── package.json
├── nginx/
│   └── nginx.conf            # Nginx configuration
├── .github/
│   └── workflows/
│       └── ci-cd.yml         # CI/CD pipeline
├── docker-compose.yml
├── Makefile
└── README.md
```

## Available Commands

### Docker Commands

```bash
make build          # Build all Docker images
make up             # Start all services
make down           # Stop all services
make restart        # Restart all services
make logs           # View logs from all services
make clean          # Remove all containers and volumes
```

### Testing Commands

```bash
make test           # Run all tests
make test-backend   # Run backend tests
make test-frontend  # Run frontend tests
```

### Development Commands

```bash
make lint           # Run linting on all code
make format         # Format all code
make shell-backend  # Open shell in backend container
make shell-frontend # Open shell in frontend container
```

### Database Commands

```bash
make db-migrate     # Run database migrations
make db-reset       # Reset database
```

## Environment Variables

See `.env.example` for all available configuration options.

### Required Variables

- `POSTGRES_PASSWORD`: Database password
- `SECRET_KEY`: Application secret key
- `JWT_SECRET_KEY`: JWT signing key

### Optional Variables

- `ENVIRONMENT`: development/staging/production
- `LOG_LEVEL`: debug/info/warning/error
- `RATE_LIMIT_PER_MINUTE`: API rate limit

## Testing

### Backend Tests

```bash
# Run all backend tests
cd backend
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Frontend Tests

```bash
# Run all frontend tests
cd frontend
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- ChatInterface.test.tsx
```

## Deployment

### Production Deployment

```bash
# Build production images
docker-compose build --target production

# Start with production profile
make deploy-prod
```

### CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Runs tests on push/PR
2. Builds Docker images on main branch
3. Pushes images to GitHub Container Registry
4. Deploys to staging/production (configure in workflow)

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :3000  # Frontend
lsof -i :8000  # Backend
lsof -i :5432  # PostgreSQL

# Kill the process or change ports in docker-compose.yml
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
make logs-postgres

# Reset database
make db-reset
```

### Frontend Build Issues

```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `make test`
4. Run linting: `make lint`
5. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions, please open a GitHub issue.
