.PHONY: help build up down restart logs clean test dev prod health

# Default target
help:
	@echo "LLM Judge Auditor - Docker Management"
	@echo ""
	@echo "Available targets:"
	@echo "  make dev          - Start development environment"
	@echo "  make prod         - Start production environment"
	@echo "  make build        - Build all Docker images"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View logs from all services"
	@echo "  make logs-backend - View backend logs"
	@echo "  make logs-frontend- View frontend logs"
	@echo "  make clean        - Remove all containers, volumes, and images"
	@echo "  make test         - Run tests"
	@echo "  make health       - Check health of all services"
	@echo "  make shell-backend- Open shell in backend container"
	@echo "  make shell-frontend- Open shell in frontend container"
	@echo "  make db-migrate   - Run database migrations"
	@echo "  make db-shell     - Open PostgreSQL shell"

# Development environment
dev:
	@echo "Starting development environment..."
	docker-compose up -d
	@echo "Services started. Access:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"

# Production environment
prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
	@echo "Production services started."
	@echo "Access: http://localhost"

# Build images
build:
	@echo "Building Docker images..."
	docker-compose build --no-cache

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# Restart services
restart:
	docker-compose restart

# View logs
logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

logs-postgres:
	docker-compose logs -f postgres

logs-redis:
	docker-compose logs -f redis

logs-nginx:
	docker-compose logs -f nginx

# Clean up
clean:
	@echo "Removing all containers, volumes, and images..."
	docker-compose down -v --rmi all
	@echo "Cleanup complete."

# Run tests
test:
	@echo "Running backend tests..."
	docker-compose exec backend pytest
	@echo "Running frontend tests..."
	docker-compose exec frontend npm test -- --watchAll=false

# Health checks
health:
	@echo "Checking service health..."
	@echo "\nBackend:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Backend not responding"
	@echo "\nDetailed health:"
	@curl -s http://localhost:8000/health/detailed | python -m json.tool || echo "Backend not responding"
	@echo "\nMetrics:"
	@curl -s http://localhost:8000/metrics | python -m json.tool || echo "Backend not responding"

# Shell access
shell-backend:
	docker-compose exec backend /bin/bash

shell-frontend:
	docker-compose exec frontend /bin/sh

shell-postgres:
	docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor

# Database operations
db-migrate:
	@echo "Running database migrations..."
	docker-compose exec backend alembic upgrade head

db-rollback:
	@echo "Rolling back last migration..."
	docker-compose exec backend alembic downgrade -1

db-shell:
	docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor

# Monitoring
monitor:
	@echo "Monitoring services..."
	watch -n 2 'docker-compose ps && echo "\n=== Resource Usage ===" && docker stats --no-stream'

# SSL certificate generation (development)
ssl-dev:
	@echo "Generating self-signed SSL certificate..."
	mkdir -p nginx/ssl
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
		-keyout nginx/ssl/key.pem \
		-out nginx/ssl/cert.pem \
		-subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
	@echo "SSL certificate generated at nginx/ssl/"

# Backup database
backup-db:
	@echo "Backing up database..."
	docker-compose exec -T postgres pg_dump -U llm_judge_user llm_judge_auditor > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backed up."

# Restore database
restore-db:
	@echo "Restoring database from $(FILE)..."
	@if [ -z "$(FILE)" ]; then echo "Usage: make restore-db FILE=backup.sql"; exit 1; fi
	docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor < $(FILE)
	@echo "Database restored."

# Update dependencies
update-deps:
	@echo "Updating backend dependencies..."
	docker-compose exec backend pip install --upgrade -r requirements.txt
	@echo "Updating frontend dependencies..."
	docker-compose exec frontend npm update

# Security scan
security-scan:
	@echo "Running security scan..."
	@echo "\nBackend dependencies:"
	docker-compose exec backend pip-audit || echo "pip-audit not installed"
	@echo "\nFrontend dependencies:"
	docker-compose exec frontend npm audit

# Performance test
perf-test:
	@echo "Running performance tests..."
	@echo "This requires 'ab' (Apache Bench) to be installed"
	ab -n 1000 -c 10 http://localhost:8000/health
