.PHONY: help build up down restart logs clean test lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make build          - Build all Docker images"
	@echo "  make up             - Start all services"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View logs from all services"
	@echo "  make logs-backend   - View backend logs"
	@echo "  make logs-frontend  - View frontend logs"
	@echo "  make clean          - Remove all containers and volumes"
	@echo "  make test           - Run all tests"
	@echo "  make test-backend   - Run backend tests"
	@echo "  make test-frontend  - Run frontend tests"
	@echo "  make lint           - Run linting on all code"
	@echo "  make format         - Format all code"
	@echo "  make shell-backend  - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"
	@echo "  make db-migrate     - Run database migrations"
	@echo "  make db-reset       - Reset database"

# Docker commands
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

clean:
	docker-compose down -v
	docker system prune -f

# Testing
test: test-backend test-frontend

test-backend:
	docker-compose exec backend pytest tests/ -v

test-frontend:
	docker-compose exec frontend npm test -- --watchAll=false

# Linting and formatting
lint:
	docker-compose exec backend flake8 app
	docker-compose exec frontend npm run lint

format:
	docker-compose exec backend black app
	docker-compose exec frontend npm run format

# Shell access
shell-backend:
	docker-compose exec backend /bin/bash

shell-frontend:
	docker-compose exec frontend /bin/sh

# Database operations
db-migrate:
	docker-compose exec backend alembic upgrade head

db-reset:
	docker-compose exec backend alembic downgrade base
	docker-compose exec backend alembic upgrade head

# Development setup
setup:
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	docker-compose build
	docker-compose up -d
	@echo "Waiting for services to start..."
	sleep 10
	make db-migrate
	@echo "Setup complete! Access the application at http://localhost:3000"

# Production deployment
deploy-prod:
	docker-compose --profile production up -d

# Health check
health:
	@echo "Checking backend health..."
	@curl -f http://localhost:8000/health || echo "Backend is not healthy"
	@echo "\nChecking frontend..."
	@curl -f http://localhost:3000 || echo "Frontend is not accessible"
