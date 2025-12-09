#!/bin/bash

# Development setup script for LLM Judge Auditor Web Application

set -e

echo "🚀 Setting up LLM Judge Auditor Web Application..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration before continuing."
    echo "   Press Enter when ready..."
    read
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p backend/migrations
mkdir -p nginx/ssl
mkdir -p logs

# Generate SSL certificates for development (self-signed)
if [ ! -f nginx/ssl/cert.pem ]; then
    echo "🔐 Generating self-signed SSL certificates for development..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
fi

# Build Docker images
echo "🏗️  Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check if services are running
echo "🔍 Checking service health..."
docker-compose ps

# Run database migrations
echo "🗄️  Running database migrations..."
# Uncomment when migrations are set up
# docker-compose exec backend alembic upgrade head

echo ""
echo "✅ Setup complete!"
echo ""
echo "📍 Access points:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📝 Useful commands:"
echo "   make logs          - View all logs"
echo "   make test          - Run tests"
echo "   make down          - Stop services"
echo "   make clean         - Remove all containers and volumes"
echo ""
