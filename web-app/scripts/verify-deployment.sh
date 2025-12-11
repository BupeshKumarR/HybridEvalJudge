#!/bin/bash

# Deployment Verification Script
# This script verifies that all deployment components are properly configured

# Don't exit on error - we want to see all checks
# set -e

echo "🔍 LLM Judge Auditor - Deployment Verification"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Check function
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAILED++))
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# 1. Check Docker
echo "1. Checking Docker..."
docker --version > /dev/null 2>&1
check "Docker is installed"

# Check for both docker-compose (v1) and docker compose (v2)
if docker-compose --version > /dev/null 2>&1; then
    check "Docker Compose is installed (v1)"
elif docker compose version > /dev/null 2>&1; then
    check "Docker Compose is installed (v2)"
else
    echo -e "${RED}✗${NC} Docker Compose is not installed"
    ((FAILED++))
fi

# 2. Check files
echo ""
echo "2. Checking configuration files..."

[ -f "docker-compose.yml" ]
check "docker-compose.yml exists"

[ -f "docker-compose.prod.yml" ]
check "docker-compose.prod.yml exists"

[ -f ".env.example" ]
check ".env.example exists"

[ -f "Makefile" ]
check "Makefile exists"

# 3. Check Dockerfiles
echo ""
echo "3. Checking Dockerfiles..."

[ -f "backend/Dockerfile" ]
check "Backend Dockerfile exists"

[ -f "frontend/Dockerfile" ]
check "Frontend Dockerfile exists"

[ -f "backend/.dockerignore" ]
check "Backend .dockerignore exists"

[ -f "frontend/.dockerignore" ]
check "Frontend .dockerignore exists"

# 4. Check Nginx configuration
echo ""
echo "4. Checking Nginx configuration..."

[ -f "nginx/nginx.conf" ]
check "nginx.conf exists"

[ -f "nginx/locations.conf" ]
check "locations.conf exists"

[ -d "nginx/ssl" ]
check "SSL directory exists"

[ -f "nginx/ssl/README.md" ]
check "SSL README exists"

# 5. Check monitoring
echo ""
echo "5. Checking monitoring setup..."

[ -f "backend/app/monitoring.py" ]
check "monitoring.py exists"

[ -f "backend/app/logging_config.py" ]
check "logging_config.py exists"

[ -d "monitoring" ]
check "Monitoring directory exists"

[ -f "monitoring/README.md" ]
check "Monitoring README exists"

# 6. Check documentation
echo ""
echo "6. Checking documentation..."

[ -f "DEPLOYMENT.md" ]
check "DEPLOYMENT.md exists"

[ -f "DEPLOYMENT_SUMMARY.md" ]
check "DEPLOYMENT_SUMMARY.md exists"

[ -f "QUICK_REFERENCE.md" ]
check "QUICK_REFERENCE.md exists"

# 7. Validate Docker Compose
echo ""
echo "7. Validating Docker Compose configuration..."

docker-compose config --quiet > /dev/null 2>&1
check "Docker Compose configuration is valid"

# 8. Check environment file
echo ""
echo "8. Checking environment configuration..."

if [ -f ".env" ]; then
    check ".env file exists"
    
    # Check for default passwords
    if grep -q "POSTGRES_PASSWORD=changeme" .env 2>/dev/null; then
        warn "Default PostgreSQL password detected - change in production!"
    fi
    
    if grep -q "SECRET_KEY=dev-secret-key" .env 2>/dev/null; then
        warn "Default SECRET_KEY detected - change in production!"
    fi
else
    warn ".env file not found - copy from .env.example"
fi

# 9. Check backend dependencies
echo ""
echo "9. Checking backend dependencies..."

if [ -f "backend/requirements.txt" ]; then
    check "requirements.txt exists"
    
    grep -q "psutil" backend/requirements.txt
    check "psutil dependency present"
    
    grep -q "sentry-sdk" backend/requirements.txt
    check "sentry-sdk dependency present"
fi

# 10. Check if services are running
echo ""
echo "10. Checking running services..."

if docker-compose ps | grep -q "Up"; then
    check "Some services are running"
    
    # Check specific services
    if docker-compose ps | grep -q "postgres.*Up"; then
        check "PostgreSQL is running"
    else
        warn "PostgreSQL is not running"
    fi
    
    if docker-compose ps | grep -q "redis.*Up"; then
        check "Redis is running"
    else
        warn "Redis is not running"
    fi
    
    if docker-compose ps | grep -q "backend.*Up"; then
        check "Backend is running"
    else
        warn "Backend is not running"
    fi
    
    if docker-compose ps | grep -q "frontend.*Up"; then
        check "Frontend is running"
    else
        warn "Frontend is not running"
    fi
else
    warn "No services are currently running (use 'make dev' to start)"
fi

# 11. Check health endpoints (if services are running)
echo ""
echo "11. Checking health endpoints..."

if docker-compose ps | grep -q "backend.*Up"; then
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        check "Backend health endpoint is accessible"
    else
        warn "Backend health endpoint is not accessible"
    fi
    
    if curl -s http://localhost:8000/health/detailed > /dev/null 2>&1; then
        check "Detailed health endpoint is accessible"
    else
        warn "Detailed health endpoint is not accessible"
    fi
    
    if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
        check "Metrics endpoint is accessible"
    else
        warn "Metrics endpoint is not accessible"
    fi
fi

# Summary
echo ""
echo "=============================================="
echo "Summary:"
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${RED}Failed:${NC} $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ Please review warnings above${NC}"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.example to .env and configure"
    echo "2. Start services: make dev"
    echo "3. Check health: make health"
    echo "4. View logs: make logs"
    echo ""
    echo "For production deployment, see DEPLOYMENT.md"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo "Please fix the issues above before deploying"
    exit 1
fi
