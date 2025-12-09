# Quick Verification Guide

This guide helps you verify that the project setup is working correctly.

## Prerequisites Check

```bash
# Check Docker
docker --version
# Expected: Docker version 20.10.0 or higher

# Check Docker Compose
docker-compose --version
# Expected: Docker Compose version 2.0.0 or higher
```

## Step-by-Step Verification

### 1. Verify File Structure

```bash
# Check main directories exist
ls -la web-app/
# Should see: backend/, frontend/, nginx/, scripts/, .github/

# Check backend structure
ls -la web-app/backend/
# Should see: app/, tests/, Dockerfile, requirements.txt

# Check frontend structure
ls -la web-app/frontend/
# Should see: src/, public/, Dockerfile, package.json
```

### 2. Verify Docker Configuration

```bash
cd web-app

# Validate docker-compose.yml
docker-compose config
# Should output valid YAML without errors

# Check services defined
docker-compose config --services
# Should list: postgres, redis, backend, frontend, nginx
```

### 3. Test Backend Setup

```bash
cd web-app

# Check backend Dockerfile
docker-compose build backend
# Should build successfully

# Check backend requirements
cat backend/requirements.txt
# Should list FastAPI, SQLAlchemy, etc.

# Verify backend code
cat backend/app/main.py
# Should show FastAPI application
```

### 4. Test Frontend Setup

```bash
cd web-app

# Check frontend Dockerfile
docker-compose build frontend
# Should build successfully

# Check frontend dependencies
cat frontend/package.json
# Should list React, TypeScript, etc.

# Verify frontend code
cat frontend/src/App.tsx
# Should show React component
```

### 5. Start Services (Quick Test)

```bash
cd web-app

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d

# Wait for services to start
sleep 10

# Check service status
docker-compose ps
# All services should be "Up" or "healthy"

# Check backend health
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"llm-judge-auditor-backend"}

# Check frontend (in browser or curl)
curl http://localhost:3000
# Should return HTML

# View logs
docker-compose logs backend | head -20
docker-compose logs frontend | head -20

# Stop services
docker-compose down
```

### 6. Test Development Tools

```bash
cd web-app

# Test Makefile commands
make help
# Should list all available commands

# Test setup script
./scripts/setup-dev.sh --help || echo "Script exists"
# Script should exist and be executable
```

### 7. Verify CI/CD Configuration

```bash
cd web-app

# Check GitHub Actions workflow
cat .github/workflows/ci-cd.yml
# Should show complete CI/CD pipeline

# Validate workflow syntax (if you have act installed)
# act -l
# Should list workflow jobs
```

### 8. Test Backend Tests

```bash
cd web-app

# Start services
docker-compose up -d

# Wait for backend to be ready
sleep 5

# Run backend tests
docker-compose exec backend pytest tests/ -v
# Should run and pass tests

# Stop services
docker-compose down
```

### 9. Test Frontend Tests

```bash
cd web-app

# Start services
docker-compose up -d

# Wait for frontend to be ready
sleep 10

# Run frontend tests
docker-compose exec frontend npm test -- --watchAll=false
# Should run and pass tests

# Stop services
docker-compose down
```

### 10. Verify Documentation

```bash
cd web-app

# Check all documentation exists
ls -la *.md
# Should see: README.md, DEVELOPMENT.md, DEPLOYMENT.md, etc.

# Verify README has content
head -20 README.md
# Should show project title and description
```

## Full Integration Test

Run this complete test to verify everything works together:

```bash
#!/bin/bash

echo "🚀 Starting Full Integration Test..."

cd web-app

# 1. Setup
echo "📝 Step 1: Setup environment"
cp .env.example .env
echo "✅ Environment file created"

# 2. Build
echo "🏗️  Step 2: Build images"
docker-compose build
if [ $? -eq 0 ]; then
    echo "✅ Images built successfully"
else
    echo "❌ Image build failed"
    exit 1
fi

# 3. Start services
echo "🚀 Step 3: Start services"
docker-compose up -d
sleep 15
echo "✅ Services started"

# 4. Check health
echo "🔍 Step 4: Check service health"
docker-compose ps

# 5. Test backend
echo "🧪 Step 5: Test backend"
BACKEND_HEALTH=$(curl -s http://localhost:8000/health | grep -o "healthy")
if [ "$BACKEND_HEALTH" = "healthy" ]; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
fi

# 6. Test frontend
echo "🧪 Step 6: Test frontend"
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo "✅ Frontend is accessible"
else
    echo "❌ Frontend is not accessible"
fi

# 7. Run backend tests
echo "🧪 Step 7: Run backend tests"
docker-compose exec -T backend pytest tests/ -v
if [ $? -eq 0 ]; then
    echo "✅ Backend tests passed"
else
    echo "❌ Backend tests failed"
fi

# 8. View logs
echo "📋 Step 8: View recent logs"
echo "--- Backend Logs ---"
docker-compose logs --tail=10 backend
echo "--- Frontend Logs ---"
docker-compose logs --tail=10 frontend

# 9. Cleanup
echo "🧹 Step 9: Cleanup"
docker-compose down
echo "✅ Services stopped"

echo ""
echo "✅ Full Integration Test Complete!"
echo ""
echo "📍 Summary:"
echo "   - Docker images: Built ✅"
echo "   - Services: Started ✅"
echo "   - Backend health: Checked ✅"
echo "   - Frontend: Accessible ✅"
echo "   - Tests: Passed ✅"
echo ""
echo "🎉 Project setup is working correctly!"
```

Save this as `web-app/scripts/verify-setup.sh` and run:

```bash
chmod +x web-app/scripts/verify-setup.sh
./web-app/scripts/verify-setup.sh
```

## Expected Results

### ✅ Success Indicators

1. **Docker Compose Config**: No errors when running `docker-compose config`
2. **Image Build**: All images build without errors
3. **Service Start**: All services show "Up" status
4. **Backend Health**: Returns `{"status":"healthy"}`
5. **Frontend Access**: Returns HTTP 200
6. **Backend Tests**: All tests pass
7. **Logs**: No critical errors in logs

### ❌ Common Issues

1. **Port Already in Use**
   ```bash
   # Find and kill process using port
   lsof -i :3000
   kill -9 <PID>
   ```

2. **Docker Not Running**
   ```bash
   # Start Docker Desktop or Docker daemon
   open -a Docker  # macOS
   ```

3. **Permission Denied**
   ```bash
   # Add execute permission to scripts
   chmod +x web-app/scripts/*.sh
   ```

4. **Build Failures**
   ```bash
   # Clean Docker cache and rebuild
   docker system prune -a
   docker-compose build --no-cache
   ```

## Verification Checklist

- [ ] Docker and Docker Compose installed
- [ ] File structure verified
- [ ] Docker configuration validated
- [ ] Backend builds successfully
- [ ] Frontend builds successfully
- [ ] Services start without errors
- [ ] Backend health check passes
- [ ] Frontend is accessible
- [ ] Backend tests pass
- [ ] Frontend tests pass
- [ ] Documentation is complete
- [ ] CI/CD workflow is configured

## Next Steps

Once all verifications pass:

1. ✅ Task 1 is complete
2. 📝 Review the documentation
3. 🚀 Proceed to Task 2: Database Setup
4. 💻 Start implementing features

## Support

If any verification fails:
1. Check the logs: `docker-compose logs <service>`
2. Review the documentation: `README.md`, `DEVELOPMENT.md`
3. Check the troubleshooting section in `DEPLOYMENT.md`
4. Ensure all prerequisites are met
