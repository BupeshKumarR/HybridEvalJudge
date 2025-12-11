# Deployment Guide

This comprehensive guide covers deploying the LLM Judge Auditor Web Application to various environments, from local development to production deployment.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Database Setup](#database-setup)
4. [Docker Setup](#docker-setup)
5. [Local Deployment](#local-deployment)
6. [Production Deployment](#production-deployment)
7. [Cloud Deployment](#cloud-deployment)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Security Best Practices](#security-best-practices)

## Prerequisites

### Required

- Docker 20.10+
- Docker Compose 2.0+
- Domain name (for production)
- SSL certificates (for production)

### Recommended

- Reverse proxy (Nginx)
- Monitoring tools (Prometheus, Grafana)
- Log aggregation (ELK stack)
- Backup solution

## Environment Configuration

### Environment Variables

The application uses environment variables for configuration. These should be set in a `.env` file in the root of the `web-app` directory.

#### Create Environment File

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

#### Required Variables

```env
# Database Configuration
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=postgresql://llm_judge_user:<password>@postgres:5432/llm_judge_auditor

# Redis Configuration (optional but recommended)
REDIS_URL=redis://redis:6379/0

# Security Keys
SECRET_KEY=<generate-strong-secret>
JWT_SECRET_KEY=<generate-strong-jwt-secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Environment
ENVIRONMENT=production  # development, staging, or production
LOG_LEVEL=warning       # debug, info, warning, error, critical

# CORS Configuration
CORS_ORIGINS=https://your-domain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Frontend Configuration
REACT_APP_API_URL=https://your-domain.com
REACT_APP_WS_URL=wss://your-domain.com
```

#### Generate Secure Secrets

**Using Python:**
```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Using OpenSSL:**
```bash
# Generate SECRET_KEY
openssl rand -base64 32

# Generate JWT_SECRET_KEY
openssl rand -base64 32
```

**Using /dev/urandom:**
```bash
# Generate SECRET_KEY
head -c 32 /dev/urandom | base64

# Generate JWT_SECRET_KEY
head -c 32 /dev/urandom | base64
```

#### Environment-Specific Configuration

**Development:**
```env
ENVIRONMENT=development
LOG_LEVEL=debug
CORS_ORIGINS=http://localhost:3000,http://localhost:80
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

**Staging:**
```env
ENVIRONMENT=staging
LOG_LEVEL=info
CORS_ORIGINS=https://staging.your-domain.com
REACT_APP_API_URL=https://staging.your-domain.com
REACT_APP_WS_URL=wss://staging.your-domain.com
```

**Production:**
```env
ENVIRONMENT=production
LOG_LEVEL=warning
CORS_ORIGINS=https://your-domain.com
REACT_APP_API_URL=https://your-domain.com
REACT_APP_WS_URL=wss://your-domain.com
```

## Database Setup

### PostgreSQL Installation

#### Using Docker (Recommended)

The database is automatically set up when using Docker Compose. Skip to [Docker Setup](#docker-setup).

#### Manual Installation

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**macOS:**
```bash
brew install postgresql@15
brew services start postgresql@15
```

**Windows:**
Download and install from [PostgreSQL Downloads](https://www.postgresql.org/download/windows/)

### Create Database and User

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create user
CREATE USER llm_judge_user WITH PASSWORD 'your-secure-password';

# Create database
CREATE DATABASE llm_judge_auditor OWNER llm_judge_user;

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE llm_judge_auditor TO llm_judge_user;

# Exit
\q
```

### Run Migrations

```bash
cd web-app/backend

# Install dependencies
pip install -r requirements.txt

# Set database URL
export DATABASE_URL="postgresql://llm_judge_user:password@localhost:5432/llm_judge_auditor"

# Run migrations
alembic upgrade head
```

### Verify Database Setup

```bash
# Connect to database
psql -U llm_judge_user -d llm_judge_auditor

# List tables
\dt

# Expected output:
#  Schema |        Name         | Type  |      Owner      
# --------+---------------------+-------+-----------------
#  public | evaluation_sessions | table | llm_judge_user
#  public | flagged_issues      | table | llm_judge_user
#  public | judge_results       | table | llm_judge_user
#  public | session_metadata    | table | llm_judge_user
#  public | user_preferences    | table | llm_judge_user
#  public | users               | table | llm_judge_user
#  public | verifier_verdicts   | table | llm_judge_user

# Exit
\q
```

For more details, see [DATABASE_SETUP.md](backend/DATABASE_SETUP.md).

## Docker Setup

### Install Docker

#### Ubuntu/Debian

```bash
# Update package index
sudo apt update

# Install prerequisites
sudo apt install apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

#### macOS

```bash
# Using Homebrew
brew install --cask docker

# Or download Docker Desktop from:
# https://www.docker.com/products/docker-desktop
```

#### Windows

Download and install Docker Desktop from [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)

### Install Docker Compose

#### Linux

```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

#### macOS/Windows

Docker Compose is included with Docker Desktop.

### Verify Docker Installation

```bash
# Check Docker
docker run hello-world

# Check Docker Compose
docker-compose --version

# Expected output:
# Docker Compose version v2.x.x
```

## Local Deployment

### Quick Start

```bash
# Navigate to web-app directory
cd web-app

# Setup and start (automated)
./scripts/setup-dev.sh

# Or manually
make setup
```

### Manual Setup Steps

```bash
# 1. Copy environment file
cp .env.example .env

# 2. Edit environment variables
nano .env

# 3. Build Docker images
docker-compose build

# 4. Start services
docker-compose up -d

# 5. Wait for services to be ready
sleep 10

# 6. Run database migrations
docker-compose exec backend alembic upgrade head

# 7. Verify services are running
docker-compose ps
```

### Access Points

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **Interactive API Docs:** http://localhost:8000/redoc
- **PostgreSQL:** localhost:5432
- **Redis:** localhost:6379

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Production Deployment

This section covers deploying the application to a production environment.

## Production Deployment

This section covers deploying the application to a production environment.

### Prerequisites

- Server with at least 4GB RAM and 2 CPU cores
- Ubuntu 20.04+ or similar Linux distribution
- Domain name pointing to your server
- SSH access to the server
- Firewall configured (ports 80, 443, 22)

### 1. Prepare Server

#### Update System

```bash
# Update package lists
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install essential tools
sudo apt install -y curl git vim ufw
```

#### Configure Firewall

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

#### Install Docker

```bash
# Install Docker using convenience script
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add current user to docker group
sudo usermod -aG docker $USER

# Apply group changes
newgrp docker

# Verify installation
docker --version
```

#### Install Docker Compose

```bash
# Download Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

#### Configure Docker for Production

```bash
# Create Docker daemon configuration
sudo mkdir -p /etc/docker

# Configure logging
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# Restart Docker
sudo systemctl restart docker
```

### 2. Clone Repository

```bash
git clone <repository-url>
cd web-app
```

### 3. Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env

# Set production values
ENVIRONMENT=production
LOG_LEVEL=warning
POSTGRES_PASSWORD=<strong-password>
SECRET_KEY=<strong-secret>
JWT_SECRET_KEY=<strong-jwt-secret>
```

### 4. SSL Certificates

#### Option A: Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt install certbot

# Generate certificates
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
```

#### Option B: Self-Signed (Development Only)

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=your-domain.com"
```

### 5. Configure Nginx

Edit `nginx/nginx.conf` to enable HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # ... rest of configuration
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### 6. Build and Deploy

#### Build Production Images

```bash
# Build all images with production target
docker-compose build

# Or build specific services
docker-compose build backend
docker-compose build frontend
```

#### Start Services

```bash
# Start all services in detached mode
docker-compose up -d

# Or with production profile (includes nginx)
docker-compose --profile production up -d

# Wait for services to be ready
sleep 15
```

#### Run Database Migrations

```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Verify migrations
docker-compose exec backend alembic current
```

#### Check Service Status

```bash
# Check all services
docker-compose ps

# Expected output:
# NAME                    STATUS              PORTS
# llm-judge-backend       Up                  0.0.0.0:8000->8000/tcp
# llm-judge-frontend      Up                  0.0.0.0:3000->3000/tcp
# llm-judge-postgres      Up (healthy)        0.0.0.0:5432->5432/tcp
# llm-judge-redis         Up (healthy)        0.0.0.0:6379->6379/tcp
# llm-judge-nginx         Up                  0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
```

### 7. Verify Deployment

#### Health Checks

```bash
# Check backend health
curl https://your-domain.com/api/health

# Expected response:
# {"status":"healthy","timestamp":"2024-01-01T00:00:00Z"}

# Check frontend
curl -I https://your-domain.com

# Expected: HTTP/1.1 200 OK
```

#### Test API Endpoints

```bash
# Check API documentation
curl https://your-domain.com/api/docs

# Test authentication endpoint
curl -X POST https://your-domain.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'
```

#### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

#### Monitor Resources

```bash
# Check resource usage
docker stats

# Check disk usage
docker system df

# Check specific container
docker stats llm-judge-backend
```

### 8. Post-Deployment Tasks

#### Create Admin User

```bash
# Access backend container
docker-compose exec backend python

# In Python shell:
from app.database import SessionLocal
from app.models import User
from app.auth import get_password_hash

db = SessionLocal()
admin = User(
    username="admin",
    email="admin@example.com",
    password_hash=get_password_hash("secure-password")
)
db.add(admin)
db.commit()
exit()
```

#### Configure Backup

```bash
# Create backup script
cat > /usr/local/bin/backup-llm-judge.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/llm-judge"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker-compose exec -T postgres pg_dump -U llm_judge_user llm_judge_auditor > $BACKUP_DIR/db_$DATE.sql

# Backup environment file
cp .env $BACKUP_DIR/env_$DATE

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete

echo "Backup completed: $DATE"
EOF

# Make executable
sudo chmod +x /usr/local/bin/backup-llm-judge.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-llm-judge.sh") | crontab -
```

#### Setup Log Rotation

```bash
# Create logrotate configuration
sudo tee /etc/logrotate.d/llm-judge > /dev/null << 'EOF'
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    missingok
    delaycompress
    copytruncate
}
EOF
```

## Cloud Deployment

### AWS Deployment

#### Using EC2

**1. Launch EC2 Instance:**
```bash
# Recommended instance type: t3.medium or larger
# AMI: Ubuntu Server 20.04 LTS
# Storage: 30GB+ EBS volume
# Security Group: Allow ports 22, 80, 443
```

**2. Connect and Setup:**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Follow production deployment steps above
```

**3. Configure Elastic IP:**
```bash
# Allocate and associate Elastic IP in AWS Console
# Update DNS records to point to Elastic IP
```

#### Using ECS (Elastic Container Service)

**1. Create ECR Repositories:**
```bash
# Create repositories for backend and frontend
aws ecr create-repository --repository-name llm-judge-backend
aws ecr create-repository --repository-name llm-judge-frontend

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

**2. Build and Push Images:**
```bash
# Tag images
docker tag llm-judge-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-backend:latest
docker tag llm-judge-frontend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-frontend:latest

# Push images
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-frontend:latest
```

**3. Create ECS Task Definition:**
```json
{
  "family": "llm-judge-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-backend:latest",
      "portMappings": [{"containerPort": 8000}],
      "environment": [
        {"name": "DATABASE_URL", "value": "postgresql://..."},
        {"name": "ENVIRONMENT", "value": "production"}
      ]
    }
  ]
}
```

**4. Use RDS for Database:**
```bash
# Create RDS PostgreSQL instance
# Update DATABASE_URL in environment variables
# Configure security groups for ECS to RDS access
```

### Google Cloud Platform (GCP)

#### Using Compute Engine

```bash
# Create VM instance
gcloud compute instances create llm-judge-app \
  --machine-type=e2-medium \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB

# SSH into instance
gcloud compute ssh llm-judge-app

# Follow production deployment steps
```

#### Using Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/llm-judge-backend
gcloud builds submit --tag gcr.io/PROJECT_ID/llm-judge-frontend

# Deploy to Cloud Run
gcloud run deploy llm-judge-backend \
  --image gcr.io/PROJECT_ID/llm-judge-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# Use Cloud SQL for database
gcloud sql instances create llm-judge-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1
```

### Azure Deployment

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name llm-judge-rg --location eastus

# Create container registry
az acr create --resource-group llm-judge-rg --name llmjudgeacr --sku Basic

# Build and push images
az acr build --registry llmjudgeacr --image llm-judge-backend:latest ./backend
az acr build --registry llmjudgeacr --image llm-judge-frontend:latest ./frontend

# Deploy container instances
az container create \
  --resource-group llm-judge-rg \
  --name llm-judge-backend \
  --image llmjudgeacr.azurecr.io/llm-judge-backend:latest \
  --dns-name-label llm-judge-backend \
  --ports 8000

# Use Azure Database for PostgreSQL
az postgres server create \
  --resource-group llm-judge-rg \
  --name llm-judge-db \
  --location eastus \
  --admin-user llm_judge_user \
  --admin-password <password> \
  --sku-name B_Gen5_1
```

### DigitalOcean Deployment

#### Using Droplets

```bash
# Create droplet via web interface or CLI
doctl compute droplet create llm-judge-app \
  --size s-2vcpu-4gb \
  --image ubuntu-20-04-x64 \
  --region nyc1

# SSH and follow production deployment steps
```

#### Using App Platform

```bash
# Create app.yaml
cat > app.yaml << EOF
name: llm-judge-app
services:
- name: backend
  github:
    repo: your-org/llm-judge-auditor
    branch: main
    deploy_on_push: true
  dockerfile_path: web-app/backend/Dockerfile
  http_port: 8000
  instance_count: 1
  instance_size_slug: basic-xs
  
- name: frontend
  github:
    repo: your-org/llm-judge-auditor
    branch: main
    deploy_on_push: true
  dockerfile_path: web-app/frontend/Dockerfile
  http_port: 80
  instance_count: 1
  instance_size_slug: basic-xs

databases:
- name: llm-judge-db
  engine: PG
  version: "15"
EOF

# Deploy
doctl apps create --spec app.yaml
```

### Kubernetes Deployment

#### Create Kubernetes Manifests

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-judge-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-judge-backend
  template:
    metadata:
      labels:
        app: llm-judge-backend
    spec:
      containers:
      - name: backend
        image: your-registry/llm-judge-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: llm-judge-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-judge-backend
spec:
  selector:
    app: llm-judge-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

**Deploy to Kubernetes:**
```bash
# Create namespace
kubectl create namespace llm-judge

# Create secrets
kubectl create secret generic llm-judge-secrets \
  --from-literal=database-url='postgresql://...' \
  --from-literal=secret-key='...' \
  -n llm-judge

# Apply manifests
kubectl apply -f deployment.yaml -n llm-judge

# Check status
kubectl get pods -n llm-judge
kubectl get services -n llm-judge
```

## Monitoring and Maintenance

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Check all services
docker-compose ps
```

### View Logs

```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

### Database Backup

```bash
# Backup database
docker-compose exec postgres pg_dump -U llm_judge_user llm_judge_auditor > backup.sql

# Restore database
docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor < backup.sql
```

### Update Application

```bash
# Pull latest changes
git pull

# Rebuild images
docker-compose build

# Restart services
docker-compose up -d

# Run migrations
docker-compose exec backend alembic upgrade head
```

### Scale Services

```bash
# Scale backend workers
docker-compose up -d --scale backend=3

# Scale with load balancer
# Edit docker-compose.yml to add load balancer configuration
```

### Monitoring Setup

#### Prometheus + Grafana

```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana
  ports:
    - "3001:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Log Aggregation

#### ELK Stack

```yaml
# Add to docker-compose.yml
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
  environment:
    - discovery.type=single-node

logstash:
  image: docker.elastic.co/logstash/logstash:8.11.0
  volumes:
    - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf

kibana:
  image: docker.elastic.co/kibana/kibana:8.11.0
  ports:
    - "5601:5601"
```

## Troubleshooting

### Common Issues and Solutions

#### Service Won't Start

**Problem:** Container fails to start or exits immediately.

**Solutions:**
```bash
# 1. Check logs for error messages
docker-compose logs <service-name>

# 2. Verify configuration syntax
docker-compose config

# 3. Check environment variables
docker-compose exec <service-name> env

# 4. Rebuild service from scratch
docker-compose build --no-cache <service-name>
docker-compose up -d <service-name>

# 5. Check for port conflicts
sudo lsof -i :<port-number>
```

#### Database Connection Issues

**Problem:** Backend cannot connect to PostgreSQL.

**Solutions:**
```bash
# 1. Check PostgreSQL is running
docker-compose ps postgres

# 2. View PostgreSQL logs
docker-compose logs postgres

# 3. Verify connection from backend
docker-compose exec backend python -c "
from app.database import engine
try:
    engine.connect()
    print('Connection successful')
except Exception as e:
    print(f'Connection failed: {e}')
"

# 4. Check DATABASE_URL format
# Correct: postgresql://user:password@host:port/database
# Verify in .env file

# 5. Reset database if corrupted
docker-compose down -v
docker-compose up -d postgres
sleep 5
docker-compose exec backend alembic upgrade head
```

#### Frontend Build Errors

**Problem:** Frontend fails to build or shows blank page.

**Solutions:**
```bash
# 1. Clear node_modules and rebuild
docker-compose exec frontend rm -rf node_modules package-lock.json
docker-compose exec frontend npm install

# 2. Check for JavaScript errors in browser console
# Open browser DevTools (F12) and check Console tab

# 3. Verify API URL configuration
docker-compose exec frontend env | grep REACT_APP

# 4. Rebuild frontend image
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

#### SSL Certificate Issues

**Problem:** HTTPS not working or certificate errors.

**Solutions:**
```bash
# 1. Verify certificate files exist
ls -la nginx/ssl/

# 2. Check certificate validity
openssl x509 -in nginx/ssl/cert.pem -text -noout | grep -A2 Validity

# 3. Test certificate chain
openssl verify -CAfile nginx/ssl/cert.pem nginx/ssl/cert.pem

# 4. Renew Let's Encrypt certificates
sudo certbot renew --dry-run  # Test renewal
sudo certbot renew            # Actual renewal

# 5. Restart nginx after certificate update
docker-compose restart nginx
```

#### WebSocket Connection Failures

**Problem:** Real-time updates not working.

**Solutions:**
```bash
# 1. Check WebSocket endpoint
curl -i -N -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  http://localhost:8000/socket.io/

# 2. Verify nginx WebSocket configuration
docker-compose exec nginx cat /etc/nginx/conf.d/default.conf | grep -A10 socket.io

# 3. Check CORS settings
# Ensure CORS_ORIGINS includes your frontend URL

# 4. Test from browser console
# Open DevTools and check Network tab for WebSocket connections
```

#### Performance Issues

**Problem:** Application is slow or unresponsive.

**Solutions:**
```bash
# 1. Check resource usage
docker stats

# 2. Check database performance
docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE state = 'active' 
ORDER BY duration DESC;
"

# 3. Optimize database
docker-compose exec postgres vacuumdb -U llm_judge_user -d llm_judge_auditor --analyze

# 4. Scale backend workers
docker-compose up -d --scale backend=3

# 5. Enable Redis caching
# Ensure REDIS_URL is set in .env
# Restart backend to apply changes

# 6. Check for memory leaks
docker stats --no-stream | grep llm-judge
```

#### Migration Errors

**Problem:** Database migrations fail.

**Solutions:**
```bash
# 1. Check current migration status
docker-compose exec backend alembic current

# 2. View migration history
docker-compose exec backend alembic history

# 3. Rollback one version
docker-compose exec backend alembic downgrade -1

# 4. Force to specific version
docker-compose exec backend alembic stamp head

# 5. Reset and reapply all migrations
docker-compose exec backend alembic downgrade base
docker-compose exec backend alembic upgrade head
```

#### Disk Space Issues

**Problem:** Running out of disk space.

**Solutions:**
```bash
# 1. Check disk usage
df -h
docker system df

# 2. Remove unused images
docker image prune -a

# 3. Remove unused volumes
docker volume prune

# 4. Remove stopped containers
docker container prune

# 5. Clean everything (WARNING: removes all unused data)
docker system prune -a --volumes

# 6. Check log file sizes
du -sh /var/lib/docker/containers/*/*-json.log
```

#### Authentication Issues

**Problem:** Cannot login or token errors.

**Solutions:**
```bash
# 1. Verify JWT_SECRET_KEY is set
docker-compose exec backend python -c "
from app.core.config import settings
print(f'JWT Secret: {settings.JWT_SECRET_KEY[:10]}...')
"

# 2. Check token expiration settings
# Verify ACCESS_TOKEN_EXPIRE_MINUTES in .env

# 3. Clear browser cookies and try again

# 4. Test authentication endpoint
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'

# 5. Reset user password
docker-compose exec backend python
# In Python shell:
from app.database import SessionLocal
from app.models import User
from app.auth import get_password_hash
db = SessionLocal()
user = db.query(User).filter(User.username == "test").first()
user.password_hash = get_password_hash("newpassword")
db.commit()
```

## Security Best Practices

### Pre-Deployment Security Checklist

- [ ] **Strong Passwords:** All service passwords are strong and unique
- [ ] **Secret Keys:** SECRET_KEY and JWT_SECRET_KEY are randomly generated
- [ ] **SSL/TLS:** HTTPS is enabled with valid certificates
- [ ] **Firewall:** Only necessary ports (80, 443, 22) are open
- [ ] **Rate Limiting:** API rate limits are configured
- [ ] **CORS:** CORS_ORIGINS is set to specific domains only
- [ ] **Security Headers:** X-Frame-Options, CSP, etc. are configured
- [ ] **Database:** PostgreSQL is not exposed to public internet
- [ ] **Environment Variables:** Sensitive data is in .env, not in code
- [ ] **Docker:** Running containers as non-root users

### Hardening Steps

#### 1. Secure Environment Variables

```bash
# Set restrictive permissions on .env file
chmod 600 .env

# Never commit .env to version control
echo ".env" >> .gitignore

# Use secrets management in production
# AWS: AWS Secrets Manager
# GCP: Secret Manager
# Azure: Key Vault
```

#### 2. Configure Security Headers

Add to `nginx/nginx.conf`:
```nginx
# Security headers
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

#### 3. Enable Rate Limiting

Already configured in nginx.conf, but verify:
```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=general_limit:10m rate=30r/s;
```

Adjust rates based on your needs:
```nginx
# Stricter limits for production
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=5r/s;
limit_req_zone $binary_remote_addr zone=login_limit:10m rate=1r/m;
```

#### 4. Secure PostgreSQL

```bash
# Change default password
docker-compose exec postgres psql -U postgres -c "
ALTER USER llm_judge_user WITH PASSWORD 'new-strong-password';
"

# Restrict network access
# In docker-compose.yml, remove ports exposure:
# ports:
#   - "5432:5432"  # Remove this line

# Use internal Docker network only
```

#### 5. Implement Fail2Ban

```bash
# Install fail2ban
sudo apt install fail2ban

# Create jail for nginx
sudo tee /etc/fail2ban/jail.d/nginx.conf > /dev/null << 'EOF'
[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 5
findtime = 600
bantime = 3600
EOF

# Restart fail2ban
sudo systemctl restart fail2ban
```

#### 6. Enable Audit Logging

Add to backend configuration:
```python
# app/middleware/audit.py
import logging
from fastapi import Request

audit_logger = logging.getLogger("audit")

async def audit_middleware(request: Request, call_next):
    # Log all requests
    audit_logger.info(f"{request.method} {request.url} - {request.client.host}")
    response = await call_next(request)
    return response
```

#### 7. Regular Security Updates

```bash
# Create update script
cat > /usr/local/bin/update-llm-judge.sh << 'EOF'
#!/bin/bash
set -e

echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Pulling latest images..."
cd /path/to/web-app
docker-compose pull

echo "Rebuilding services..."
docker-compose build

echo "Restarting services..."
docker-compose up -d

echo "Running migrations..."
docker-compose exec backend alembic upgrade head

echo "Update complete!"
EOF

chmod +x /usr/local/bin/update-llm-judge.sh

# Schedule weekly updates (Sundays at 3 AM)
(crontab -l 2>/dev/null; echo "0 3 * * 0 /usr/local/bin/update-llm-judge.sh") | crontab -
```

#### 8. Implement Intrusion Detection

```bash
# Install AIDE (Advanced Intrusion Detection Environment)
sudo apt install aide

# Initialize database
sudo aideinit

# Check for changes
sudo aide --check
```

#### 9. Secure SSH Access

```bash
# Disable password authentication
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Disable root login
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config

# Change default SSH port (optional)
sudo sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config

# Restart SSH
sudo systemctl restart sshd
```

#### 10. Enable Two-Factor Authentication

For application-level 2FA, implement in backend:
```python
# Install pyotp
pip install pyotp

# Add to user model
class User(Base):
    # ... existing fields
    totp_secret = Column(String, nullable=True)
    two_factor_enabled = Column(Boolean, default=False)
```

### Security Monitoring

#### 1. Setup Log Monitoring

```bash
# Install logwatch
sudo apt install logwatch

# Configure daily reports
sudo logwatch --detail High --mailto admin@example.com --service all --range today
```

#### 2. Monitor Failed Login Attempts

```bash
# Check authentication logs
docker-compose logs backend | grep "authentication failed"

# Set up alerts
# Use tools like Sentry, Datadog, or custom scripts
```

#### 3. Vulnerability Scanning

```bash
# Scan Docker images
docker scan llm-judge-backend:latest
docker scan llm-judge-frontend:latest

# Use Trivy for comprehensive scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image llm-judge-backend:latest
```

### Compliance Considerations

#### GDPR Compliance

- Implement data retention policies
- Add user data export functionality
- Implement right to be forgotten (data deletion)
- Add privacy policy and terms of service
- Log consent for data processing

#### HIPAA Compliance (if handling health data)

- Encrypt data at rest and in transit
- Implement audit logging
- Add access controls and authentication
- Regular security assessments
- Business Associate Agreements (BAAs)

### Security Incident Response

#### Incident Response Plan

1. **Detection:** Monitor logs and alerts
2. **Containment:** Isolate affected systems
3. **Investigation:** Analyze logs and determine scope
4. **Eradication:** Remove threat and patch vulnerabilities
5. **Recovery:** Restore services from backups
6. **Post-Incident:** Document and improve processes

#### Emergency Procedures

```bash
# Immediately stop all services
docker-compose down

# Backup current state
docker-compose exec postgres pg_dump -U llm_judge_user llm_judge_auditor > emergency_backup.sql

# Review logs
docker-compose logs > incident_logs.txt

# Restore from known good backup
docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor < last_good_backup.sql

# Update all secrets
# Generate new SECRET_KEY and JWT_SECRET_KEY
# Update .env file

# Restart with new configuration
docker-compose up -d
```

## Rollback Procedure

```bash
# Stop current deployment
docker-compose down

# Checkout previous version
git checkout <previous-commit>

# Rebuild and deploy
docker-compose build
docker-compose up -d

# Restore database if needed
docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor < backup.sql
```

## Additional Resources

### Documentation

- [Development Guide](DEVELOPMENT.md) - Local development setup and workflows
- [API Documentation](docs/API_DOCUMENTATION.md) - Complete API reference
- [User Guide](docs/USER_GUIDE.md) - End-user documentation
- [Database Setup](backend/DATABASE_SETUP.md) - Detailed database documentation
- [WebSocket Events](docs/WEBSOCKET_EVENTS.md) - Real-time communication reference

### Useful Commands Reference

#### Docker Commands

```bash
# View all containers
docker ps -a

# View all images
docker images

# Remove all stopped containers
docker container prune

# Remove all unused images
docker image prune -a

# View container logs
docker logs <container-id>

# Execute command in container
docker exec -it <container-id> /bin/bash

# Copy files from container
docker cp <container-id>:/path/to/file ./local/path

# View container resource usage
docker stats

# Inspect container
docker inspect <container-id>
```

#### Docker Compose Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart <service-name>

# View logs
docker-compose logs -f <service-name>

# Scale service
docker-compose up -d --scale backend=3

# Rebuild service
docker-compose build <service-name>

# Pull latest images
docker-compose pull

# Validate configuration
docker-compose config

# List services
docker-compose ps
```

#### Database Commands

```bash
# Connect to database
docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor

# Backup database
docker-compose exec postgres pg_dump -U llm_judge_user llm_judge_auditor > backup.sql

# Restore database
docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor < backup.sql

# Run migrations
docker-compose exec backend alembic upgrade head

# Rollback migration
docker-compose exec backend alembic downgrade -1

# View migration history
docker-compose exec backend alembic history

# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "Description"
```

#### Nginx Commands

```bash
# Test configuration
docker-compose exec nginx nginx -t

# Reload configuration
docker-compose exec nginx nginx -s reload

# View access logs
docker-compose exec nginx tail -f /var/log/nginx/access.log

# View error logs
docker-compose exec nginx tail -f /var/log/nginx/error.log
```

### Performance Tuning

#### PostgreSQL Optimization

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM evaluation_sessions WHERE user_id = 'xxx';

-- Update statistics
ANALYZE;

-- Vacuum database
VACUUM ANALYZE;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

#### Backend Optimization

```python
# Enable connection pooling
# In app/database.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Enable response caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```

#### Frontend Optimization

```javascript
// Enable code splitting
const LazyComponent = React.lazy(() => import('./Component'));

// Memoize expensive computations
const memoizedValue = useMemo(() => computeExpensiveValue(a, b), [a, b]);

// Debounce search inputs
const debouncedSearch = useMemo(
  () => debounce((value) => performSearch(value), 300),
  []
);
```

### Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-judge-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "LLM Judge Auditor Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

### Backup and Disaster Recovery

#### Automated Backup Script

```bash
#!/bin/bash
# /usr/local/bin/backup-llm-judge-full.sh

set -e

BACKUP_DIR="/var/backups/llm-judge"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p $BACKUP_DIR

echo "Starting backup: $DATE"

# Backup database
echo "Backing up database..."
docker-compose exec -T postgres pg_dump -U llm_judge_user llm_judge_auditor | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup environment configuration
echo "Backing up configuration..."
cp .env $BACKUP_DIR/env_$DATE
cp docker-compose.yml $BACKUP_DIR/docker-compose_$DATE.yml

# Backup uploaded files (if any)
if [ -d "./uploads" ]; then
    echo "Backing up uploads..."
    tar -czf $BACKUP_DIR/uploads_$DATE.tar.gz ./uploads
fi

# Remove old backups
echo "Cleaning old backups..."
find $BACKUP_DIR -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "env_*" -mtime +$RETENTION_DAYS -delete

# Upload to S3 (optional)
if command -v aws &> /dev/null; then
    echo "Uploading to S3..."
    aws s3 sync $BACKUP_DIR s3://your-backup-bucket/llm-judge/
fi

echo "Backup completed: $DATE"
```

#### Disaster Recovery Procedure

```bash
# 1. Stop all services
docker-compose down

# 2. Restore database
gunzip < /var/backups/llm-judge/db_YYYYMMDD_HHMMSS.sql.gz | \
  docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor

# 3. Restore configuration
cp /var/backups/llm-judge/env_YYYYMMDD_HHMMSS .env

# 4. Restore uploads (if any)
tar -xzf /var/backups/llm-judge/uploads_YYYYMMDD_HHMMSS.tar.gz

# 5. Start services
docker-compose up -d

# 6. Verify
curl http://localhost:8000/health
```

### Cost Optimization

#### Cloud Cost Reduction Tips

1. **Use Reserved Instances:** Save 30-70% on compute costs
2. **Auto-scaling:** Scale down during low traffic periods
3. **Spot Instances:** Use for non-critical workloads
4. **Storage Optimization:** Use appropriate storage tiers
5. **CDN:** Use CloudFront/CloudFlare for static assets
6. **Database:** Use read replicas for read-heavy workloads
7. **Monitoring:** Set up billing alerts

#### Resource Right-Sizing

```bash
# Monitor actual resource usage
docker stats --no-stream

# Adjust container resources in docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## Support and Community

### Getting Help

For deployment issues:

1. **Check Logs:** `docker-compose logs -f`
2. **Review Configuration:** `docker-compose config`
3. **Consult Documentation:** See links above
4. **Search Issues:** Check GitHub issues for similar problems
5. **Ask Community:** Post in discussions or forums
6. **Open Issue:** Create detailed bug report on GitHub

### Reporting Issues

When reporting deployment issues, include:

- Operating system and version
- Docker and Docker Compose versions
- Relevant logs (`docker-compose logs`)
- Configuration (sanitized .env)
- Steps to reproduce
- Expected vs actual behavior

### Contributing

Contributions are welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Changelog

### Version 1.0.0 (2024-01-01)

- Initial production release
- Docker-based deployment
- PostgreSQL database
- Redis caching
- Nginx reverse proxy
- SSL/TLS support
- Monitoring and logging
- Backup and recovery procedures

---

**Last Updated:** 2024-01-01  
**Maintained By:** LLM Judge Auditor Team  
**License:** MIT


## Docker Images

### Building Images

The application uses multi-stage Docker builds for optimization:

```bash
# Build all images
make build

# Or manually:
docker-compose build --no-cache

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Image Optimization

Our Dockerfiles include several optimizations:

1. **Multi-stage builds**: Separate build and runtime stages
2. **Layer caching**: Dependencies installed before code copy
3. **Non-root users**: Security best practice
4. **.dockerignore**: Excludes unnecessary files
5. **Health checks**: Built-in container health monitoring

### Image Sizes

Approximate sizes after optimization:
- Backend: ~200MB (Python 3.11 slim + dependencies)
- Frontend: ~25MB (Nginx alpine + React build)
- Total: ~225MB

## Docker Compose Configuration

### Development Configuration

The default `docker-compose.yml` is configured for development:

```bash
# Start development environment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Features:
- Hot reload for backend and frontend
- Volume mounts for live code updates
- Exposed ports for direct access
- Debug logging enabled

### Production Configuration

Use `docker-compose.prod.yml` for production:

```bash
# Start production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Or use make:
make prod
```

Features:
- Optimized production builds
- No volume mounts (immutable containers)
- Services only accessible through Nginx
- Production logging levels
- Automatic restarts

### Service Configuration

#### PostgreSQL

```yaml
# Default configuration
POSTGRES_DB: llm_judge_auditor
POSTGRES_USER: llm_judge_user
POSTGRES_PASSWORD: changeme  # CHANGE IN PRODUCTION!
```

Health check: Automatic with pg_isready

#### Redis

```yaml
# Configuration
- Persistence: AOF enabled
- Max memory: 256MB
- Eviction policy: allkeys-lru
```

Health check: redis-cli ping

#### Backend (FastAPI)

```yaml
# Workers: 4 (adjust based on CPU cores)
# Timeout: 300s for long evaluations
# Health check: /health endpoint
```

#### Frontend (React)

```yaml
# Served by Nginx
# Port: 8080 (internal)
# Health check: HTTP GET /
```

## Nginx Configuration

### Reverse Proxy

Nginx acts as a reverse proxy with:

- **Rate limiting**: Protects against abuse
- **SSL/TLS termination**: HTTPS support
- **Static file caching**: Improved performance
- **WebSocket support**: Real-time features
- **Security headers**: XSS, CSRF protection

### Rate Limits

```nginx
# API endpoints: 10 req/s
# Authentication: 5 req/min
# WebSocket: 20 req/s
# General: 30 req/s
```

### SSL/TLS Configuration

#### Development (Self-signed)

```bash
# Generate self-signed certificate
make ssl-dev

# Or manually:
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

#### Production (Let's Encrypt)

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Copy to nginx/ssl directory
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem

# Set permissions
chmod 644 nginx/ssl/cert.pem
chmod 600 nginx/ssl/key.pem

# Auto-renewal (add to crontab)
0 0 * * * certbot renew --quiet
```

#### SSL Security

Our configuration includes:
- TLS 1.2 and 1.3 only
- Strong cipher suites
- HSTS headers
- OCSP stapling
- Perfect forward secrecy

Test your SSL configuration:
- https://www.ssllabs.com/ssltest/

## Monitoring and Logging

### Built-in Monitoring

#### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Application metrics
curl http://localhost:8000/metrics
```

#### Using Make Commands

```bash
# Check all service health
make health

# View logs
make logs              # All services
make logs-backend      # Backend only
make logs-frontend     # Frontend only
make logs-postgres     # Database only
```

### Sentry Error Tracking

Enable Sentry for error tracking:

1. Sign up at https://sentry.io
2. Create a new project
3. Add to `.env`:

```bash
SENTRY_DSN=https://your-key@sentry.io/project-id
SENTRY_TRACES_SAMPLE_RATE=0.1
```

4. Restart backend:

```bash
docker-compose restart backend
```

### Log Management

#### Log Locations

- **Application logs**: `backend/logs/`
- **Container logs**: `docker-compose logs`
- **Nginx logs**: Stored in nginx_logs volume

#### Log Rotation

Logs are automatically rotated:
- Max size: 10MB per file
- Backup count: 5 files
- Total: ~50MB per log type

#### Viewing Logs

```bash
# Real-time logs
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Specific time range
docker-compose logs --since 2024-01-01T00:00:00 backend
```

### Performance Monitoring

Monitor system resources:

```bash
# Docker stats
docker stats

# Continuous monitoring
make monitor

# Or manually:
watch -n 2 'docker-compose ps && docker stats --no-stream'
```

### Optional: Prometheus + Grafana

See `monitoring/README.md` for advanced monitoring setup.

## Database Management

### Migrations

```bash
# Run migrations
make db-migrate

# Or manually:
docker-compose exec backend alembic upgrade head

# Rollback last migration
make db-rollback
```

### Backup

```bash
# Create backup
make backup-db

# Or manually:
docker-compose exec -T postgres pg_dump -U llm_judge_user llm_judge_auditor > backup.sql
```

### Restore

```bash
# Restore from backup
make restore-db FILE=backup.sql

# Or manually:
docker-compose exec -T postgres psql -U llm_judge_user llm_judge_auditor < backup.sql
```

### Database Shell

```bash
# Open PostgreSQL shell
make db-shell

# Or manually:
docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor
```

## Production Deployment Checklist

### Pre-deployment

- [ ] Update `.env` with production values
- [ ] Change all default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Obtain SSL certificates
- [ ] Configure domain DNS
- [ ] Set up backup solution
- [ ] Configure monitoring/alerting
- [ ] Review security settings

### Security

- [ ] Use strong passwords (min 16 characters)
- [ ] Enable HTTPS only
- [ ] Configure firewall rules
- [ ] Set up fail2ban or similar
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Implement rate limiting
- [ ] Use secrets management (e.g., Vault)

### Performance

- [ ] Adjust worker count based on CPU cores
- [ ] Configure database connection pooling
- [ ] Enable Redis caching
- [ ] Set up CDN for static assets (optional)
- [ ] Configure Nginx caching
- [ ] Monitor resource usage

### Reliability

- [ ] Set up automated backups
- [ ] Test backup restoration
- [ ] Configure health checks
- [ ] Set up monitoring alerts
- [ ] Document recovery procedures
- [ ] Plan for scaling

## Cloud Deployment

### AWS

#### Using ECS (Elastic Container Service)

1. Push images to ECR:

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag llm-judge-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/llm-judge-backend:latest
```

2. Create ECS task definitions
3. Set up RDS for PostgreSQL
4. Set up ElastiCache for Redis
5. Configure ALB (Application Load Balancer)
6. Set up CloudWatch for monitoring

#### Using EC2

1. Launch EC2 instance (t3.medium or larger)
2. Install Docker and Docker Compose
3. Clone repository
4. Configure environment
5. Run deployment

### Google Cloud Platform

#### Using Cloud Run

1. Build and push to Container Registry
2. Deploy services to Cloud Run
3. Set up Cloud SQL for PostgreSQL
4. Set up Memorystore for Redis
5. Configure Cloud Load Balancing

### DigitalOcean

#### Using App Platform

1. Connect GitHub repository
2. Configure build settings
3. Set up Managed Database (PostgreSQL)
4. Set up Managed Redis
5. Configure environment variables

### Kubernetes

See `k8s/` directory for Kubernetes manifests (if available).

## Scaling

### Horizontal Scaling

Scale backend workers:

```bash
# Scale to 4 instances
docker-compose up -d --scale backend=4

# With load balancer
# Update nginx upstream configuration
```

### Vertical Scaling

Adjust resources in docker-compose.yml:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Database Scaling

- Enable connection pooling
- Add read replicas
- Implement caching strategy
- Optimize queries

## Maintenance

### Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up -d --build

# Or with zero downtime:
docker-compose up -d --build --no-deps backend
```

### Dependency Updates

```bash
# Update all dependencies
make update-deps

# Security audit
make security-scan
```

### Cleanup

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup (WARNING: removes all data)
make clean
```

## Troubleshooting

### Common Issues

#### Services won't start

```bash
# Check logs
docker-compose logs

# Check service status
docker-compose ps

# Restart services
docker-compose restart
```

#### Database connection errors

```bash
# Check database is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Verify connection string in .env
```

#### Out of memory

```bash
# Check resource usage
docker stats

# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Reduce worker count in .env
WORKERS=2
```

#### Port conflicts

```bash
# Check what's using the port
lsof -i :8000

# Change port in .env
BACKEND_PORT=8001
```

### Debug Mode

Enable debug logging:

```bash
# In .env
LOG_LEVEL=debug
ENVIRONMENT=development

# Restart services
docker-compose restart backend
```

### Getting Help

1. Check logs: `make logs`
2. Check health: `make health`
3. Review documentation
4. Check GitHub issues
5. Contact support

## Security Best Practices

### Secrets Management

Never commit secrets to version control:

```bash
# Add to .gitignore
.env
*.pem
*.key
```

Use environment variables or secrets management:
- AWS Secrets Manager
- HashiCorp Vault
- Docker Secrets

### Network Security

```bash
# Configure firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Restrict database access
# Only allow from backend container
```

### Regular Updates

```bash
# Update base images
docker-compose pull

# Update dependencies
make update-deps

# Security patches
apt-get update && apt-get upgrade
```

### Audit Logging

All authentication and evaluation requests are logged.

Review logs regularly:

```bash
# Check for suspicious activity
docker-compose logs backend | grep "authentication"
docker-compose logs backend | grep "ERROR"
```

## Performance Optimization

### Backend

- Adjust worker count: `WORKERS=<cpu_cores * 2>`
- Enable connection pooling
- Use Redis caching
- Optimize database queries

### Frontend

- Enable Nginx caching
- Use CDN for static assets
- Enable compression (gzip/brotli)
- Optimize bundle size

### Database

- Add indexes for frequent queries
- Regular VACUUM and ANALYZE
- Monitor slow queries
- Consider read replicas

## Backup Strategy

### Automated Backups

Set up cron job:

```bash
# Add to crontab
0 2 * * * cd /path/to/web-app && make backup-db
```

### Backup Retention

- Daily backups: Keep 7 days
- Weekly backups: Keep 4 weeks
- Monthly backups: Keep 12 months

### Disaster Recovery

1. Document recovery procedures
2. Test restoration regularly
3. Store backups off-site
4. Maintain backup of configuration

## Support

For issues or questions:
- GitHub Issues: <repository-url>/issues
- Documentation: See docs/ directory
- Email: support@example.com
