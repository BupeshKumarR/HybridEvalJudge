# Deployment Guide

This guide covers deploying the LLM Judge Auditor Web Application to various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

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

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

### Required Variables

```env
# Database
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=postgresql://llm_judge_user:<password>@postgres:5432/llm_judge_auditor

# Security
SECRET_KEY=<generate-strong-secret>
JWT_SECRET_KEY=<generate-strong-jwt-secret>

# Environment
ENVIRONMENT=production
LOG_LEVEL=warning
```

### Generate Secrets

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Local Deployment

### Quick Start

```bash
# Setup and start
./scripts/setup-dev.sh

# Or manually
make setup
```

### Access Points

- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Docker Deployment

### Build Images

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build backend
docker-compose build frontend
```

### Start Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d backend

# View logs
docker-compose logs -f
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Production Deployment

### 1. Prepare Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
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

```bash
# Build production images
docker-compose build --target production

# Start services with production profile
docker-compose --profile production up -d

# Run database migrations
docker-compose exec backend alembic upgrade head

# Check status
docker-compose ps
```

### 7. Verify Deployment

```bash
# Check health
curl https://your-domain.com/health

# Check API docs
curl https://your-domain.com/docs

# View logs
docker-compose logs -f
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

### Service Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Check configuration
docker-compose config

# Rebuild service
docker-compose build <service-name>
docker-compose up -d <service-name>
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Verify connection
docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor

# Reset database
make db-reset
```

### SSL Certificate Issues

```bash
# Verify certificates
openssl x509 -in nginx/ssl/cert.pem -text -noout

# Renew Let's Encrypt certificates
sudo certbot renew
```

### Performance Issues

```bash
# Check resource usage
docker stats

# Scale services
docker-compose up -d --scale backend=3

# Optimize database
docker-compose exec postgres vacuumdb -U llm_judge_user -d llm_judge_auditor
```

## Security Checklist

- [ ] Strong passwords for all services
- [ ] SSL/TLS enabled
- [ ] Firewall configured
- [ ] Rate limiting enabled
- [ ] CORS properly configured
- [ ] Security headers set
- [ ] Regular backups configured
- [ ] Monitoring and alerting set up
- [ ] Log rotation configured
- [ ] Regular security updates

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

## Support

For deployment issues:
1. Check logs: `docker-compose logs -f`
2. Review configuration: `docker-compose config`
3. Consult documentation
4. Open an issue on GitHub
