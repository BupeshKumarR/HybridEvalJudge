# Quick Reference Guide

Essential commands for deploying and managing the LLM Judge Auditor Web Application.

## 🚀 Quick Start

```bash
# Development
cp .env.example .env
make dev

# Production
cp .env.example .env
# Edit .env with production values
make prod
```

## 📦 Docker Commands

```bash
# Build images
make build

# Start services
make up              # or: docker-compose up -d

# Stop services
make down            # or: docker-compose down

# Restart services
make restart         # or: docker-compose restart

# View logs
make logs            # All services
make logs-backend    # Backend only
make logs-frontend   # Frontend only
```

## 🏥 Health & Monitoring

```bash
# Check health
make health

# View metrics
curl http://localhost:8000/metrics | python -m json.tool

# Monitor resources
make monitor

# View detailed health
curl http://localhost:8000/health/detailed | python -m json.tool
```

## 🗄️ Database

```bash
# Run migrations
make db-migrate

# Rollback migration
make db-rollback

# Database shell
make db-shell

# Backup
make backup-db

# Restore
make restore-db FILE=backup.sql
```

## 🔐 SSL/TLS

```bash
# Generate self-signed cert (dev)
make ssl-dev

# Let's Encrypt (production)
sudo certbot --nginx -d your-domain.com
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
```

## 🐚 Shell Access

```bash
# Backend shell
make shell-backend   # or: docker-compose exec backend /bin/bash

# Frontend shell
make shell-frontend  # or: docker-compose exec frontend /bin/sh

# Database shell
make shell-postgres  # or: docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor
```

## 🧪 Testing

```bash
# Run all tests
make test

# Backend tests only
docker-compose exec backend pytest

# Frontend tests only
docker-compose exec frontend npm test -- --watchAll=false
```

## 🔧 Maintenance

```bash
# Update dependencies
make update-deps

# Security scan
make security-scan

# Clean up
make clean           # WARNING: Removes all data!

# View running containers
docker-compose ps

# View resource usage
docker stats
```

## 📊 Logs

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100 backend

# Specific time range
docker-compose logs --since 2024-01-01T00:00:00 backend

# Save logs to file
docker-compose logs > logs.txt
```

## 🌐 Access Points

### Development
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### Production
- Application: http://localhost (or https://your-domain.com)
- Health Check: http://localhost/health

## 🔥 Troubleshooting

```bash
# Service not starting?
docker-compose logs <service-name>

# Port conflict?
lsof -i :8000
# Change port in .env

# Out of memory?
docker stats
# Reduce WORKERS in .env

# Database connection error?
docker-compose ps postgres
docker-compose logs postgres

# Reset everything
make down
make clean
make dev
```

## 📝 Environment Variables

Key variables in `.env`:

```bash
# Database
POSTGRES_PASSWORD=changeme

# Backend
SECRET_KEY=your-secret-key
WORKERS=4
LOG_LEVEL=info

# Frontend
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
```

## 🔒 Security Checklist

- [ ] Change default passwords
- [ ] Generate strong SECRET_KEY
- [ ] Configure SSL/TLS
- [ ] Enable Sentry error tracking
- [ ] Set up firewall rules
- [ ] Configure rate limiting
- [ ] Enable audit logging
- [ ] Regular security updates

## 📈 Performance Tips

```bash
# Adjust workers (CPU cores * 2)
WORKERS=8

# Enable Redis caching
REDIS_URL=redis://redis:6379/0

# Monitor performance
make monitor

# Check slow queries
docker-compose exec postgres psql -U llm_judge_user -d llm_judge_auditor
# SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;
```

## 🆘 Emergency Commands

```bash
# Stop everything immediately
docker-compose down

# Restart specific service
docker-compose restart backend

# View recent errors
docker-compose logs --tail=50 backend | grep ERROR

# Check disk space
df -h

# Check memory
free -h

# Force rebuild
docker-compose up -d --build --force-recreate
```

## 📚 Documentation

- Full deployment guide: `DEPLOYMENT.md`
- Monitoring setup: `monitoring/README.md`
- SSL setup: `nginx/ssl/README.md`
- API documentation: http://localhost:8000/docs

## 🎯 Common Workflows

### Deploy to Production

```bash
1. cp .env.example .env
2. Edit .env with production values
3. make ssl-dev  # or configure Let's Encrypt
4. make prod
5. make health
```

### Update Application

```bash
1. git pull origin main
2. make down
3. make build
4. make up
5. make db-migrate
6. make health
```

### Backup and Restore

```bash
# Backup
make backup-db

# Restore
make down
make up
make restore-db FILE=backup_20240101_120000.sql
```

### Debug Issues

```bash
1. make logs
2. make health
3. docker-compose ps
4. docker stats
5. Check DEPLOYMENT.md troubleshooting section
```

## 💡 Tips

- Use `make help` to see all available commands
- Check logs first when troubleshooting
- Monitor resource usage regularly
- Set up automated backups
- Test restore procedures
- Keep documentation updated
- Use health checks in load balancers
- Enable monitoring and alerting

## 🔗 Useful Links

- Docker Docs: https://docs.docker.com
- Docker Compose: https://docs.docker.com/compose
- FastAPI: https://fastapi.tiangolo.com
- React: https://react.dev
- Nginx: https://nginx.org/en/docs
- PostgreSQL: https://www.postgresql.org/docs
- Redis: https://redis.io/docs
- Sentry: https://docs.sentry.io

---

For detailed information, see `DEPLOYMENT.md`
