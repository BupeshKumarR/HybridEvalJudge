# Monitoring Setup

This directory contains configuration for optional monitoring tools.

## Built-in Monitoring

The application includes built-in monitoring endpoints:

### Health Checks

```bash
# Basic health check (for load balancers)
curl http://localhost:8000/health

# Detailed health check (includes component status)
curl http://localhost:8000/health/detailed

# Application metrics
curl http://localhost:8000/metrics
```

### Logs

View logs using Docker Compose:

```bash
# All services
make logs

# Specific service
make logs-backend
make logs-frontend
make logs-postgres
```

## Sentry Error Tracking

To enable Sentry error tracking:

1. Sign up for Sentry at https://sentry.io
2. Create a new project
3. Add your DSN to `.env`:

```bash
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
SENTRY_TRACES_SAMPLE_RATE=0.1  # Sample 10% of transactions
```

4. Restart the backend service

## Prometheus + Grafana (Optional)

For advanced monitoring with Prometheus and Grafana:

### 1. Add to docker-compose.yml

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: llm-judge-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    ports:
      - "9090:9090"
    networks:
      - llm-judge-network

  grafana:
    image: grafana/grafana:latest
    container_name: llm-judge-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
    networks:
      - llm-judge-network

volumes:
  prometheus_data:
  grafana_data:
```

### 2. Start services

```bash
docker-compose up -d prometheus grafana
```

### 3. Access dashboards

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

### 4. Configure Grafana

1. Add Prometheus as a data source:
   - URL: http://prometheus:9090
2. Import dashboards or create custom ones

## Application Performance Monitoring (APM)

For detailed application performance monitoring, consider:

### New Relic

```bash
# Add to requirements.txt
newrelic

# Add to .env
NEW_RELIC_LICENSE_KEY=your-key
NEW_RELIC_APP_NAME=llm-judge-auditor

# Run with New Relic
newrelic-admin run-program gunicorn app.main:app
```

### Datadog

```bash
# Add to docker-compose.yml
  datadog:
    image: datadog/agent:latest
    environment:
      - DD_API_KEY=${DD_API_KEY}
      - DD_SITE=datadoghq.com
      - DD_LOGS_ENABLED=true
      - DD_APM_ENABLED=true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - /proc/:/host/proc/:ro
      - /sys/fs/cgroup/:/host/sys/fs/cgroup:ro
```

## Log Aggregation

### ELK Stack (Elasticsearch, Logstash, Kibana)

For centralized log management:

```bash
# Add to docker-compose.yml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Alerting

### Prometheus Alertmanager

Create alert rules in `monitoring/alerts.yml`:

```yaml
groups:
  - name: llm_judge_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

### Uptime Monitoring

Use external services for uptime monitoring:
- UptimeRobot (free tier available)
- Pingdom
- StatusCake

## Best Practices

1. **Set up alerts** for critical metrics:
   - High error rates
   - Database connection failures
   - High memory/CPU usage
   - Slow response times

2. **Monitor key metrics**:
   - Request rate and latency
   - Error rate
   - Database query performance
   - Cache hit rate
   - System resources (CPU, memory, disk)

3. **Log retention**:
   - Keep logs for at least 30 days
   - Archive older logs to S3 or similar
   - Use log rotation to manage disk space

4. **Regular reviews**:
   - Review error logs weekly
   - Check performance trends monthly
   - Update alert thresholds as needed

5. **Security monitoring**:
   - Monitor failed authentication attempts
   - Track unusual API usage patterns
   - Set up alerts for security events
