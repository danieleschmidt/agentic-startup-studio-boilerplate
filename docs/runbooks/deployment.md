# Deployment Runbook

This runbook covers deployment procedures for the Agentic Startup Studio application across different environments.

## Overview

The application uses a GitOps approach with automated deployments triggered by:
- **Staging**: Pushes to `main` branch
- **Production**: GitHub releases

## Prerequisites

- Access to the deployment environments
- GitHub repository access with appropriate permissions
- Docker registry access (GitHub Container Registry)
- Kubernetes cluster access (if using K8s deployment)

## Staging Deployment

### Automatic Deployment

Staging deployments are triggered automatically when code is pushed to the `main` branch.

### Manual Staging Deployment

If you need to deploy manually:

```bash
# 1. Ensure you're on the main branch with latest changes
git checkout main
git pull origin main

# 2. Build and tag the image
docker build -t ghcr.io/your-org/agentic-startup-studio:staging .
docker push ghcr.io/your-org/agentic-startup-studio:staging

# 3. Deploy to staging environment
# Option A: Docker Compose
docker-compose -f docker-compose.staging.yml up -d

# Option B: Kubernetes
kubectl apply -f k8s/staging/ -n staging
```

### Staging Verification

After deployment, verify the staging environment:

```bash
# Health check
curl https://staging-api.yourapp.com/health

# API documentation
curl https://staging-api.yourapp.com/docs

# Frontend accessibility
curl https://staging.yourapp.com

# Database connectivity
kubectl exec -it deployment/postgres -n staging -- psql -U postgres -c "SELECT 1"
```

## Production Deployment

### Release Process

1. **Create Release Branch**:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b release/v1.2.3
   ```

2. **Update Version**:
   ```bash
   # Update version in package.json
   npm version 1.2.3
   
   # Update version in pyproject.toml (backend)
   # Update CHANGELOG.md
   ```

3. **Create Release**:
   ```bash
   git push origin release/v1.2.3
   # Create PR to main
   # After merge, create GitHub release
   ```

### Automated Production Deployment

Production deployment is triggered by creating a GitHub release:

1. Go to GitHub repository
2. Click "Releases" → "Create a new release"
3. Choose tag version (e.g., v1.2.3)
4. Add release notes
5. Publish release

This triggers the production deployment workflow.

### Manual Production Deployment

If automated deployment fails, deploy manually:

```bash
# 1. Build production image
docker build -t ghcr.io/your-org/agentic-startup-studio:v1.2.3 .
docker push ghcr.io/your-org/agentic-startup-studio:v1.2.3

# 2. Deploy with blue-green strategy
kubectl apply -f k8s/production/blue/ -n production

# 3. Verify health
kubectl get pods -n production
kubectl logs deployment/app-blue -n production

# 4. Switch traffic
kubectl patch service app-service -n production -p '{"spec":{"selector":{"version":"blue"}}}'

# 5. Clean up old deployment
kubectl delete -f k8s/production/green/ -n production
```

## Environment-Specific Configurations

### Staging Environment

```yaml
# staging.env
NODE_ENV=staging
API_URL=https://staging-api.yourapp.com
DATABASE_URL=postgresql://user:pass@staging-db:5432/appdb
REDIS_URL=redis://staging-redis:6379
LOG_LEVEL=debug
RATE_LIMIT_ENABLED=false
```

### Production Environment

```yaml
# production.env
NODE_ENV=production
API_URL=https://api.yourapp.com
DATABASE_URL=postgresql://user:pass@prod-db:5432/appdb
REDIS_URL=redis://prod-redis:6379
LOG_LEVEL=info
RATE_LIMIT_ENABLED=true
MONITORING_ENABLED=true
```

## Database Migrations

### Staging Migrations

```bash
# Run migrations in staging
kubectl exec -it deployment/backend-staging -n staging -- python -m alembic upgrade head
```

### Production Migrations

```bash
# 1. Backup database
kubectl exec -it deployment/postgres -n production -- pg_dump -U postgres appdb > backup-$(date +%Y%m%d).sql

# 2. Run migrations
kubectl exec -it deployment/backend -n production -- python -m alembic upgrade head

# 3. Verify migration
kubectl exec -it deployment/postgres -n production -- psql -U postgres -d appdb -c "\dt"
```

## Rollback Procedures

### Quick Rollback (Production)

```bash
# 1. Identify previous working version
kubectl rollout history deployment/backend -n production

# 2. Rollback to previous version
kubectl rollout undo deployment/backend -n production

# 3. Verify rollback
kubectl rollout status deployment/backend -n production
```

### Full Rollback (with Database)

```bash
# 1. Stop application
kubectl scale deployment/backend --replicas=0 -n production

# 2. Restore database backup
kubectl exec -it deployment/postgres -n production -- psql -U postgres -d appdb < backup-20250727.sql

# 3. Deploy previous application version
kubectl set image deployment/backend backend=ghcr.io/your-org/agentic-startup-studio:v1.2.2 -n production

# 4. Verify deployment
kubectl get pods -n production
curl https://api.yourapp.com/health
```

## Monitoring and Alerting

### Health Checks

```bash
# Application health
curl https://api.yourapp.com/health

# Database health
kubectl exec -it deployment/postgres -n production -- pg_isready

# Redis health
kubectl exec -it deployment/redis -n production -- redis-cli ping
```

### Log Monitoring

```bash
# Application logs
kubectl logs -f deployment/backend -n production

# Nginx logs
kubectl logs -f deployment/nginx -n production

# Database logs
kubectl logs -f deployment/postgres -n production
```

### Metrics Monitoring

- **Prometheus**: http://prometheus.yourapp.com
- **Grafana**: http://grafana.yourapp.com
- **Application Dashboard**: Monitor API response times, error rates
- **Infrastructure Dashboard**: Monitor CPU, memory, disk usage

## Troubleshooting

### Common Deployment Issues

#### Image Pull Errors

```bash
# Check image registry access
docker pull ghcr.io/your-org/agentic-startup-studio:latest

# Verify registry credentials
kubectl get secret regcred -n production -o yaml
```

#### Database Connection Issues

```bash
# Check database connectivity
kubectl exec -it deployment/backend -n production -- python -c "import psycopg2; psycopg2.connect('postgresql://...')"

# Check database pod status
kubectl get pods -l app=postgres -n production
kubectl describe pod postgres-xxx -n production
```

#### Resource Limits

```bash
# Check resource usage
kubectl top pods -n production
kubectl describe node

# Scale up if needed
kubectl scale deployment/backend --replicas=5 -n production
```

### Emergency Procedures

#### Complete Outage

1. **Assess the situation**:
   ```bash
   kubectl get pods -A
   kubectl get nodes
   ```

2. **Check external dependencies**:
   - Database connectivity
   - External API availability
   - DNS resolution

3. **Scale down to minimal resources**:
   ```bash
   kubectl scale deployment/backend --replicas=1 -n production
   ```

4. **Activate maintenance mode**:
   ```bash
   kubectl apply -f k8s/maintenance-mode.yml -n production
   ```

#### Data Loss Prevention

1. **Immediate database backup**:
   ```bash
   kubectl exec -it deployment/postgres -n production -- pg_dump -U postgres appdb > emergency-backup-$(date +%Y%m%d-%H%M%S).sql
   ```

2. **Stop write operations**:
   ```bash
   kubectl scale deployment/backend --replicas=0 -n production
   ```

3. **Assess data integrity**:
   ```bash
   kubectl exec -it deployment/postgres -n production -- psql -U postgres -d appdb -c "SELECT COUNT(*) FROM critical_table"
   ```

## Post-Deployment Checklist

### Immediate Checks (0-15 minutes)

- [ ] Health endpoints return 200 OK
- [ ] Frontend loads successfully
- [ ] API documentation accessible
- [ ] Database migrations completed
- [ ] No error spikes in logs

### Extended Checks (15-60 minutes)

- [ ] User authentication working
- [ ] Core user flows functional
- [ ] Performance metrics within normal range
- [ ] No memory leaks detected
- [ ] Background jobs processing

### Long-term Monitoring (1-24 hours)

- [ ] Error rates below threshold (< 1%)
- [ ] Response times within SLA (< 200ms p95)
- [ ] Resource usage stable
- [ ] No customer complaints
- [ ] Monitoring alerts silent

## Contacts and Escalation

### On-Call Rotation

- **Primary**: [Engineer Name] - [Contact Info]
- **Secondary**: [Engineer Name] - [Contact Info]
- **Manager**: [Manager Name] - [Contact Info]

### Escalation Matrix

1. **Level 1**: Development team member
2. **Level 2**: Team lead or senior engineer
3. **Level 3**: Engineering manager
4. **Level 4**: CTO or VP Engineering

### Communication Channels

- **Slack**: #incidents channel
- **Email**: engineering@yourcompany.com
- **Phone**: Emergency hotline for critical issues

## Documentation Updates

After each deployment:

1. Update this runbook with any new procedures
2. Document any issues encountered and resolutions
3. Update monitoring thresholds if needed
4. Review and update emergency procedures