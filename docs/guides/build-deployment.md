# Build & Deployment Guide

This guide covers building, containerizing, and deploying the Agentic Startup Studio Boilerplate.

## Overview

The project uses a multi-stage Docker build process with:
- **Development**: Hot-reload environment with debugging tools
- **Testing**: Automated testing with coverage reporting
- **Production**: Optimized, security-hardened runtime
- **Security Scanning**: Vulnerability analysis and reporting

## Build System

### Docker Multi-Stage Build

The `Dockerfile` includes multiple build stages:

```dockerfile
# Python dependencies stage
FROM python:3.11-slim as python-builder

# Node.js dependencies stage  
FROM node:18-alpine as node-builder

# Production runtime stage
FROM python:3.11-slim as production

# Development stage
FROM production as development

# Testing stage
FROM development as testing

# Security scanning stage
FROM python:3.11-slim as security-scanner
```

### Build Commands

```bash
# Production build
make build
docker build -t agentic-startup-studio:latest .

# Development build
make build-dev
docker build -t agentic-startup-studio:dev --target development .

# Testing build
docker build -t agentic-startup-studio:test --target testing .

# Security scanning
docker build -t agentic-startup-studio:security --target security-scanner .
```

## Development Environment

### Quick Start

```bash
# Start development environment
make dev-up
# or
npm run dev:up

# Stop development environment
make dev-down
# or
npm run dev:down
```

### Services Available

- **Frontend**: http://localhost:3000 (React)
- **API**: http://localhost:8000 (FastAPI)
- **API Docs**: http://localhost:8000/docs (Swagger)
- **Database Admin**: http://localhost:8081 (Adminer)
- **Monitoring**: http://localhost:9090 (Prometheus)
- **Dashboards**: http://localhost:3001 (Grafana)

### Development Tools

```bash
# View logs
make dev-logs

# Restart services
make dev-restart

# Check status
make status

# Database shell
make db-shell

# Redis shell
make redis-shell
```

## Production Deployment

### Production Build

```bash
# Build production image
make build

# Deploy to production
make deploy-prod
```

### Environment Configuration

Create production `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit production values
nano .env
```

Key production settings:

```env
ENVIRONMENT=production
DEBUG=false
API_SECRET_KEY=your-strong-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
```

### Production Services

The production stack includes:

- **App**: Main application server
- **Database**: PostgreSQL with health checks
- **Cache**: Redis for sessions and caching
- **Proxy**: Nginx reverse proxy with SSL
- **Workers**: Celery background workers
- **Scheduler**: Celery beat task scheduler
- **Monitoring**: Prometheus + Grafana

### SSL/TLS Configuration

Place SSL certificates in `./ssl/`:

```
ssl/
├── cert.pem
├── key.pem
└── dhparam.pem
```

Update `nginx/nginx.conf` for SSL configuration.

## Container Security

### Security Features

- **Non-root user**: All processes run as `appuser` (UID 1000)
- **Read-only filesystem**: Container filesystem is read-only
- **No new privileges**: Prevents privilege escalation
- **Minimal attack surface**: Only necessary packages installed
- **Security updates**: Base images regularly updated

### Security Scanning

```bash
# Run security scans
make security-full

# Individual tools
bandit -r .              # Python security analysis
safety check             # Dependency vulnerability scan
semgrep --config=auto .  # Static analysis
```

### Docker Security Best Practices

1. **Use specific tags**: Never use `latest` in production
2. **Scan images**: Regular vulnerability scanning
3. **Minimal base**: Use Alpine or distroless images
4. **Secrets management**: Use Docker secrets or external vault
5. **Network isolation**: Use custom networks
6. **Resource limits**: Set memory and CPU limits

## Database Management

### Migrations

```bash
# Run migrations
make migrations

# Create new migration
make migration-create

# Rollback migration
alembic downgrade -1
```

### Backup and Restore

```bash
# Create backup
make backup-db

# Restore from backup
docker-compose exec db psql -U postgres -d agentic_startup < backup.sql
```

### Seeding Data

```bash
# Seed development data
make seed-data
```

## Monitoring and Observability

### Metrics Collection

Prometheus scrapes metrics from:
- Application metrics (FastAPI)
- Container metrics (cAdvisor)
- System metrics (Node Exporter)

### Grafana Dashboards

Pre-configured dashboards for:
- Application performance
- Infrastructure metrics
- Business metrics
- Error tracking

### Health Checks

All services include health checks:

```bash
# Check service health
make health-check

# Individual service checks
curl http://localhost:8000/health  # API health
curl http://localhost:3000         # Frontend health
```

## CI/CD Integration

### GitHub Actions

Example workflow for automated deployment:

```yaml
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t ${{ secrets.REGISTRY }}/app:${{ github.sha }} .
        
    - name: Run tests
      run: |
        docker run --rm ${{ secrets.REGISTRY }}/app:${{ github.sha }} pytest
        
    - name: Security scan
      run: |
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy image ${{ secrets.REGISTRY }}/app:${{ github.sha }}
        
    - name: Push to registry
      run: |
        docker push ${{ secrets.REGISTRY }}/app:${{ github.sha }}
        
    - name: Deploy to production
      run: |
        # Deploy using your preferred method (k8s, docker-compose, etc.)
```

### Semantic Versioning

Automated versioning using semantic-release:

```bash
# Dry run release
make release-dry

# Create release
make release
```

Commit message format:
- `feat:` → Minor version bump
- `fix:` → Patch version bump
- `feat!:` or `BREAKING CHANGE:` → Major version bump

## Multi-Environment Deployment

### Staging Environment

```bash
# Deploy to staging
make deploy-staging

# Use staging compose file
docker-compose -f docker-compose.staging.yml up -d
```

### Environment-Specific Configuration

Use different `.env` files:
- `.env.development`
- `.env.staging`
- `.env.production`

### Blue-Green Deployment

For zero-downtime deployments:

1. Build new version
2. Deploy to "green" environment
3. Run smoke tests
4. Switch traffic to "green"
5. Keep "blue" as fallback

## Performance Optimization

### Build Optimization

- **Multi-stage builds**: Separate build and runtime dependencies
- **Layer caching**: Optimize Dockerfile layer order
- **Dependency caching**: Cache npm/pip installations
- **Minimal images**: Use Alpine or distroless base images

### Runtime Optimization

- **Resource limits**: Set appropriate CPU/memory limits
- **Connection pooling**: Database connection optimization
- **Caching strategies**: Redis for API and session caching
- **Static file serving**: Nginx for static assets

### Monitoring Build Performance

```bash
# Analyze build time
docker build --progress=plain -t app .

# Check image size
docker images app

# Dive into layers
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  wagoodman/dive:latest app
```

## Troubleshooting

### Common Build Issues

1. **Out of space**: Clean Docker cache
   ```bash
   docker system prune -a
   ```

2. **Permission denied**: Check file permissions
   ```bash
   chmod +x docker-entrypoint.sh
   ```

3. **Port conflicts**: Check for running services
   ```bash
   netstat -tulpn | grep :8000
   ```

### Container Debugging

```bash
# Shell into running container
docker exec -it container-name bash

# Check container logs
docker logs container-name

# Inspect container
docker inspect container-name

# Check resource usage
docker stats
```

### Performance Issues

1. **Slow builds**: Use build cache and multi-stage builds
2. **High memory usage**: Set memory limits and optimize code
3. **Network issues**: Check Docker network configuration
4. **Database connections**: Monitor connection pool usage

## Best Practices

### Docker Best Practices

1. **Use .dockerignore**: Exclude unnecessary files
2. **Multi-stage builds**: Separate build and runtime
3. **Non-root user**: Run as non-privileged user
4. **Specific versions**: Pin base image versions
5. **Health checks**: Always include health checks
6. **Secrets**: Never include secrets in images

### Deployment Best Practices

1. **Infrastructure as Code**: Use Terraform/Ansible
2. **Blue-green deployment**: Zero-downtime deployments
3. **Monitoring**: Comprehensive observability
4. **Backups**: Automated backup strategies
5. **Security**: Regular security scans and updates
6. **Testing**: Automated testing in CI/CD

### Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database migrations applied
- [ ] Monitoring configured
- [ ] Backups scheduled
- [ ] Security scans passing
- [ ] Load testing completed
- [ ] Rollback plan prepared

## Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/develop/dev-best-practices/#use-multi-stage-builds)
- [Container Security](https://docs.docker.com/engine/security/)
- [Docker Compose Production](https://docs.docker.com/compose/production/)
- [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)