# Quantum Task Planner - Deployment Guide

## 🚀 Deployment Overview

This guide provides comprehensive instructions for deploying the Quantum Task Planner system across development, staging, and production environments using the built-in Quantum Production Orchestrator.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (NGINX)                    │
├─────────────────────────────────────────────────────────────────┤
│                     API Instances (3+)                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Quantum API   │  │   Quantum API   │  │   Quantum API   │ │
│  │   + Security    │  │   + Security    │  │   + Security    │ │
│  │   + Monitoring  │  │   + Monitoring  │  │   + Monitoring  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   PostgreSQL    │  │      Redis      │  │    RabbitMQ     │ │
│  │   (Primary DB)  │  │    (Cache)      │  │  (Message Queue) │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Monitoring Stack                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Prometheus    │  │     Grafana     │  │   ELK Stack     │ │
│  │   (Metrics)     │  │  (Dashboards)   │  │    (Logs)       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 recommended)
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 100GB SSD (500GB recommended)
- **Network**: 1Gbps connection

### Recommended Production Setup
- **CPU**: 8+ cores per node
- **Memory**: 32GB+ RAM per node
- **Storage**: 1TB+ NVMe SSD
- **Network**: 10Gbps connection
- **Nodes**: 3+ for high availability

## Deployment Options

### 1. Docker Compose (Single Node)

Best for: Development, staging, small production deployments

```bash
# Quick deployment
./scripts/deploy.sh docker-compose production

# Custom configuration
./scripts/deploy.sh docker-compose production v2.0.0
```

**Features:**
- Single-node deployment
- Full monitoring stack
- SSL termination
- Automated backups
- Health checks

### 2. Kubernetes (Multi-Node)

Best for: Production, high availability, auto-scaling

```bash
# Deploy to Kubernetes
./scripts/deploy.sh kubernetes production

# Check deployment status
kubectl get pods -n quantum-tasks
```

**Features:**
- Multi-node deployment
- Auto-scaling (HPA)
- Rolling updates
- Service discovery
- Resource management
- Network policies

### 3. Cloud Deployments

#### AWS EKS
```bash
# Create EKS cluster
eksctl create cluster --name quantum-tasks --nodes 3

# Deploy application
./scripts/deploy.sh kubernetes production
```

#### Google GKE
```bash
# Create GKE cluster
gcloud container clusters create quantum-tasks --num-nodes=3

# Deploy application
./scripts/deploy.sh kubernetes production
```

#### Azure AKS
```bash
# Create AKS cluster
az aks create --name quantum-tasks --node-count 3

# Deploy application
./scripts/deploy.sh kubernetes production
```

## Configuration

### Environment Variables

Create `.env.production` file:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://quantum_user:password@postgres:5432/quantum_tasks

# Security
JWT_SECRET_KEY=your_jwt_secret_key_base64
ENCRYPTION_KEY=your_encryption_key_base64

# Cache
REDIS_PASSWORD=your_redis_password
REDIS_URL=redis://:password@redis:6379

# Monitoring
GRAFANA_PASSWORD=your_grafana_password

# SSL/TLS
DOMAIN_NAME=quantum-tasks.yourdomain.com
SSL_EMAIL=admin@yourdomain.com

# Backup
BACKUP_S3_BUCKET=your-backup-bucket
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

### Security Configuration

1. **SSL/TLS Certificates**
   ```bash
   # Generate certificates with Let's Encrypt
   certbot certonly --webroot -w /var/www/html -d quantum-tasks.yourdomain.com
   ```

2. **Database Security**
   - Use strong passwords (32+ characters)
   - Enable SSL connections
   - Restrict network access
   - Regular security updates

3. **API Security**
   - JWT tokens with rotation
   - Rate limiting enabled
   - Input validation
   - CORS properly configured

### Performance Tuning

#### Database Optimization
```sql
-- PostgreSQL configuration
shared_buffers = '4GB'
effective_cache_size = '12GB'
maintenance_work_mem = '1GB'
checkpoint_completion_target = 0.9
wal_buffers = '16MB'
default_statistics_target = 100
```

#### Redis Optimization
```
# Redis configuration
maxmemory 2gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 300
```

#### API Optimization
```yaml
# Kubernetes resource limits
resources:
  requests:
    cpu: 250m
    memory: 512Mi
  limits:
    cpu: 1000m
    memory: 2Gi
```

## Monitoring and Observability

### Metrics Dashboard

Access Grafana at `http://your-domain:3000`

**Key Metrics:**
- API response times (p95, p99)
- Task throughput
- Quantum coherence levels
- Error rates
- Resource utilization

### Alerting

**Critical Alerts:**
- API down (> 1 minute)
- High error rate (> 5%)
- Database connection failure
- Low disk space (< 10%)
- Quantum coherence critical (< 0.1)

**Warning Alerts:**
- High latency (> 2 seconds p95)
- Memory usage (> 80%)
- CPU usage (> 80%)
- Low quantum coherence (< 0.3)

### Logging

**Log Levels:**
- **ERROR**: System failures, exceptions
- **WARN**: Performance degradation, recoverable errors
- **INFO**: Normal operations, business events
- **DEBUG**: Detailed troubleshooting information

**Log Aggregation:**
```bash
# View logs
docker-compose logs -f quantum-api
kubectl logs -f deployment/quantum-api -n quantum-tasks

# Search logs
curl "http://elasticsearch:9200/logs-*/_search?q=ERROR"
```

## Backup and Recovery

### Automated Backups

**Database Backups:**
- Daily full backups
- Hourly incremental backups
- 30-day retention policy
- S3 storage with encryption

**Configuration Backups:**
- Kubernetes manifests
- Environment configurations
- SSL certificates
- Monitoring configurations

### Recovery Procedures

#### Database Recovery
```bash
# Restore from backup
gunzip -c backup/database_20240106_020000.sql.gz | \
  docker exec -i quantum-postgres psql -U quantum_user -d quantum_tasks

# Point-in-time recovery
pg_basebackup -h postgres -U quantum_user -D /backup/base
```

#### Application Recovery
```bash
# Rollback deployment
./scripts/deploy.sh rollback

# Restore from previous version
docker-compose up -d --scale quantum-api=3
```

## High Availability

### Multi-Region Setup

```yaml
# Primary region (us-west-2)
regions:
  primary:
    name: us-west-2
    nodes: 3
    database: primary
  
# Secondary region (us-east-1)
  secondary:
    name: us-east-1
    nodes: 2
    database: read-replica
```

### Disaster Recovery

**RTO (Recovery Time Objective):** < 15 minutes
**RPO (Recovery Point Objective):** < 5 minutes

**Procedures:**
1. Automated failover to secondary region
2. DNS update for traffic routing
3. Database promotion (read-replica → primary)
4. Application scaling in new region

## Scaling

### Horizontal Scaling

**Kubernetes HPA:**
```yaml
spec:
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

**Manual Scaling:**
```bash
# Scale API pods
kubectl scale deployment quantum-api --replicas=10 -n quantum-tasks

# Scale with Docker Compose
docker-compose up -d --scale quantum-api=5
```

### Vertical Scaling

**Resource Limits:**
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

## Security Hardening

### Network Security
- VPC with private subnets
- Network ACLs and security groups
- WAF protection
- DDoS mitigation

### Application Security
- Regular security scans
- Dependency updates
- Secret management
- Audit logging

### Compliance
- SOC 2 compliance ready
- GDPR data protection
- HIPAA compatible configurations
- PCI DSS for payment data

## Troubleshooting

### Common Issues

#### API Not Responding
```bash
# Check health
curl http://localhost/api/v1/health

# Check logs
docker-compose logs quantum-api

# Restart service
docker-compose restart quantum-api
```

#### Database Connection Issues
```bash
# Test connection
docker exec quantum-postgres pg_isready -U quantum_user

# Check configuration
kubectl describe configmap quantum-config -n quantum-tasks

# Reset connection pool
kubectl rollout restart deployment/quantum-api -n quantum-tasks
```

#### High Memory Usage
```bash
# Check resource usage
kubectl top pods -n quantum-tasks

# Analyze memory usage
docker exec quantum-api ps aux --sort=-%mem

# Restart with more memory
kubectl patch deployment quantum-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"quantum-api","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

### Performance Issues

#### Slow Response Times
1. Check database query performance
2. Verify cache hit rates
3. Analyze API endpoint metrics
4. Review resource utilization

#### Quantum Coherence Degradation
1. Check task complexity distribution
2. Verify entanglement bond efficiency
3. Analyze decoherence patterns
4. Review optimization algorithms

## Maintenance

### Regular Tasks

**Daily:**
- Monitor system health
- Check backup success
- Review alert status
- Verify SSL certificate validity

**Weekly:**
- Update security patches
- Review performance metrics
- Clean up old logs
- Test disaster recovery procedures

**Monthly:**
- Security audit
- Performance optimization
- Capacity planning review
- Update documentation

### Upgrade Procedures

```bash
# Rolling update (Kubernetes)
kubectl set image deployment/quantum-api quantum-api=quantum-task-planner:v2.1.0 -n quantum-tasks

# Zero-downtime update (Docker Compose)
docker-compose up -d --no-deps quantum-api

# Rollback if needed
kubectl rollout undo deployment/quantum-api -n quantum-tasks
```

## Support and Resources

### Documentation
- API Documentation: `/docs`
- Architecture Guide: `ARCHITECTURE.md`
- Development Guide: `DEVELOPMENT.md`

### Monitoring Dashboards
- Grafana: `http://your-domain:3000`
- Prometheus: `http://your-domain:9091`
- Kibana: `http://your-domain:5601`

### Emergency Contacts
- On-call engineer: +1-XXX-XXX-XXXX
- DevOps team: devops@yourcompany.com
- Security team: security@yourcompany.com

---

## Quick Start Commands

```bash
# Deploy with Docker Compose
git clone <repository>
cd quantum-task-planner
cp .env.example .env.production
./scripts/deploy.sh docker-compose production

# Deploy with Kubernetes
kubectl create namespace quantum-tasks
kubectl apply -f k8s/
./scripts/deploy.sh kubernetes production

# Check deployment
curl http://localhost/api/v1/health

# Access monitoring
open http://localhost:3000  # Grafana
```

For additional support, please refer to our documentation or contact the development team.