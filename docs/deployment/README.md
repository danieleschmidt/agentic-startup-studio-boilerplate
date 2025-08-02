# Deployment Guide

This guide covers deployment strategies and procedures for the Agentic Startup Studio Boilerplate.

## Deployment Overview

The boilerplate supports multiple deployment strategies:
- **Local Development**: Docker Compose for rapid development
- **Staging Environment**: Kubernetes or Docker Swarm for testing
- **Production Environment**: Kubernetes with high availability and auto-scaling
- **Cloud Deployment**: AWS, GCP, Azure with managed services

## Build System

### Multi-Stage Docker Build

Our Dockerfile uses multi-stage builds for optimal image size and security:

```dockerfile
# Build stage for Python dependencies
FROM python:3.11-slim as python-builder
# ... Python build steps

# Build stage for Node.js dependencies  
FROM node:18-alpine as node-builder
# ... Node.js build steps

# Final runtime stage
FROM python:3.11-slim
# ... Runtime configuration
```

### Build Commands

```bash
# Development build
make build-dev
docker build --target development -t agentic-studio:dev .

# Production build
make build
docker build -t agentic-studio:latest .

# Multi-architecture build
docker buildx build --platform linux/amd64,linux/arm64 -t agentic-studio:latest .
```

## Environment Configuration

### Environment Variables

Create environment-specific configuration files:

```bash
# Development
cp .env.example .env.development

# Staging
cp .env.example .env.staging

# Production
cp .env.example .env.production
```

Key production environment variables:
```bash
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
DATABASE_URL=postgresql://user:pass@prod-db:5432/agentic_studio
REDIS_URL=redis://prod-redis:6379/0
KEYCLOAK_URL=https://auth.yourdomain.com
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
```

### Secrets Management

**Development:**
```bash
# Use .env files (not committed)
echo "SECRET_KEY=dev-secret" >> .env
```

**Production:**
```bash
# Kubernetes secrets
kubectl create secret generic app-secrets \
  --from-literal=secret-key=$SECRET_KEY \
  --from-literal=db-password=$DB_PASSWORD

# Docker secrets
echo $SECRET_KEY | docker secret create app_secret_key -
```

## Local Development Deployment

### Quick Start
```bash
# Start all services
make dev-up
# or
docker-compose -f docker-compose.dev.yml up -d

# View logs
make dev-logs
# or
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
make dev-down
# or
docker-compose -f docker-compose.dev.yml down
```

### Service URLs
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Keycloak: http://localhost:8080
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

## Staging Deployment

### Docker Compose (Simple Staging)

```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Update staging deployment
docker-compose -f docker-compose.staging.yml pull
docker-compose -f docker-compose.staging.yml up -d --force-recreate
```

### Docker Swarm (Orchestrated Staging)

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml agentic-studio

# Update service
docker service update --image agentic-studio:v1.2.0 agentic-studio_api

# Remove stack
docker stack rm agentic-studio
```

## Production Deployment

### Kubernetes Deployment

#### Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
```

#### Namespace Setup
```bash
# Create namespace
kubectl create namespace agentic-studio

# Set default namespace
kubectl config set-context --current --namespace=agentic-studio
```

#### Secrets and ConfigMaps
```bash
# Create secrets
kubectl create secret generic app-secrets \
  --from-literal=secret-key=$SECRET_KEY \
  --from-literal=jwt-secret=$JWT_SECRET \
  --from-literal=db-password=$DB_PASSWORD

# Create config map
kubectl create configmap app-config \
  --from-env-file=.env.production
```

#### Database Deployment
```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: agentic_studio
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: db-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
```

#### Application Deployment
```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-studio
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentic-studio
  template:
    metadata:
      labels:
        app: agentic-studio
    spec:
      containers:
      - name: app
        image: agentic-studio:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: postgresql://postgres:$(DB_PASSWORD)@postgres:5432/agentic_studio
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: secret-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

#### Service and Ingress
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agentic-studio-service
spec:
  selector:
    app: agentic-studio
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agentic-studio-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: agentic-studio-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agentic-studio-service
            port:
              number: 80
```

#### Deploy to Kubernetes
```bash
# Apply all configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/agentic-studio

# Scale deployment
kubectl scale deployment agentic-studio --replicas=5
```

### Helm Deployment

#### Create Helm Chart
```bash
# Generate chart
helm create agentic-studio-chart

# Install chart
helm install agentic-studio ./agentic-studio-chart \
  --set image.tag=v1.0.0 \
  --set environment=production

# Upgrade deployment
helm upgrade agentic-studio ./agentic-studio-chart \
  --set image.tag=v1.1.0

# Rollback deployment
helm rollback agentic-studio 1
```

## Cloud Provider Deployments

### AWS Deployment

#### EKS (Elastic Kubernetes Service)
```bash
# Create EKS cluster
eksctl create cluster --name agentic-studio --region us-west-2

# Deploy to EKS
kubectl apply -f k8s/aws/

# Set up Application Load Balancer
kubectl apply -f https://raw.githubusercontent.com/aws/aws-load-balancer-controller/main/docs/install/iam_policy.json
```

#### ECS (Elastic Container Service)
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name agentic-studio

# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service --cluster agentic-studio --service-name api --task-definition agentic-studio:1
```

### GCP Deployment

#### GKE (Google Kubernetes Engine)
```bash
# Create GKE cluster
gcloud container clusters create agentic-studio \
  --zone us-central1-a \
  --num-nodes 3

# Get credentials
gcloud container clusters get-credentials agentic-studio --zone us-central1-a

# Deploy application
kubectl apply -f k8s/gcp/
```

#### Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy agentic-studio \
  --image gcr.io/PROJECT_ID/agentic-studio:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Deployment

#### AKS (Azure Kubernetes Service)
```bash
# Create resource group
az group create --name agentic-studio-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group agentic-studio-rg \
  --name agentic-studio-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group agentic-studio-rg --name agentic-studio-aks

# Deploy application
kubectl apply -f k8s/azure/
```

## Monitoring and Observability

### Health Checks

The application provides comprehensive health checks:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Database health
curl http://localhost:8000/health/db

# External services health
curl http://localhost:8000/health/external
```

### Metrics Collection

Prometheus metrics are available at `/metrics`:

```bash
# Scrape metrics
curl http://localhost:8000/metrics
```

### Log Aggregation

Configure centralized logging:

```yaml
# fluentd-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
    </source>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name kubernetes
    </match>
```

## Backup and Recovery

### Database Backups

#### Automated Backups
```bash
# PostgreSQL backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > backup_$DATE.sql
aws s3 cp backup_$DATE.sql s3://backups/db/
```

#### Kubernetes CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15-alpine
            command:
            - /bin/sh
            - -c
            - pg_dump $DATABASE_URL > /backup/backup_$(date +%Y%m%d_%H%M%S).sql
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: app-secrets
                  key: database-url
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

### Disaster Recovery

#### Backup Strategy
1. **Database**: Daily automated backups with 30-day retention
2. **Application State**: Persistent volumes with snapshots
3. **Configuration**: Git-based infrastructure as code
4. **Secrets**: Encrypted backup in secure storage

#### Recovery Procedures
```bash
# Database recovery
pg_restore -d $DATABASE_URL backup_20250101_020000.sql

# Application recovery
kubectl apply -f k8s/
helm install agentic-studio ./chart --values production.yaml
```

## Performance Optimization

### Resource Tuning

#### CPU and Memory
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

#### Database Connection Pooling
```python
# In application configuration
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
```

### Auto-scaling

#### Horizontal Pod Autoscaler
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentic-studio-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentic-studio
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security Considerations

### Network Security
- Use TLS/SSL for all communications
- Implement network policies in Kubernetes
- Use service mesh for advanced traffic management

### Container Security
- Use non-root users in containers
- Scan images for vulnerabilities
- Use minimal base images (alpine, distroless)
- Implement Pod Security Standards

### Access Control
- Use RBAC for Kubernetes access
- Implement API authentication and authorization
- Regular security audits and penetration testing

## Troubleshooting

### Common Issues

#### Pod Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it <pod-name> -- pg_isready -h postgres -p 5432

# Check database logs
kubectl logs postgres-deployment-xxx
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods
kubectl top nodes

# Check metrics
curl http://localhost:8000/metrics | grep http_request_duration
```

### Debugging Tools

```bash
# Enter container for debugging
kubectl exec -it <pod-name> -- /bin/bash

# Port forward for local access
kubectl port-forward pod/<pod-name> 8000:8000

# Check cluster info
kubectl cluster-info dump
```

## Continuous Deployment

### GitOps with ArgoCD

```yaml
# application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: agentic-studio
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/yourusername/agentic-startup-studio-boilerplate
    targetRevision: HEAD
    path: k8s
  destination:
    server: https://kubernetes.default.svc
    namespace: agentic-studio
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### CI/CD Pipeline Integration

See [GitHub Actions Guide](../workflows/README.md) for complete CI/CD setup including:
- Automated testing
- Security scanning
- Image building and pushing
- Deployment automation
- Rollback procedures

## Best Practices

1. **Infrastructure as Code**: Version control all configuration
2. **Immutable Infrastructure**: Replace rather than update
3. **Blue-Green Deployments**: Zero-downtime deployments
4. **Monitoring First**: Implement observability before deployment
5. **Security by Default**: Secure configurations from the start
6. **Automation**: Automate all repetitive tasks
7. **Documentation**: Keep deployment docs up to date
8. **Testing**: Test deployments in staging first
9. **Rollback Plan**: Always have a rollback strategy
10. **Incident Response**: Prepare for deployment issues