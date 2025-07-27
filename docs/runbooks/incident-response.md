# Incident Response Runbook

This runbook provides procedures for responding to incidents in the Agentic Startup Studio application.

## Incident Classification

### Severity Levels

#### Severity 1 (Critical)
- **Impact**: Complete service outage or data loss
- **Response Time**: Immediate (< 15 minutes)
- **Examples**: 
  - Application completely down
  - Database corruption or data loss
  - Security breach
  - Payment processing failure

#### Severity 2 (High)
- **Impact**: Major functionality degraded
- **Response Time**: < 1 hour
- **Examples**:
  - Significant performance degradation
  - Core features unavailable
  - Elevated error rates (> 5%)

#### Severity 3 (Medium)
- **Impact**: Minor functionality issues
- **Response Time**: < 4 hours
- **Examples**:
  - Non-critical features unavailable
  - Performance slightly degraded
  - Moderate error rates (1-5%)

#### Severity 4 (Low)
- **Impact**: Cosmetic or minor issues
- **Response Time**: Next business day
- **Examples**:
  - UI glitches
  - Documentation errors
  - Low error rates (< 1%)

## Incident Response Process

### 1. Detection and Alerting

#### Automated Monitoring
- **Prometheus Alerts**: API response time, error rates, resource usage
- **Database Monitoring**: Connection issues, slow queries
- **External Monitoring**: Uptime checks, SSL certificate expiry

#### Manual Reporting
- User reports via support channels
- Team member discovery
- Customer success team notifications

### 2. Initial Response

#### Immediate Actions (First 5 minutes)
1. **Acknowledge the incident**
2. **Assess severity level**
3. **Start incident tracking**
4. **Assemble response team**

```bash
# Quick health check commands
curl -I https://api.yourapp.com/health
kubectl get pods -A | grep -v Running
kubectl top nodes
```

#### Communication
1. **Internal notification** (Slack #incidents channel)
2. **Customer notification** (if Severity 1-2)
3. **Stakeholder notification** (if business critical)

### 3. Investigation and Diagnosis

#### Information Gathering
```bash
# System status
kubectl get pods -o wide -A
kubectl get nodes -o wide
kubectl get events --sort-by='.lastTimestamp' -A

# Resource usage
kubectl top pods -A
kubectl top nodes

# Recent deployments
kubectl rollout history deployment/backend -n production
git log --oneline -10

# Error logs
kubectl logs -f deployment/backend -n production --tail=100
kubectl logs -f deployment/frontend -n production --tail=100
```

#### Database Investigation
```bash
# Connection status
kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Slow queries
kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Database size and locks
kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database;"
```

#### Performance Analysis
```bash
# API response times
curl -w "@curl-format.txt" -s -o /dev/null https://api.yourapp.com/health

# Memory usage
kubectl exec -it deployment/backend -n production -- ps aux --sort=-%mem | head

# Disk usage
kubectl exec -it deployment/backend -n production -- df -h
```

### 4. Mitigation and Resolution

#### Common Mitigation Strategies

##### High Memory Usage
```bash
# Scale up replicas
kubectl scale deployment/backend --replicas=5 -n production

# Increase memory limits
kubectl patch deployment backend -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"backend","resources":{"limits":{"memory":"1Gi"}}}]}}}}'

# Restart deployment
kubectl rollout restart deployment/backend -n production
```

##### Database Performance Issues
```bash
# Kill long-running queries
kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# Analyze slow queries
kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 5;"

# Restart database (last resort)
kubectl rollout restart deployment/postgres -n production
```

##### High Error Rates
```bash
# Check recent error logs
kubectl logs deployment/backend -n production --since=10m | grep ERROR

# Rollback to previous version
kubectl rollout undo deployment/backend -n production

# Scale down temporarily
kubectl scale deployment/backend --replicas=1 -n production
```

### 5. Recovery Verification

#### Health Checks
```bash
# Application health
curl https://api.yourapp.com/health
curl https://yourapp.com

# API functionality
curl -X POST https://api.yourapp.com/auth/login -d '{"username":"test","password":"test"}'

# Database connectivity
kubectl exec -it deployment/backend -n production -- python -c "import psycopg2; print('DB OK')"
```

#### Performance Verification
```bash
# Response time check
for i in {1..10}; do curl -w "%{time_total}\n" -s -o /dev/null https://api.yourapp.com/health; done

# Error rate check
kubectl logs deployment/backend -n production --since=5m | grep -c ERROR
```

## Specific Incident Scenarios

### Application Outage

#### Symptoms
- Health check endpoints return 5xx errors
- Users cannot access the application
- High error rates in monitoring

#### Investigation Steps
1. **Check pod status**:
   ```bash
   kubectl get pods -n production
   kubectl describe pod <failing-pod> -n production
   ```

2. **Check recent deployments**:
   ```bash
   kubectl rollout history deployment/backend -n production
   ```

3. **Check resource limits**:
   ```bash
   kubectl top pods -n production
   kubectl describe node
   ```

#### Resolution Steps
1. **Quick rollback** (if recent deployment):
   ```bash
   kubectl rollout undo deployment/backend -n production
   ```

2. **Scale up resources** (if resource constrained):
   ```bash
   kubectl scale deployment/backend --replicas=3 -n production
   ```

3. **Restart services** (if configuration issue):
   ```bash
   kubectl rollout restart deployment/backend -n production
   ```

### Database Issues

#### Symptoms
- Database connection timeouts
- Slow query performance
- High database CPU/memory usage

#### Investigation Steps
1. **Check database pod status**:
   ```bash
   kubectl get pods -l app=postgres -n production
   kubectl logs deployment/postgres -n production --tail=50
   ```

2. **Check active connections**:
   ```bash
   kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
   ```

3. **Identify slow queries**:
   ```bash
   kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT query, state, query_start FROM pg_stat_activity WHERE state != 'idle' ORDER BY query_start;"
   ```

#### Resolution Steps
1. **Kill long-running queries**:
   ```bash
   kubectl exec -it deployment/postgres -n production -- psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < now() - interval '10 minutes';"
   ```

2. **Restart database connections**:
   ```bash
   kubectl rollout restart deployment/backend -n production
   ```

3. **Scale database resources** (if resource issue):
   ```bash
   kubectl patch deployment postgres -n production -p '{"spec":{"template":{"spec":{"containers":[{"name":"postgres","resources":{"limits":{"memory":"2Gi","cpu":"1000m"}}}]}}}}'
   ```

### Performance Degradation

#### Symptoms
- Increased response times
- Higher resource usage
- User complaints about slowness

#### Investigation Steps
1. **Check response times**:
   ```bash
   # Use monitoring dashboard or
   curl -w "%{time_total}\n" -s -o /dev/null https://api.yourapp.com/health
   ```

2. **Check resource usage**:
   ```bash
   kubectl top pods -n production
   kubectl top nodes
   ```

3. **Check for memory leaks**:
   ```bash
   kubectl exec -it deployment/backend -n production -- ps aux --sort=-%mem
   ```

#### Resolution Steps
1. **Scale horizontally**:
   ```bash
   kubectl scale deployment/backend --replicas=5 -n production
   ```

2. **Restart to clear memory**:
   ```bash
   kubectl rollout restart deployment/backend -n production
   ```

3. **Enable caching**:
   ```bash
   kubectl patch configmap app-config -n production -p '{"data":{"CACHE_ENABLED":"true"}}'
   ```

## Communication Templates

### Initial Incident Notification

```
🚨 INCIDENT ALERT - SEV [1-4]

Title: [Brief description]
Status: Investigating
Impact: [Description of user impact]
Started: [Timestamp]

Next Update: [Timestamp]
Incident Commander: [Name]
```

### Status Update

```
📊 INCIDENT UPDATE - SEV [1-4]

Title: [Brief description]
Status: [Investigating/Mitigating/Resolved]
Progress: [What has been done]
Next Steps: [What will be done next]

Next Update: [Timestamp]
```

### Resolution Notification

```
✅ INCIDENT RESOLVED - SEV [1-4]

Title: [Brief description]
Status: Resolved
Duration: [Start time - End time]
Root Cause: [Brief explanation]
Impact: [What was affected]

Post-incident review will be scheduled.
```

## Post-Incident Activities

### Immediate Post-Incident (< 2 hours)
1. **Verify complete resolution**
2. **Document timeline and actions taken**
3. **Notify stakeholders of resolution**
4. **Schedule post-incident review**

### Post-Incident Review (< 1 week)
1. **Conduct blameless post-mortem**
2. **Identify root causes**
3. **Create action items for prevention**
4. **Update runbooks and procedures**
5. **Share learnings with team**

### Follow-up Actions
1. **Implement preventive measures**
2. **Update monitoring and alerting**
3. **Improve documentation**
4. **Training if knowledge gaps identified**

## Tools and Resources

### Monitoring URLs
- **Grafana**: https://grafana.yourapp.com
- **Prometheus**: https://prometheus.yourapp.com
- **Application Logs**: https://logs.yourapp.com
- **Status Page**: https://status.yourapp.com

### Emergency Contacts
- **On-call Engineer**: [Phone/Slack]
- **Engineering Manager**: [Phone/Slack]
- **DevOps Lead**: [Phone/Slack]
- **CTO**: [Phone/Slack] (Sev 1 only)

### Documentation Links
- [Architecture Documentation](../ARCHITECTURE.md)
- [Deployment Runbook](./deployment.md)
- [Troubleshooting Guide](../troubleshooting/common-issues.md)
- [API Documentation](https://api.yourapp.com/docs)

### Useful Commands Reference

```bash
# Quick health check
kubectl get pods -A | grep -v Running

# Scale deployment
kubectl scale deployment/[name] --replicas=[count] -n [namespace]

# Rollback deployment
kubectl rollout undo deployment/[name] -n [namespace]

# View logs
kubectl logs -f deployment/[name] -n [namespace] --tail=100

# Resource usage
kubectl top pods -n [namespace]

# Database access
kubectl exec -it deployment/postgres -n production -- psql -U postgres
```