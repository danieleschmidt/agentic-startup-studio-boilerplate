# Runbook: Service Down

## Alert Description
**Alert**: ServiceDown  
**Severity**: Critical  
**Category**: Availability  

A service is not responding to health checks and appears to be down.

## Immediate Actions (First 5 minutes)

### 1. Acknowledge the Alert
- Log into Grafana/AlertManager
- Acknowledge the alert to prevent spam
- Note the time and affected service

### 2. Verify the Issue
```bash
# Check service status
docker-compose ps
kubectl get pods  # if using k8s

# Check specific service health
curl -f http://service:port/health

# Check service logs
docker logs container-name
kubectl logs pod-name  # if using k8s
```

### 3. Quick Status Check
```bash
# System resources
docker stats
top
df -h

# Network connectivity
ping service-host
telnet service-host port
```

## Investigation Steps

### Check Service Logs
```bash
# Recent logs
docker logs --tail 100 service-name

# Logs from specific time
docker logs --since="2023-07-28T12:00:00" service-name

# Follow live logs
docker logs -f service-name
```

### Check System Resources
```bash
# Memory usage
free -h
docker stats --no-stream

# Disk space
df -h
du -sh /var/lib/docker

# CPU usage
top
htop
```

### Check Dependencies
```bash
# Database connectivity
docker exec app-container psql -h db -U user -c "SELECT 1"

# Redis connectivity
docker exec app-container redis-cli -h redis ping

# External API connectivity
curl -I https://api.openai.com/v1/models
```

## Common Causes and Solutions

### Application Crashes
**Symptoms**: Container stops, exit code != 0
**Solution**:
```bash
# Restart the service
docker-compose restart service-name

# Check for recent code deployments
git log --oneline -10

# Check application configuration
docker exec service-name env | grep -E "(DATABASE|REDIS|API)"
```

### Resource Exhaustion
**Symptoms**: OOMKilled, high CPU/memory usage
**Solution**:
```bash
# Increase resource limits
# Edit docker-compose.yml or k8s resources

# Scale horizontally
docker-compose up --scale app=3

# Clear caches
docker exec redis-container redis-cli FLUSHALL
```

### Database Connection Issues
**Symptoms**: Connection timeout, connection refused
**Solution**:
```bash
# Check database status
docker exec db-container pg_isready

# Check connection pool
# Look for "too many connections" in logs

# Restart database if needed
docker-compose restart db
```

### Network Issues
**Symptoms**: Connection timeouts, DNS resolution failures
**Solution**:
```bash
# Check DNS resolution
nslookup service-name
dig service-name

# Check network connectivity
docker network ls
docker network inspect network-name

# Restart networking
docker-compose down && docker-compose up -d
```

## Recovery Procedures

### Standard Recovery
```bash
# 1. Restart the affected service
docker-compose restart service-name

# 2. Verify service is healthy
curl -f http://service:port/health

# 3. Check dependent services
docker-compose ps

# 4. Monitor for 10 minutes
watch -n 30 'curl -s http://service:port/health'
```

### Full Stack Recovery
```bash
# 1. Stop all services
docker-compose down

# 2. Clean up resources (if needed)
docker system prune -f

# 3. Start services in order
docker-compose up -d db redis
sleep 30
docker-compose up -d app worker

# 4. Verify all services
docker-compose ps
```

### Rollback Procedure
```bash
# 1. Identify last known good version
git log --oneline -10

# 2. Rollback to previous version
git checkout <previous-commit>
docker-compose build
docker-compose up -d

# 3. Verify rollback success
curl -f http://service:port/health
```

## Escalation

### When to Escalate
- Service cannot be restored within 15 minutes
- Multiple services are down simultaneously  
- Data corruption is suspected
- Security breach is suspected

### Escalation Contacts
- **Engineering Lead**: @engineering-lead
- **DevOps Lead**: @devops-lead  
- **On-call Engineer**: +1-555-ONCALL
- **Security Team**: security@company.com (if security related)

### Escalation Information to Provide
- Alert details and time
- Steps taken so far
- Current service status
- Impact assessment
- Recent changes or deployments

## Post-Incident

### Immediate Post-Recovery
1. Update stakeholders on resolution
2. Document the incident in incident tracking system
3. Clear/resolve the alert
4. Monitor service stability for 24 hours

### Post-Incident Review
1. Schedule post-mortem meeting within 24 hours
2. Create timeline of events
3. Identify root cause
4. Document lessons learned
5. Create action items to prevent recurrence

### Follow-up Actions
- [ ] Update monitoring/alerting if needed
- [ ] Update runbooks based on lessons learned
- [ ] Implement preventive measures
- [ ] Schedule infrastructure improvements
- [ ] Update incident response procedures

## Prevention

### Monitoring Improvements
- Add more granular health checks
- Implement synthetic monitoring
- Set up proactive alerting
- Monitor key business metrics

### Infrastructure Improvements
- Implement auto-scaling
- Add redundancy/failover
- Improve resource allocation
- Enhance backup procedures

### Process Improvements
- Automate common recovery tasks
- Improve deployment procedures
- Enhance testing procedures
- Regular disaster recovery drills

## Useful Commands

### Docker Troubleshooting
```bash
# Container status and resource usage
docker ps -a
docker stats

# Network troubleshooting
docker network ls
docker network inspect bridge

# Volume troubleshooting
docker volume ls
docker system df
```

### Kubernetes Troubleshooting
```bash
# Pod status
kubectl get pods -o wide
kubectl describe pod pod-name

# Service status
kubectl get services
kubectl describe service service-name

# Events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### System Troubleshooting
```bash
# Process monitoring
ps aux | grep service-name
lsof -i:port-number

# System resources
vmstat 1 5
iostat 1 5
sar -u 1 5
```

## Related Runbooks
- [High Error Rate](high-error-rate.md)
- [Database Issues](database-issues.md)
- [Performance Issues](performance-issues.md)
- [Security Incidents](security-incidents.md)