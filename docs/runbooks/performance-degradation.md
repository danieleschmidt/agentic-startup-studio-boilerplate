# Performance Degradation Runbook

## Overview
This runbook provides step-by-step procedures for investigating and resolving performance degradation issues in the Agentic Startup Studio Boilerplate.

## Alert Triggers
- API response time > 500ms (95th percentile)
- Error rate > 1%
- Database query time > 1s
- Memory usage > 85%
- CPU usage > 80%

## Initial Assessment (5 minutes)

### 1. Check System Health
```bash
# Check overall system status
kubectl get pods -A
docker ps  # for local development

# Check application health endpoints
curl http://localhost:8000/health/detailed

# Quick metrics check
curl http://localhost:9090/api/v1/query?query=up
```

### 2. Review Recent Changes
- Check recent deployments in the last 24 hours
- Review recent code changes
- Verify configuration changes
- Check for infrastructure modifications

### 3. Identify Scope
- Is the issue affecting all users or specific segments?
- Are all endpoints affected or specific ones?
- Is the issue intermittent or persistent?

## Investigation Steps

### Step 1: Application Layer Analysis

#### Check API Performance
```bash
# Query Prometheus for response times
curl -G http://localhost:9090/api/v1/query_range \
  --data-urlencode 'query=histogram_quantile(0.95, http_request_duration_seconds_bucket)' \
  --data-urlencode 'start=2025-01-01T00:00:00Z' \
  --data-urlencode 'end=2025-01-01T01:00:00Z' \
  --data-urlencode 'step=60s'

# Check error rates
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(http_requests_total{status=~"5.."}[5m])'

# Review slow endpoints
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=topk(10, avg by (endpoint) (http_request_duration_seconds))'
```

#### Check Application Logs
```bash
# Recent error logs
kubectl logs deployment/agentic-studio --since=1h | grep ERROR

# Search for specific patterns
kubectl logs deployment/agentic-studio --since=1h | grep -E "(timeout|connection|memory)"

# Check for slow queries
kubectl logs deployment/agentic-studio --since=1h | grep "slow query"
```

#### CrewAI Performance Check
```bash
# Check agent task performance
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(crewai_tasks_total{status="error"}[5m])'

# Agent execution times
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, crewai_task_duration_seconds_bucket)'

# AI API usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(ai_api_calls_total[5m])'
```

### Step 2: Database Layer Analysis

#### PostgreSQL Performance
```sql
-- Connect to database
psql $DATABASE_URL

-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- Find slow running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes'
AND state = 'active';

-- Check for locks
SELECT blocked_locks.pid     AS blocked_pid,
       blocked_activity.usename  AS blocked_user,
       blocking_locks.pid     AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query    AS blocked_statement,
       blocking_activity.query   AS current_statement_in_blocking_process
FROM  pg_catalog.pg_locks         blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity  ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks         blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;

-- Check database size and table statistics
SELECT schemaname,tablename,attname,n_distinct,correlation FROM pg_stats;

-- Check cache hit ratio
SELECT 
  sum(heap_blks_read) as heap_read,
  sum(heap_blks_hit)  as heap_hit,
  (sum(heap_blks_hit) - sum(heap_blks_read)) / sum(heap_blks_hit) as ratio
FROM pg_statio_user_tables;
```

#### Database Metrics from Prometheus
```bash
# Database connection pool usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=pg_stat_database_numbackends'

# Query execution time
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=pg_stat_statements_mean_time_ms'

# Cache hit ratio
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)'
```

### Step 3: Infrastructure Layer Analysis

#### Resource Usage
```bash
# CPU usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'

# Memory usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'

# Disk I/O
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(node_disk_io_time_seconds_total[5m])'

# Network usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(node_network_receive_bytes_total[5m])'
```

#### Container Resource Usage
```bash
# Container CPU usage
kubectl top pods

# Container memory usage
kubectl describe pod <pod-name> | grep -A5 "Limits\|Requests"

# Check for OOMKilled containers
kubectl get events --field-selector reason=OOMKilled
```

#### Redis Performance
```bash
# Redis metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=redis_connected_clients'

# Memory usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=redis_memory_used_bytes'

# Command stats
redis-cli info commandstats
```

### Step 4: External Dependencies

#### AI API Performance
```bash
# Check OpenAI API response times
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, openai_api_duration_seconds_bucket)'

# API error rates
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(openai_api_errors_total[5m])'

# Rate limiting issues
kubectl logs deployment/agentic-studio | grep "rate limit"
```

## Common Performance Issues and Solutions

### Issue 1: High API Response Times

#### Symptoms
- 95th percentile response time > 500ms
- User complaints about slow application
- Grafana dashboard showing degraded performance

#### Investigation
```bash
# Check which endpoints are slow
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=topk(10, avg by (endpoint) (http_request_duration_seconds))'

# Check for database bottlenecks
kubectl logs deployment/agentic-studio | grep "slow query"
```

#### Solutions
1. **Database Optimization**
   ```sql
   -- Add missing indexes
   CREATE INDEX CONCURRENTLY idx_table_column ON table_name(column_name);
   
   -- Update table statistics
   ANALYZE table_name;
   ```

2. **Query Optimization**
   ```python
   # Add database query optimization
   # Use select_related() and prefetch_related() for ORM queries
   # Implement query result caching
   ```

3. **Resource Scaling**
   ```bash
   # Scale application horizontally
   kubectl scale deployment agentic-studio --replicas=5
   
   # Increase resource limits
   kubectl patch deployment agentic-studio -p '{"spec":{"template":{"spec":{"containers":[{"name":"app","resources":{"limits":{"cpu":"1000m","memory":"2Gi"}}}]}}}}'
   ```

### Issue 2: High Error Rates

#### Symptoms
- Error rate > 1%
- 5xx status codes increasing
- Application throwing exceptions

#### Investigation
```bash
# Check error patterns
kubectl logs deployment/agentic-studio --since=1h | grep ERROR | head -20

# Check specific error types
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(http_requests_total{status=~"5.."}[5m]) by (status)'
```

#### Solutions
1. **Fix Application Bugs**
   - Review error logs for patterns
   - Deploy hotfixes for critical issues
   - Implement better error handling

2. **External Dependency Issues**
   ```bash
   # Check external service health
   curl -f https://api.openai.com/v1/models
   
   # Implement circuit breakers
   # Add retry logic with exponential backoff
   ```

### Issue 3: Database Performance Issues

#### Symptoms
- Slow database queries
- High database CPU usage
- Connection pool exhaustion

#### Investigation
```sql
-- Check for slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
AND n_distinct > 100;
```

#### Solutions
1. **Query Optimization**
   ```sql
   -- Add indexes for frequently queried columns
   CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
   
   -- Optimize queries
   EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'user@example.com';
   ```

2. **Connection Pool Tuning**
   ```python
   # Adjust connection pool settings
   DATABASE_POOL_SIZE = 20
   DATABASE_MAX_OVERFLOW = 10
   DATABASE_POOL_TIMEOUT = 30
   ```

### Issue 4: Memory Issues

#### Symptoms
- High memory usage (>85%)
- OOMKilled containers
- Application crashes

#### Investigation
```bash
# Check memory usage patterns
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100'

# Check for memory leaks
kubectl top pods --sort-by=memory
```

#### Solutions
1. **Memory Optimization**
   ```python
   # Implement proper resource cleanup
   # Use context managers for database connections
   # Clear large objects from memory
   ```

2. **Resource Limits**
   ```yaml
   resources:
     limits:
       memory: "2Gi"
     requests:
       memory: "1Gi"
   ```

### Issue 5: CrewAI Performance Issues

#### Symptoms
- Slow agent task execution
- High AI API costs
- Agent task failures

#### Investigation
```bash
# Check agent performance
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, crewai_task_duration_seconds_bucket) by (agent)'

# Check AI API usage
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(ai_api_calls_total[5m]) by (provider)'
```

#### Solutions
1. **Agent Optimization**
   ```python
   # Optimize agent prompts
   # Implement caching for similar requests
   # Use appropriate AI models for tasks
   ```

2. **Parallel Processing**
   ```python
   # Implement parallel task execution
   # Use async/await for non-blocking operations
   # Optimize crew workflows
   ```

## Escalation Procedures

### Level 1: Developer On-Call (0-30 minutes)
1. Follow this runbook for initial investigation
2. Implement quick fixes if available
3. Scale resources if needed
4. Document findings

### Level 2: Senior Engineer (30-60 minutes)
1. Deep dive into application logs and metrics
2. Implement complex fixes
3. Coordinate with infrastructure team
4. Review architectural decisions

### Level 3: Engineering Manager (60+ minutes)
1. Make decisions about service degradation
2. Coordinate cross-team efforts
3. Communicate with stakeholders
4. Plan longer-term solutions

## Prevention Strategies

### Proactive Monitoring
- Set up comprehensive alerts
- Regular performance reviews
- Capacity planning
- Load testing

### Code Quality
- Performance testing in CI/CD
- Code reviews focusing on performance
- Regular dependency updates
- Database migration reviews

### Infrastructure
- Auto-scaling configurations
- Resource monitoring
- Regular infrastructure audits
- Disaster recovery testing

## Post-Incident Review

### Documentation Requirements
1. Timeline of events
2. Root cause analysis
3. Actions taken
4. Lessons learned
5. Prevention measures

### Follow-up Actions
- Implement monitoring improvements
- Update runbooks
- Code improvements
- Infrastructure optimizations
- Team training updates

## Emergency Contacts

- **On-Call Engineer**: +1-XXX-XXX-XXXX
- **Database Team**: db-team@company.com
- **Infrastructure Team**: infra-team@company.com
- **Security Team**: security@company.com
- **Management Escalation**: engineering-managers@company.com