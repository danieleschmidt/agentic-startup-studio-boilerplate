# Monitoring & Observability

This directory contains the complete monitoring and observability setup for the Agentic Startup Studio Boilerplate.

## Architecture Overview

Our observability stack follows industry best practices with the three pillars of observability:

### 🔢 Metrics (Prometheus + Grafana)
- **Collection**: Prometheus scrapes metrics from application and infrastructure
- **Storage**: Time-series data stored in Prometheus with configurable retention
- **Visualization**: Grafana dashboards provide real-time and historical views
- **Alerting**: AlertManager handles alert routing and notification

### 📝 Logs (Structured Logging)
- **Format**: Structured JSON logs with contextual information
- **Collection**: Centralized log aggregation with log rotation
- **Processing**: Log parsing and enrichment for better searchability
- **Storage**: Configurable log retention and archival

### 🔍 Traces (OpenTelemetry)
- **Instrumentation**: Automatic and manual tracing of requests
- **Collection**: OpenTelemetry collectors gather trace data
- **Analysis**: Distributed tracing for performance optimization
- **Correlation**: Linking traces with logs and metrics

## Quick Start

### Local Development
```bash
# Start monitoring stack
docker-compose -f docker-compose.dev.yml up -d prometheus grafana

# Access dashboards
open http://localhost:3001  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

### Production Deployment
```bash
# Deploy monitoring stack to Kubernetes
kubectl apply -f k8s/monitoring/

# Verify deployment
kubectl get pods -n monitoring
```

## Component Details

### Prometheus Configuration
- **File**: `prometheus.yml`
- **Scrape Interval**: 15 seconds
- **Retention**: 30 days (configurable)
- **Targets**: Application, database, Redis, system metrics

### Grafana Dashboards
- **Application Performance**: Response times, throughput, errors
- **Infrastructure**: CPU, memory, disk, network metrics  
- **Business Metrics**: User activity, feature usage, revenue
- **CrewAI Metrics**: Agent performance, task completion, AI costs

### Alert Rules
- **File**: `alert_rules.yml`
- **Categories**: SLA breaches, resource exhaustion, errors
- **Notifications**: Slack, email, PagerDuty integration
- **Escalation**: Automatic escalation for critical alerts

## Metrics Collection

### Application Metrics

#### API Performance
```python
# Example metrics instrumentation
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(duration)
    return response
```

#### CrewAI Metrics
```python
# CrewAI performance tracking
AGENT_TASK_COUNT = Counter('crewai_tasks_total', 'Total tasks executed', ['agent', 'status'])
AGENT_DURATION = Histogram('crewai_task_duration_seconds', 'Task execution time', ['agent'])
AI_API_CALLS = Counter('ai_api_calls_total', 'AI API calls', ['provider', 'model'])
AI_COSTS = Counter('ai_costs_total', 'AI API costs in USD', ['provider'])

class MetricsCollector:
    @staticmethod
    def record_task_completion(agent_name: str, duration: float, status: str):
        AGENT_TASK_COUNT.labels(agent=agent_name, status=status).inc()
        AGENT_DURATION.labels(agent=agent_name).observe(duration)
    
    @staticmethod
    def record_ai_call(provider: str, model: str, cost: float):
        AI_API_CALLS.labels(provider=provider, model=model).inc()
        AI_COSTS.labels(provider=provider).inc(cost)
```

### Infrastructure Metrics

#### Database Metrics (PostgreSQL)
- Connection pool usage
- Query performance
- Lock statistics
- Replication lag

#### Redis Metrics
- Memory usage
- Key statistics
- Command statistics
- Persistence metrics

#### System Metrics
- CPU utilization
- Memory usage
- Disk I/O
- Network traffic

## Log Management

### Structured Logging Configuration

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'agent_name'):
            log_entry['agent_name'] = record.agent_name
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)
logger.handlers[0].setFormatter(JSONFormatter())
```

### Log Categories

#### Application Logs
- API requests and responses
- Authentication events
- Business logic events
- Error conditions

#### Security Logs
- Authentication attempts
- Authorization failures
- Security policy violations
- Audit trail events

#### Performance Logs
- Slow queries
- Long-running requests
- Resource exhaustion
- Performance bottlenecks

## Distributed Tracing

### OpenTelemetry Setup

```python
# tracing_config.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing(app, service_name="agentic-startup-api"):
    # Configure tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument frameworks
    FastAPIInstrumentor.instrument_app(app)
    SQLAlchemyInstrumentor().instrument()
    RedisInstrumentor().instrument()
    
    return tracer

# Custom tracing for CrewAI
class TracedCrew:
    def __init__(self, crew, tracer):
        self.crew = crew
        self.tracer = tracer
    
    async def kickoff(self, inputs):
        with self.tracer.start_as_current_span("crew_execution") as span:
            span.set_attribute("crew.agents_count", len(self.crew.agents))
            span.set_attribute("crew.input_keys", list(inputs.keys()))
            
            try:
                result = await self.crew.kickoff(inputs)
                span.set_attribute("crew.status", "success")
                return result
            except Exception as e:
                span.set_attribute("crew.status", "error")
                span.set_attribute("crew.error", str(e))
                raise
```

## Dashboards

### Grafana Dashboard Configuration

#### Application Performance Dashboard
- Request rate and response times
- Error rates and status codes
- Database query performance
- Cache hit rates
- Active user sessions

#### Infrastructure Dashboard
- System resource utilization
- Container metrics
- Network performance
- Storage usage

#### Business Metrics Dashboard
- User engagement metrics
- Feature adoption rates
- Revenue and conversion metrics
- AI usage and costs

#### CrewAI Performance Dashboard
- Agent task completion rates
- Average task execution time
- AI API usage and costs
- Agent error rates
- Workflow efficiency metrics

### Custom Dashboard Creation

```json
{
  "dashboard": {
    "title": "CrewAI Performance",
    "panels": [
      {
        "title": "Task Completion Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(crewai_tasks_total{status=\"success\"}[5m]) / rate(crewai_tasks_total[5m]) * 100"
          }
        ]
      },
      {
        "title": "Agent Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, crewai_task_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, crewai_task_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Alert Rules Configuration

```yaml
# alert_rules.yml
groups:
  - name: application_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
      
      - alert: CrewAITaskFailures
        expr: rate(crewai_tasks_total{status="error"}[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "CrewAI task failures detected"
          description: "{{ $value }} task failures per second"

  - name: infrastructure_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
      
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
```

### Notification Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@yourdomain.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://slack-webhook-url'
        send_resolved: true
    
  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourdomain.com'
        subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
    
    pagerduty_configs:
      - service_key: 'your-pagerduty-integration-key'
        description: '{{ .GroupLabels.alertname }}'
```

## Health Checks

### Application Health Endpoints

```python
# health_checks.py
from fastapi import APIRouter, HTTPException
from sqlalchemy import text
import redis
import httpx

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.2.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with dependency status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Database health
    try:
        result = await database.fetch_one("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Redis health
    try:
        redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # External API health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.openai.com/v1/models", timeout=5)
            if response.status_code == 200:
                health_status["services"]["openai"] = "healthy"
            else:
                health_status["services"]["openai"] = f"unhealthy: HTTP {response.status_code}"
    except Exception as e:
        health_status["services"]["openai"] = f"unhealthy: {str(e)}"
    
    return health_status

@router.get("/health/readiness")
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if application is ready to serve traffic
    if not application_ready:
        raise HTTPException(status_code=503, detail="Application not ready")
    return {"status": "ready"}

@router.get("/health/liveness")
async def liveness_check():
    """Kubernetes liveness probe."""
    # Check if application is alive (basic check)
    return {"status": "alive"}
```

## Performance Monitoring

### SLI/SLO Configuration

```python
# sli_slo_config.py
SLO_TARGETS = {
    "api_availability": {
        "target": 99.9,  # 99.9% availability
        "measurement_window": "30d",
        "alert_threshold": 99.5
    },
    "api_latency": {
        "target": 200,  # 200ms 95th percentile
        "measurement_window": "1h", 
        "alert_threshold": 500
    },
    "error_rate": {
        "target": 0.1,  # 0.1% error rate
        "measurement_window": "1h",
        "alert_threshold": 1.0
    }
}

class SLOMonitor:
    def __init__(self, prometheus_client):
        self.prometheus = prometheus_client
    
    async def check_availability_slo(self):
        query = 'avg_over_time(up[30d]) * 100'
        result = await self.prometheus.query(query)
        current_availability = float(result['data']['result'][0]['value'][1])
        return current_availability >= SLO_TARGETS["api_availability"]["target"]
    
    async def check_latency_slo(self):
        query = 'histogram_quantile(0.95, http_request_duration_seconds_bucket[1h])'
        result = await self.prometheus.query(query)
        current_latency = float(result['data']['result'][0]['value'][1]) * 1000
        return current_latency <= SLO_TARGETS["api_latency"]["target"]
```

## Troubleshooting

### Common Monitoring Issues

#### High Cardinality Metrics
```python
# Avoid high cardinality labels
# BAD: Including user ID in metric labels
user_requests = Counter('user_requests_total', 'Requests per user', ['user_id'])

# GOOD: Use histograms or sampling for high cardinality data
request_count = Counter('requests_total', 'Total requests', ['endpoint', 'method'])
```

#### Missing Metrics
```bash
# Check if application is exposing metrics
curl http://localhost:8000/metrics

# Verify Prometheus is scraping targets
curl http://localhost:9090/api/v1/targets

# Check Prometheus configuration
promtool check config prometheus.yml
```

#### Dashboard Issues
- Verify metric names and labels in queries
- Check time ranges and aggregation functions
- Ensure proper data source configuration
- Validate PromQL query syntax

### Performance Optimization

#### Prometheus Optimization
```yaml
# prometheus.yml optimizations
global:
  scrape_interval: 30s  # Reduce frequency for less critical metrics
  scrape_timeout: 10s
  evaluation_interval: 30s

# Storage optimization
storage:
  tsdb:
    retention.time: 15d  # Adjust based on storage capacity
    retention.size: 10GB
    min-block-duration: 2h
    max-block-duration: 24h
```

#### Grafana Optimization
- Use template variables for dynamic dashboards
- Implement proper caching strategies
- Optimize query performance with recording rules
- Use appropriate refresh intervals

## Best Practices

### Metrics Best Practices
1. **Naming Convention**: Use descriptive, consistent naming
2. **Labels**: Keep cardinality low, use meaningful labels
3. **Units**: Always specify units in metric names
4. **Help Text**: Provide clear descriptions for all metrics
5. **Recording Rules**: Pre-compute expensive queries

### Logging Best Practices
1. **Structured Format**: Use JSON for machine-readable logs
2. **Context**: Include relevant context (user_id, request_id)
3. **Log Levels**: Use appropriate log levels (DEBUG, INFO, WARN, ERROR)
4. **Sensitive Data**: Never log sensitive information
5. **Performance**: Use asynchronous logging for high-throughput

### Alerting Best Practices
1. **Actionable**: Alerts should require immediate action
2. **Context**: Provide sufficient context for investigation
3. **Escalation**: Implement proper escalation procedures
4. **Documentation**: Include runbooks for alert resolution
5. **Testing**: Regularly test alert configurations

## Integration with CI/CD

### Monitoring in Deployment Pipeline

```yaml
# .github/workflows/deploy.yml
- name: Deploy Monitoring Stack
  run: |
    kubectl apply -f k8s/monitoring/
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=ready pod -l app=prometheus --timeout=300s
    
    # Validate monitoring setup
    curl -f http://prometheus:9090/-/healthy
    curl -f http://grafana:3000/api/health

- name: Run SLO Checks
  run: |
    python scripts/validate_slos.py --environment=staging
```

### Monitoring-as-Code

```python
# monitoring_config.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AlertRule:
    name: str
    expr: str
    duration: str
    severity: str
    summary: str
    description: str

@dataclass
class Dashboard:
    title: str
    panels: List[Dict]
    tags: List[str]

class MonitoringConfig:
    def __init__(self):
        self.alert_rules = []
        self.dashboards = []
    
    def add_alert_rule(self, rule: AlertRule):
        self.alert_rules.append(rule)
    
    def generate_prometheus_rules(self):
        # Generate Prometheus alert rules YAML
        pass
    
    def generate_grafana_dashboards(self):
        # Generate Grafana dashboard JSON
        pass
```

This monitoring setup provides comprehensive observability for the Agentic Startup Studio Boilerplate, enabling proactive issue detection and performance optimization.