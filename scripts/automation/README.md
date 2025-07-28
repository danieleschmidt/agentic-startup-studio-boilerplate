# Repository Automation Scripts

This directory contains automation scripts for maintaining the Agentic Startup Studio Boilerplate repository.

## Scripts Overview

### 1. `metrics_collector.py`
Collects comprehensive metrics from various sources and updates the project metrics file.

**Features:**
- Git repository metrics (commits, contributors, branches)
- Code quality metrics (lines of code, complexity)
- Test coverage and results
- Security scan results
- Docker configuration analysis
- GitHub API metrics (stars, forks, issues, PRs)
- Dependency analysis

**Usage:**
```bash
python scripts/automation/metrics_collector.py
```

**Environment Variables:**
- `GITHUB_TOKEN`: GitHub API token for fetching repository metrics
- `GITHUB_REPOSITORY`: Repository name (e.g., "owner/repo")

### 2. `repo_maintenance.py`
Performs comprehensive repository maintenance tasks.

**Features:**
- Dependency update checking
- Security scanning (Bandit, Safety, Trivy)
- Repository cleanup (cache files, temp files, old reports)
- Health checks (Git status, required files, configuration validation)
- Documentation health assessment
- Backup verification
- Performance metrics analysis

**Usage:**
```bash
python scripts/automation/repo_maintenance.py
```

**Generated Reports:**
- `maintenance-report.md`: Detailed maintenance report
- Various security reports (JSON format)

### 3. `scheduler.py`
Automation scheduler that manages periodic execution of maintenance tasks.

**Features:**
- Configurable task scheduling (cron-like)
- Task status tracking
- Notification system (Slack, email)
- Manual task execution
- Task enable/disable management
- Comprehensive logging

**Usage:**
```bash
# Start the scheduler daemon
python scripts/automation/scheduler.py --start

# Check scheduler status
python scripts/automation/scheduler.py --status

# Run a task immediately
python scripts/automation/scheduler.py --run-now metrics_collection

# List available tasks
python scripts/automation/scheduler.py --list-tasks

# Enable/disable tasks
python scripts/automation/scheduler.py --enable dependency_check
python scripts/automation/scheduler.py --disable security_scan
```

## Configuration

### Automation Configuration
The scheduler uses `.github/automation-config.json` for configuration:

```json
{
  "schedules": {
    "metrics_collection": {
      "enabled": true,
      "cron": "0 6 * * *",
      "description": "Collect repository metrics"
    },
    "security_scan": {
      "enabled": true,
      "cron": "0 1 * * *",
      "description": "Run security scans"
    }
  },
  "notifications": {
    "slack_webhook": "https://hooks.slack.com/...",
    "failure_only": true
  },
  "settings": {
    "max_concurrent_tasks": 2,
    "task_timeout_minutes": 60,
    "retry_failed_tasks": true
  }
}
```

### Environment Variables
```bash
# Required for GitHub metrics
export GITHUB_TOKEN="your_github_token"
export GITHUB_REPOSITORY="owner/repository"

# Optional for notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export EMAIL_NOTIFICATIONS="true"
```

## Scheduled Tasks

### Default Schedule

| Task | Schedule | Description |
|------|----------|-------------|
| **Metrics Collection** | Daily 6 AM | Collect and update repository metrics |
| **Dependency Check** | Weekly Mon 2 AM | Check for dependency updates |
| **Security Scan** | Daily 1 AM | Run comprehensive security scans |
| **Repository Cleanup** | Weekly Sun 3 AM | Clean up artifacts and temp files |
| **Health Check** | Every 6 hours | Repository health assessment |
| **Backup Verification** | Weekly Sun 4 AM | Verify backup integrity |
| **Performance Monitoring** | Every 12 hours | Monitor performance metrics |

### Cron Expression Format
```
minute hour day_of_month month day_of_week
  0     6       *         *        *        # Daily at 6 AM
  0     2       *         *        1        # Weekly Monday at 2 AM
  0     */6     *         *        *        # Every 6 hours
```

## Dependencies

### Python Packages
```bash
pip install schedule requests pyyaml toml
```

### Optional Tools (for enhanced functionality)
```bash
# Security scanning
pip install bandit safety

# Code quality analysis
pip install radon

# Container scanning
# Install Trivy: https://aquasecurity.github.io/trivy/

# Secrets detection
pip install detect-secrets
```

## Integration with CI/CD

### GitHub Actions Integration
Add to your workflow:

```yaml
- name: Run Repository Maintenance
  run: |
    python scripts/automation/repo_maintenance.py
    
- name: Collect Metrics
  run: |
    python scripts/automation/metrics_collector.py
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Docker Integration
Run automation in a container:

```dockerfile
# Add to your Dockerfile
COPY scripts/automation/ /app/scripts/automation/
RUN pip install schedule requests pyyaml toml

# Run scheduler
CMD ["python", "/app/scripts/automation/scheduler.py", "--start"]
```

## Monitoring and Alerts

### Status Tracking
Automation status is tracked in `.github/automation-status.json`:

```json
{
  "metrics_collection": {
    "status": "completed",
    "timestamp": "2025-07-28T06:00:00Z",
    "details": {
      "duration_seconds": 15.2,
      "metrics_collected": 8
    }
  }
}
```

### Notifications
- **Slack**: Sends formatted messages to configured webhook
- **Email**: Basic email notifications (requires SMTP configuration)
- **Logs**: Comprehensive logging to `automation-scheduler.log`

### Health Monitoring
The automation system monitors its own health:
- Task execution success/failure rates
- Performance metrics (execution time)
- Resource usage monitoring
- Error tracking and recovery

## Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Install required packages
pip install -r requirements-dev.txt

# Install optional security tools
pip install bandit safety detect-secrets
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/automation/*.py

# Check file permissions
ls -la scripts/automation/
```

#### GitHub API Rate Limits
- Use authenticated requests with `GITHUB_TOKEN`
- Implement retry logic with exponential backoff
- Monitor rate limit headers

#### Task Failures
1. Check `automation-scheduler.log` for detailed errors
2. Review task-specific reports (e.g., `maintenance-report.md`)
3. Verify environment variables and dependencies
4. Run tasks manually for debugging

### Debugging Commands
```bash
# Test metrics collection
python scripts/automation/metrics_collector.py

# Run maintenance with verbose logging
python scripts/automation/repo_maintenance.py

# Check scheduler configuration
python scripts/automation/scheduler.py --status

# Run individual tasks
python scripts/automation/scheduler.py --run-now health_check
```

## Customization

### Adding New Tasks
1. Add task method to `AutomationScheduler` class
2. Update `run_task()` method to include new task
3. Add schedule configuration to default config
4. Test task execution

### Custom Metrics
1. Extend `MetricsCollector` class
2. Add new collection methods
3. Update metrics structure in `project-metrics.json`
4. Add visualization to monitoring dashboards

### Notification Channels
1. Implement new notification method in `send_notification()`
2. Add configuration options
3. Test notification delivery
4. Update documentation

## Security Considerations

### Sensitive Data
- Never commit API tokens or secrets
- Use environment variables for sensitive configuration
- Implement proper access controls for automation logs
- Regular review of notification channels

### Script Security
- Validate all input parameters
- Use subprocess with proper escaping
- Implement timeouts for external calls
- Regular security updates for dependencies

### Access Control
- Limit automation script permissions
- Use read-only tokens where possible
- Implement audit logging
- Regular access reviews

## Best Practices

### Monitoring
- Set up alerts for automation failures
- Monitor resource usage of automation tasks
- Track execution times and performance trends
- Regular review of automation effectiveness

### Maintenance
- Regular updates of automation dependencies
- Review and update task schedules
- Clean up old logs and reports
- Backup automation configuration

### Development
- Test automation changes in development environment
- Use feature flags for experimental automation
- Document all configuration changes
- Implement gradual rollout for automation updates

## Resources

- [Schedule Library Documentation](https://schedule.readthedocs.io/)
- [GitHub API Documentation](https://docs.github.com/en/rest)
- [Slack Webhook Documentation](https://api.slack.com/messaging/webhooks)
- [Cron Expression Guide](https://crontab.guru/)