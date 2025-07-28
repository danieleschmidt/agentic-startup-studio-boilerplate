# CI/CD Workflows Documentation

This directory contains comprehensive documentation and templates for GitHub Actions workflows that should be manually created by repository maintainers.

## Overview

Due to GitHub App permission limitations, the Terragon automation cannot directly create workflow files. Repository maintainers must manually create these workflows from the provided templates.

## Required Workflows

### Core Workflows
1. **[CI Pipeline](examples/ci.yml)** - Pull request validation, testing, security scanning
2. **[CD Pipeline](examples/cd.yml)** - Deployment automation for staging/production
3. **[Dependency Updates](examples/dependency-update.yml)** - Automated dependency management
4. **[Security Scanning](examples/security-scan.yml)** - Comprehensive security analysis
5. **[Release Automation](examples/release.yml)** - Semantic versioning and release management

### Quality Assurance Workflows
6. **[Code Quality](examples/code-quality.yml)** - Linting, formatting, and quality gates
7. **[Performance Testing](examples/performance.yml)** - Load testing and performance regression
8. **[E2E Testing](examples/e2e-tests.yml)** - End-to-end testing with Playwright
9. **[Infrastructure Validation](examples/infrastructure.yml)** - Terraform validation and security

### Specialized Workflows
10. **[AI Model Testing](examples/ai-model-tests.yml)** - AI/ML model validation and testing
11. **[Database Migrations](examples/db-migrations.yml)** - Automated database migration management
12. **[Backup Automation](examples/backup.yml)** - Automated backup and disaster recovery testing

## Setup Instructions

### 1. Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### 2. Copy Template Files
Copy the example workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
# ... copy other required workflows
```

### 3. Configure Secrets
Set up the following repository secrets in GitHub:

#### Required Secrets
- `GITHUB_TOKEN` (automatically provided)
- `DOCKER_REGISTRY_URL` - Container registry URL
- `DOCKER_REGISTRY_USERNAME` - Registry username
- `DOCKER_REGISTRY_PASSWORD` - Registry password/token

#### Optional Secrets (based on integrations)
- `SLACK_WEBHOOK_URL` - Slack notifications
- `SONAR_TOKEN` - SonarCloud integration
- `CODECOV_TOKEN` - Code coverage reporting
- `SENTRY_DSN` - Error tracking
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS deployments
- `GCP_SERVICE_ACCOUNT_KEY` - Google Cloud deployments

### 4. Configure Environments
Create the following GitHub environments:
- `development` - Development deployments
- `staging` - Staging environment
- `production` - Production environment (with required reviewers)

### 5. Set Branch Protection Rules
Configure branch protection for `main` branch:
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date
- Restrict pushes to specific people/teams

## Workflow Documentation

### [CI Pipeline](ci-pipeline.md)
Continuous Integration workflow that runs on every pull request:
- Code quality checks (linting, formatting)
- Unit and integration tests
- Security scanning
- Build validation
- Test coverage reporting

### [CD Pipeline](cd-pipeline.md)  
Continuous Deployment workflow for automated deployments:
- Environment-specific deployments
- Database migrations
- Health checks and rollback
- Deployment notifications

### [Security Workflows](security-workflows.md)
Comprehensive security scanning and validation:
- SAST (Static Application Security Testing)
- DAST (Dynamic Application Security Testing)  
- Dependency vulnerability scanning
- Container security scanning
- Infrastructure security validation

### [AI/ML Workflows](ai-ml-workflows.md)
Specialized workflows for AI/ML components:
- Model validation and testing
- Performance benchmarking
- Drift detection
- A/B testing automation

## Workflow Templates

### Basic Template Structure
```yaml
name: Workflow Name
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  job-name:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        # Setup steps
      - name: Test
        # Test steps
      - name: Deploy
        # Deployment steps
```

### Environment-Specific Deployments
```yaml
deploy:
  if: github.ref == 'refs/heads/main'
  needs: [test, security-scan]
  runs-on: ubuntu-latest
  environment: production
  steps:
    - name: Deploy to Production
      run: |
        # Deployment commands
```

### Matrix Testing
```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: [3.9, 3.10, 3.11, 3.12]
      node-version: [18, 20]
  steps:
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
```

## Best Practices

### Security Best Practices
1. **Use OIDC tokens** instead of long-lived secrets when possible
2. **Limit token permissions** to minimum required scope
3. **Use environments** for production deployments with required reviewers
4. **Scan for secrets** in code and prevent commits
5. **Regular security audits** of workflows and dependencies

### Performance Best Practices
1. **Cache dependencies** to speed up builds
2. **Use matrix builds** for parallel testing
3. **Conditional execution** to skip unnecessary jobs
4. **Artifact management** to share build outputs
5. **Efficient Docker builds** with layer caching

### Reliability Best Practices
1. **Retry mechanisms** for flaky tests and external dependencies
2. **Proper error handling** and meaningful failure messages
3. **Rollback procedures** for failed deployments
4. **Health checks** after deployments
5. **Monitoring integration** for workflow success/failure tracking

## Monitoring and Observability

### Workflow Metrics
Track workflow performance and reliability:
- Build success/failure rates
- Build duration trends
- Test execution times
- Deployment frequency
- Mean time to recovery (MTTR)

### Integration with Monitoring Stack
```yaml
- name: Send Metrics
  if: always()
  run: |
    curl -X POST prometheus-pushgateway:9091/metrics/job/github-actions \
      -d "github_workflow_duration_seconds ${{ job.duration }}"
```

### Alerting on Workflow Failures
Set up alerts for:
- Consecutive build failures
- Security scan failures
- Deployment failures
- Performance regression

## Troubleshooting

### Common Issues

#### Build Failures
- Check action versions compatibility
- Verify environment variables and secrets
- Review dependency conflicts
- Check resource limits

#### Permission Issues
- Verify GITHUB_TOKEN permissions
- Check repository settings
- Validate environment protection rules
- Review branch protection settings

#### Deployment Issues
- Verify deployment environment accessibility
- Check service health after deployment
- Validate configuration changes
- Review rollback procedures

### Debug Techniques
```yaml
- name: Debug Environment
  run: |
    echo "Event: ${{ github.event_name }}"
    echo "Ref: ${{ github.ref }}"
    echo "SHA: ${{ github.sha }}"
    env | sort
```

## Advanced Features

### Custom Actions
Create reusable actions for common tasks:
```yaml
# .github/actions/setup-app/action.yml
name: 'Setup Application'
description: 'Setup application dependencies'
inputs:
  python-version:
    description: 'Python version'
    required: true
    default: '3.11'
runs:
  using: 'composite'
  steps:
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ inputs.python-version }}
```

### Workflow Composition
```yaml
jobs:
  call-reusable-workflow:
    uses: ./.github/workflows/reusable-workflow.yml
    with:
      environment: staging
    secrets: inherit
```

### Dynamic Configuration
```yaml
- name: Set Environment Variables
  run: |
    if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
      echo "ENVIRONMENT=production" >> $GITHUB_ENV
    else
      echo "ENVIRONMENT=staging" >> $GITHUB_ENV
    fi
```

## Migration Guide

### From Jenkins
- Map Jenkins pipeline stages to GitHub Actions jobs
- Convert Groovy scripts to shell commands or actions
- Migrate credentials to GitHub secrets
- Update artifact management

### From GitLab CI
- Convert `.gitlab-ci.yml` to workflow files
- Map GitLab variables to GitHub secrets
- Update Docker registry configurations
- Migrate deployment scripts

### From CircleCI
- Convert `.circleci/config.yml` to workflows
- Map CircleCI orbs to GitHub actions
- Update environment configurations
- Migrate deployment processes

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Marketplace Actions](https://github.com/marketplace?type=actions)
- [Self-hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)