# GitHub Actions Workflow Setup Instructions

This guide provides step-by-step instructions for manually setting up GitHub Actions workflows from the provided templates.

## Prerequisites

Before setting up workflows, ensure you have:

- [ ] Repository admin access
- [ ] Access to create GitHub secrets
- [ ] Docker registry account (if using containerization)
- [ ] Cloud provider accounts (if deploying to cloud)

## Step 1: Create Workflow Directory

```bash
# Navigate to your repository root
cd /path/to/your/repository

# Create the GitHub workflows directory
mkdir -p .github/workflows

# Verify the directory was created
ls -la .github/
```

## Step 2: Copy Workflow Templates

Copy the workflow templates from the documentation to your workflows directory:

### Core Workflows (Required)

```bash
# Copy core CI/CD workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/

# Copy security and quality workflows
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/

# Copy release automation
cp docs/workflows/examples/release.yml .github/workflows/
```

### Specialized Workflows (Optional)

```bash
# Copy AI/ML specific workflows
cp docs/workflows/examples/ai-model-validation.yml .github/workflows/

# Copy performance testing workflows
cp docs/workflows/examples/performance.yml .github/workflows/

# Copy infrastructure workflows
cp docs/workflows/examples/infrastructure.yml .github/workflows/
```

## Step 3: Configure Repository Secrets

Navigate to your GitHub repository settings and add the following secrets:

### Required Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `DOCKER_REGISTRY_URL` | Container registry URL | `ghcr.io` or `docker.io` |
| `DOCKER_REGISTRY_USERNAME` | Registry username | `your-username` |
| `DOCKER_REGISTRY_PASSWORD` | Registry password/token | `ghp_xxxxxxxxxxxxx` |

### AI/ML Secrets

| Secret Name | Description | Notes |
|-------------|-------------|-------|
| `OPENAI_API_KEY` | OpenAI API key | Required for GPT models |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Claude models |
| `HUGGINGFACE_TOKEN` | Hugging Face token | For model downloads |

### Cloud Provider Secrets

#### AWS Deployment
```bash
# Add these secrets for AWS deployment
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
```

#### Google Cloud Deployment
```bash
# Add these secrets for GCP deployment
GCP_PROJECT_ID=your-project-id
GCP_SERVICE_ACCOUNT_KEY=your-service-account-json
GCP_REGION=us-central1
```

#### Azure Deployment
```bash
# Add these secrets for Azure deployment
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_TENANT_ID=your-tenant-id
AZURE_SUBSCRIPTION_ID=your-subscription-id
```

### Notification Secrets (Optional)

| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `SLACK_WEBHOOK_URL` | Slack webhook for notifications | Build status alerts |
| `DISCORD_WEBHOOK_URL` | Discord webhook | Team notifications |
| `EMAIL_SMTP_CONFIG` | SMTP configuration | Email notifications |

### Monitoring and Observability

| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `SONAR_TOKEN` | SonarCloud token | Code quality analysis |
| `CODECOV_TOKEN` | Codecov token | Coverage reporting |
| `SENTRY_DSN` | Sentry DSN | Error tracking |

## Step 4: Set Up GitHub Environments

Create the following environments in your repository settings:

### Development Environment
```yaml
Environment name: development
Protection rules: None
Environment secrets: 
  - DATABASE_URL: postgresql://dev-db-url
  - REDIS_URL: redis://dev-redis-url
```

### Staging Environment
```yaml
Environment name: staging
Protection rules:
  - Required reviewers: 1
  - Wait timer: 0 minutes
Environment secrets:
  - DATABASE_URL: postgresql://staging-db-url
  - REDIS_URL: redis://staging-redis-url
```

### Production Environment
```yaml
Environment name: production
Protection rules:
  - Required reviewers: 2
  - Wait timer: 5 minutes
  - Deployment branches: main only
Environment secrets:
  - DATABASE_URL: postgresql://prod-db-url
  - REDIS_URL: redis://prod-redis-url
```

## Step 5: Configure Branch Protection Rules

Set up branch protection for your main branch:

### Basic Protection Rules

```bash
# Navigate to Settings > Branches in your GitHub repository
# Create a rule for 'main' branch with these settings:

✅ Require a pull request before merging
  ✅ Require approvals (2 reviewers recommended)
  ✅ Dismiss stale reviews when new commits are pushed
  ✅ Require review from code owners

✅ Require status checks to pass before merging
  ✅ Require branches to be up to date before merging
  Required status checks:
    - CI Pipeline / quality-gate
    - Security Analysis / security-scan
    - Unit Tests / unit-tests
    - Integration Tests / integration-tests

✅ Require conversation resolution before merging
✅ Require signed commits
✅ Require linear history
✅ Restrict pushes that create files
```

### Advanced Protection Rules

```bash
# For enterprise repositories, also enable:

✅ Restrict force pushes
✅ Allow specified actors to bypass required pull requests
  - Service accounts for automated releases
  - Emergency deployment accounts

✅ Allow specified actors to bypass required status checks
  - Only for critical hotfixes with approval
```

## Step 6: Configure Workflow Variables

Set up repository variables for common configuration:

### Repository Variables

| Variable Name | Value | Description |
|---------------|-------|-------------|
| `PYTHON_VERSION` | `3.11` | Default Python version |
| `NODE_VERSION` | `18` | Default Node.js version |
| `DOCKER_REGISTRY` | `ghcr.io/${{ github.repository_owner }}` | Container registry |
| `DEFAULT_BRANCH` | `main` | Default branch name |

### Environment-Specific Variables

```bash
# Development Environment Variables
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG_MODE=true

# Staging Environment Variables  
ENVIRONMENT=staging
LOG_LEVEL=INFO
DEBUG_MODE=false

# Production Environment Variables
ENVIRONMENT=production
LOG_LEVEL=WARNING
DEBUG_MODE=false
```

## Step 7: Test Workflow Configuration

### Initial Workflow Test

1. **Create a test branch:**
   ```bash
   git checkout -b test/workflow-setup
   echo "# Workflow Test" > test-file.md
   git add test-file.md
   git commit -m "test: verify workflow configuration"
   git push origin test/workflow-setup
   ```

2. **Create a Pull Request:**
   - Navigate to your repository
   - Create a PR from `test/workflow-setup` to `main`
   - Verify that workflows are triggered

3. **Monitor Workflow Execution:**
   - Go to Actions tab in GitHub
   - Watch for successful execution of:
     - CI Pipeline
     - Security Scan
     - Code Quality checks

### Workflow Validation Checklist

After setting up workflows, verify:

- [ ] CI pipeline runs on pull requests
- [ ] Security scans complete without critical issues
- [ ] Docker images build successfully
- [ ] Tests pass in all environments
- [ ] Branch protection rules are enforced
- [ ] Deployment workflows are triggered correctly
- [ ] Notifications are sent to configured channels

## Step 8: Customize Workflows for Your Project

### Modify CI Pipeline

Edit `.github/workflows/ci.yml` to match your project:

```yaml
# Update Python/Node versions
env:
  PYTHON_VERSION: '3.11'  # Change to your version
  NODE_VERSION: '18'      # Change to your version

# Add project-specific test commands
- name: Run custom tests
  run: |
    # Add your custom test commands here
    pytest tests/custom/
    npm run test:custom
```

### Configure Deployment Targets

Edit `.github/workflows/cd.yml` for your deployment:

```yaml
# Update deployment configuration
deploy-production:
  environment: production
  steps:
    - name: Deploy to your infrastructure
      run: |
        # Add your deployment commands
        kubectl apply -f k8s/
        helm upgrade myapp ./charts/myapp
```

### Customize AI/ML Workflows

If using AI features, edit `.github/workflows/ai-model-validation.yml`:

```yaml
# Update model configurations
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  # Add your specific model configurations

# Add custom model tests
- name: Test custom models
  run: |
    # Add your model-specific tests
    python tests/models/test_custom_model.py
```

## Step 9: Monitor and Maintain Workflows

### Regular Maintenance Tasks

1. **Update Dependencies:**
   ```bash
   # Monthly: Update action versions
   # Check for security updates in workflow dependencies
   ```

2. **Review Workflow Performance:**
   ```bash
   # Weekly: Check workflow execution times
   # Optimize slow-running jobs
   ```

3. **Audit Security:**
   ```bash
   # Monthly: Review secrets and permissions
   # Update expired credentials
   ```

### Workflow Monitoring

Set up monitoring for:

- Workflow success/failure rates
- Build duration trends
- Resource usage patterns
- Security scan results

## Troubleshooting Common Issues

### Workflow Not Triggering

**Problem:** Workflow doesn't run on push/PR

**Solutions:**
1. Check workflow file syntax: `cat .github/workflows/ci.yml | yaml-lint`
2. Verify trigger conditions match your branch names
3. Check if paths-ignore is excluding your changes
4. Ensure workflow file is in correct location

### Permission Denied Errors

**Problem:** Workflow fails with permission errors

**Solutions:**
1. Check GITHUB_TOKEN permissions in workflow
2. Verify repository settings allow workflow execution
3. Update workflow permissions section:
   ```yaml
   permissions:
     contents: read
     security-events: write
     actions: read
   ```

### Secret Not Found Errors

**Problem:** Workflow can't access secrets

**Solutions:**
1. Verify secret names match exactly (case-sensitive)
2. Check secret is set at repository level, not personal
3. For environment secrets, ensure environment name matches
4. Verify workflow job has access to environment

### Build Failures

**Problem:** Tests or builds fail in CI

**Solutions:**
1. Run tests locally first: `make test`
2. Check dependency versions match local environment
3. Verify environment variables are set correctly
4. Review workflow logs for specific error messages

## Getting Help

If you encounter issues:

1. **Check GitHub Documentation:** [GitHub Actions Docs](https://docs.github.com/en/actions)
2. **Review Workflow Logs:** Actions tab in your repository
3. **Test Locally:** Use `act` to run workflows locally
4. **Community Support:** GitHub Community Forum
5. **Professional Support:** GitHub Support (for Enterprise)

## Next Steps

After successful workflow setup:

1. **Configure Monitoring:** Set up dashboards for workflow metrics
2. **Optimize Performance:** Profile and optimize slow workflows  
3. **Enhance Security:** Regular security audits and updates
4. **Document Customizations:** Keep internal documentation current
5. **Train Team:** Ensure team understands workflow processes