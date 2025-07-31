# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Agentic Startup Studio Boilerplate.

## Workflows Overview

### 🔄 CI Pipeline (`ci.yml`)
**Triggers**: Pull requests and pushes to main branch

**Features**:
- Code quality checks (linting, formatting, type checking)
- Security scanning (Bandit, Safety, CodeQL, npm audit)
- Unit tests across multiple Python versions (3.9-3.12)
- Docker image building and vulnerability scanning
- Configuration validation
- Quality gate enforcement

**Quality Standards**:
- Code coverage threshold: 80%
- Security scan: Must pass
- All linting: Must pass
- Type checking: Must pass

### 🚀 CD Pipeline (`cd.yml`)
**Triggers**: Successful CI completion on main branch, tags

**Features**:
- Automatic container image building and publishing to GHCR
- SBOM generation for supply chain security
- Staging deployment simulation
- Production deployment with manual approval
- Blue-green deployment strategy (template)
- Deployment summary and notifications

**Deployment Strategy**:
- **Staging**: Automatic deployment after CI passes
- **Production**: Manual approval required + additional checks

### 🔒 Security Scan (`security.yml`)
**Triggers**: Weekly schedule (Monday 2 AM), dependency changes, manual

**Features**:
- Comprehensive dependency vulnerability scanning
- Secret detection across repository history
- Container security scanning with Trivy
- Static Application Security Testing (SAST) with CodeQL and Semgrep
- Security findings uploaded to GitHub Security tab

**Security Tools**:
- **Dependencies**: pip-audit, npm audit
- **Secrets**: detect-secrets
- **Containers**: Trivy
- **SAST**: CodeQL, Semgrep

### 📦 Release (`release.yml`)
**Triggers**: Pushes to main (automatic), manual dispatch

**Features**:
- Semantic versioning with Commitizen
- Automated release notes generation
- GitHub release creation
- Container image publishing with version tags
- Python package publishing (template)
- Documentation updates
- Release notifications

**Release Types**:
- **Automatic**: Based on conventional commits
- **Manual**: Patch/Minor/Major via workflow dispatch

## Configuration Requirements

### Secrets
The following secrets should be configured in GitHub repository settings:

```bash
# Optional: For Slack notifications
SLACK_WEBHOOK_URL

# Optional: For PyPI publishing
PYPI_API_TOKEN

# Optional: For external services
GRAFANA_API_TOKEN
```

### Repository Settings
- **Actions permissions**: Allow GitHub Actions to create and approve pull requests
- **Package permissions**: Allow Actions to write to GitHub Container Registry
- **Security tab**: Enable to view security scan results

## Workflow Status Badges

Add these badges to your README.md:

```markdown
[![CI Pipeline](https://github.com/your-username/your-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/ci.yml)
[![Security Scan](https://github.com/your-username/your-repo/actions/workflows/security.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/security.yml)
[![Release](https://github.com/your-username/your-repo/actions/workflows/release.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/release.yml)
```

## Customization Guide

### Environment-Specific Deployments
1. Update environment URLs in `cd.yml`
2. Configure deployment secrets for your infrastructure
3. Modify deployment commands for your platform (Kubernetes, Docker Swarm, etc.)

### Additional Security Scans
- Add DAST scanning for web applications
- Include license compliance checking
- Integrate with external security tools

### Advanced Release Features
- Add changelog automation
- Include artifact signing
- Configure multi-registry publishing

## Monitoring and Alerts

### Workflow Failure Alerts
- Configure GitHub branch protection rules
- Set up team notifications for critical failures
- Monitor workflow execution times and success rates

### Security Alert Management
- Review security findings in GitHub Security tab
- Configure automated issue creation for high-severity findings
- Set up regular security report reviews

## Best Practices

### Workflow Maintenance
- Regularly update action versions
- Review and update security scan configurations
- Monitor workflow performance and optimize as needed

### Security Considerations
- Use least-privilege permissions for all workflows
- Regularly rotate secrets and tokens
- Review and approve third-party actions before use

### Performance Optimization
- Use caching for dependencies and build artifacts
- Parallelize independent jobs
- Minimize workflow execution time

## Troubleshooting

### Common Issues
1. **Permission denied**: Check repository/workflow permissions
2. **Secrets not found**: Verify secret names and availability
3. **Cache issues**: Clear GitHub Actions cache if needed
4. **Dependency conflicts**: Review requirements.txt and package.json

### Debug Mode
Enable debug logging by setting repository secret:
```
ACTIONS_STEP_DEBUG=true
```

## Integration with Existing Tools

These workflows integrate seamlessly with:
- **Pre-commit hooks**: Consistency between local and CI checks
- **Dependabot**: Automated dependency updates
- **Renovate**: Advanced dependency management
- **CodeQL**: Advanced security analysis
- **Docker**: Container building and security scanning

## Support

For issues with these workflows:
1. Check the workflow run logs in GitHub Actions tab
2. Review the troubleshooting section above
3. Consult the individual workflow documentation
4. Create an issue in the repository for assistance