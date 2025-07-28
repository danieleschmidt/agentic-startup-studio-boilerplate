# Manual Setup Requirements

## GitHub Actions Workflows
Due to permission limitations, GitHub Actions workflows must be created manually:

1. **Create workflow directory**:
   ```bash
   mkdir -p .github/workflows
   ```

2. **Copy template files** from `docs/workflows/examples/` to `.github/workflows/`

3. **Configure repository secrets** (see docs/workflows/README.md for complete list)

## Repository Settings

### Branch Protection Rules
Configure protection for `main` branch:
- Require pull request reviews (minimum 1)
- Require status checks to pass
- Require branches to be up to date before merging
- Restrict pushes to administrators

### Repository Topics
Add relevant topics for discoverability:
- `agentic-startup`
- `cookiecutter-template` 
- `fastapi`
- `react`
- `crewai`

### Security Settings
- Enable dependency alerts
- Enable security advisories
- Configure code scanning alerts

## External Integrations

### Required for Full SDLC
- **Monitoring**: Prometheus/Grafana setup
- **Security**: SonarCloud or CodeQL
- **Coverage**: Codecov integration
- **Notifications**: Slack webhook configuration

### Optional Enhancements
- **Error Tracking**: Sentry integration
- **Performance**: DataDog or New Relic
- **Documentation**: GitBook or Notion integration