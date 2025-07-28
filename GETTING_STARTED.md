# Getting Started with Agentic Startup Studio Boilerplate

Welcome to the Agentic Startup Studio Boilerplate - a comprehensive foundation for building AI-powered startups with CrewAI, FastAPI, and React. This guide will help you get up and running quickly with your new agentic startup project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Deployment](#deployment)
- [Monitoring and Observability](#monitoring-and-observability)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Prerequisites

### Required Software

- **Docker** (v20.10+) and **Docker Compose** (v2.0+)
- **Git** (v2.30+)
- **Node.js** (v18+) and **npm** (v8+)
- **Python** (v3.9+) and **pip** (v21+)

### Optional but Recommended

- **VS Code** with the Dev Containers extension
- **GitHub CLI** for repository management
- **Make** for build automation

### System Requirements

- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free disk space
- **Network**: Stable internet connection for package downloads

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/danieleschmidt/agentic-startup-studio-boilerplate.git
cd agentic-startup-studio-boilerplate
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section below)
nano .env
```

### 3. Start with Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check frontend
open http://localhost:3000

# Check monitoring
open http://localhost:3001  # Grafana
open http://localhost:9090  # Prometheus
```

## Development Environment Setup

### Option 1: Dev Containers (Recommended)

If using VS Code with Dev Containers:

1. Open the project in VS Code
2. When prompted, click "Reopen in Container"
3. Wait for the container to build and start
4. All dependencies will be automatically installed

### Option 2: Local Development

#### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run database migrations (if applicable)
alembic upgrade head

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

#### Additional Development Tools

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
npm run lint        # Frontend
python -m flake8    # Backend

# Run type checking
npm run type-check  # Frontend
mypy .             # Backend

# Run tests
npm test           # Frontend
pytest             # Backend
```

## Project Architecture

### High-Level Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Web     │    │   FastAPI       │    │   CrewAI        │
│   Frontend      │◄──►│   Backend       │◄──►│   Agents        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Database      │
                    └─────────────────┘
```

### Directory Structure

```
├── app/                    # FastAPI backend application
│   ├── api/               # API routes and endpoints
│   ├── core/              # Core configuration and utilities
│   ├── models/            # Database models
│   ├── services/          # Business logic services
│   └── agents/            # CrewAI agents and crews
├── frontend/              # React frontend application
│   ├── src/               # Frontend source code
│   ├── components/        # React components
│   └── pages/             # Page components
├── docs/                  # Documentation
│   ├── api/               # API documentation
│   ├── guides/            # User and developer guides
│   └── adr/               # Architecture Decision Records
├── scripts/               # Automation and utility scripts
│   └── automation/        # Repository automation
├── monitoring/            # Monitoring and observability
│   ├── prometheus.yml     # Prometheus configuration
│   └── grafana/           # Grafana dashboards
├── tests/                 # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
└── .github/               # GitHub workflows and templates
```

### Technology Stack

- **Backend**: FastAPI, Python 3.9+, SQLAlchemy, PostgreSQL
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **AI Framework**: CrewAI for multi-agent orchestration
- **Database**: PostgreSQL with Alembic migrations
- **Caching**: Redis for session and application caching
- **Monitoring**: Prometheus, Grafana, custom metrics
- **Testing**: Jest, React Testing Library, pytest, Playwright
- **Deployment**: Docker, Docker Compose, GitHub Actions

## Configuration

### Environment Variables

Edit your `.env` file with the following required variables:

```bash
# Application
APP_NAME=agentic-startup-studio
APP_VERSION=0.2.0
DEBUG=true
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_startup
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=agentic_startup

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-this
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# AI Configuration
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
CREW_AI_API_KEY=your-crewai-api-key

# External APIs
GITHUB_TOKEN=your-github-token
SLACK_WEBHOOK_URL=your-slack-webhook-url

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### AI Agent Configuration

Configure your AI agents in `app/agents/config.py`:

```python
# Example agent configuration
AGENTS_CONFIG = {
    "research_agent": {
        "role": "Market Research Specialist",
        "goal": "Conduct comprehensive market research",
        "backstory": "Expert in market analysis and competitor research",
        "tools": ["web_search", "data_analysis"]
    },
    "content_agent": {
        "role": "Content Creator",
        "goal": "Create engaging and informative content",
        "backstory": "Professional content writer and storyteller",
        "tools": ["writing_tools", "image_generation"]
    }
}
```

## Running the Application

### Development Mode

```bash
# Start all services in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Or run individual services
docker-compose up backend frontend database redis
```

### Production Mode

```bash
# Build and start production containers
docker-compose -f docker-compose.prod.yml up -d

# Scale services if needed
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

### Individual Service Management

```bash
# Backend only
cd app
uvicorn main:app --reload --port 8000

# Frontend only
cd frontend
npm run dev

# Database only
docker-compose up database -d
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000

# Database connection
docker-compose exec database pg_isready -U user -d agentic_startup

# Redis connection
docker-compose exec redis redis-cli ping
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Backend tests
pytest tests/

# Frontend tests
npm test

# End-to-end tests
npm run test:e2e

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=app --cov-report=html

# View coverage
open htmlcov/index.html
```

### Test Configuration

Tests are configured via:
- `pytest.ini` - Python test settings
- `jest.config.js` - JavaScript test settings
- `.coveragerc` - Coverage reporting settings

## Deployment

### Local Deployment

```bash
# Build and deploy locally
make build
make deploy-local
```

### Cloud Deployment

The repository includes deployment templates for:

- **AWS**: ECS, EKS, Lambda configurations
- **Google Cloud**: GKE, Cloud Run configurations
- **Azure**: AKS, Container Instances configurations
- **DigitalOcean**: App Platform configurations

See `docs/guides/deployment.md` for detailed deployment instructions.

### CI/CD Pipeline

GitHub Actions workflows are included for:
- Automated testing on PR and push
- Security scanning and vulnerability assessment
- Docker image building and publishing
- Automated deployment to staging/production

## Monitoring and Observability

### Accessing Monitoring Tools

- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application Metrics**: http://localhost:8000/metrics

### Key Metrics to Monitor

- **Application Performance**: Response times, throughput, error rates
- **AI Agents**: Task completion rates, processing times, success rates
- **Infrastructure**: CPU, memory, disk usage, network traffic
- **Business Metrics**: User engagement, conversion rates, revenue

### Alerting

Configure alerts in `monitoring/alert_rules.yml`:

```yaml
- alert: HighErrorRate
  expr: http_requests_total{status=~"5.."} / http_requests_total > 0.1
  for: 5m
  annotations:
    summary: High error rate detected
```

### Log Management

Logs are structured and include:
- Application logs via Python logging
- Access logs from FastAPI
- Agent activity logs from CrewAI
- Infrastructure logs from Docker

## Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Follow coding standards and write tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**: Describe your changes and link any issues

### Code Standards

- **Python**: Follow PEP 8, use type hints, document functions
- **JavaScript/TypeScript**: Follow ESLint configuration
- **Documentation**: Update docs for any user-facing changes
- **Testing**: Maintain test coverage above 80%

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check database status
docker-compose ps database

# Check database logs
docker-compose logs database

# Reset database
docker-compose down database
docker volume rm agentic-startup-studio-boilerplate_postgres_data
docker-compose up database -d
```

#### Port Conflicts

```bash
# Check what's using a port
lsof -i :8000

# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Change from 8000:8000
```

#### Memory Issues

```bash
# Increase Docker memory allocation (Docker Desktop)
# Or add memory limits to docker-compose.yml
mem_limit: 2g
```

#### AI API Issues

```bash
# Check API key configuration
echo $OPENAI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Debug Mode

Enable debug mode for detailed error information:

```bash
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

### Performance Issues

```bash
# Monitor resource usage
docker stats

# Profile the application
python -m cProfile -o profile.stats app/main.py

# Analyze database performance
docker-compose exec database psql -U user -d agentic_startup -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC LIMIT 10;"
```

## Next Steps

### Immediate Tasks

1. **Customize AI Agents**: Modify agents in `app/agents/` for your use case
2. **Design Database Schema**: Update models in `app/models/`
3. **Create Frontend Components**: Build your UI in `frontend/src/components/`
4. **Configure Integrations**: Set up external APIs and services
5. **Write Tests**: Add test cases for your specific functionality

### Advanced Configuration

1. **Multi-Environment Setup**: Configure staging, production environments
2. **Scaling**: Set up horizontal scaling for backend services
3. **Security Hardening**: Implement additional security measures
4. **Performance Optimization**: Profile and optimize critical paths
5. **Custom Monitoring**: Add business-specific metrics and dashboards

### Business Development

1. **Market Research**: Use AI agents for competitive analysis
2. **Content Strategy**: Leverage content generation agents
3. **Customer Support**: Implement AI-powered support agents
4. **Analytics**: Build custom analytics and reporting
5. **Growth Hacking**: Automate marketing and growth initiatives

### Learning Resources

- **CrewAI Documentation**: https://docs.crewai.com/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/
- **Docker Documentation**: https://docs.docker.com/
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/

### Community and Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Contributing Guide**: See `CONTRIBUTING.md` for contribution guidelines
- **Security**: Report security issues via `SECURITY.md`

---

## Quick Reference Commands

```bash
# Development
make dev                    # Start development environment
make test                   # Run all tests
make lint                   # Run linting
make format                # Format code

# Docker
docker-compose up -d        # Start all services
docker-compose down         # Stop all services
docker-compose logs -f      # View logs
docker-compose ps           # Check service status

# Database
make db-migrate            # Run database migrations
make db-reset              # Reset database
make db-backup             # Backup database
make db-restore            # Restore database

# Deployment
make build                 # Build all containers
make deploy-local          # Deploy locally
make deploy-staging        # Deploy to staging
make deploy-production     # Deploy to production

# Monitoring
make monitor               # Open monitoring dashboard
make logs                  # View application logs
make metrics               # View metrics
make alerts                # Check alerts
```

---

**Welcome to your agentic startup journey!** 🚀

This boilerplate provides a solid foundation, but the real magic happens when you customize it for your specific vision. Start building, iterating, and scaling your AI-powered startup today.

For questions, issues, or contributions, please see our [contributing guidelines](CONTRIBUTING.md) or open an issue on GitHub.

Happy building! 🎉