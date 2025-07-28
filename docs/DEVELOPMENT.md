# Development Setup

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd <project-name>

# 2. Install dependencies
pip install -r requirements-dev.txt
npm install  # if using Node.js components

# 3. Run tests
pytest
npm test  # if applicable
```

## Development Tools

- **Testing**: `pytest` and `playwright` for E2E
- **Linting**: `ruff` for Python, `eslint` for JS
- **Type Checking**: `mypy` for static analysis
- **Security**: `pip-audit` and `safety` for vulnerability scanning

## Local Environment

- **Python**: 3.9+ required
- **Docker**: For containerized development
- **Database**: SQLite for local, PostgreSQL for production

## Useful Commands

```bash
make test          # Run all tests
make lint          # Code quality checks
make security      # Security scanning
make docker-dev    # Start development environment
```

## Documentation References

- [Contributing Guidelines](../CONTRIBUTING.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Security Policy](../SECURITY.md)
- [Project Requirements](REQUIREMENTS.md)