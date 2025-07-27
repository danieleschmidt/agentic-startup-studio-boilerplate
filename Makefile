# Makefile for Agentic Startup Studio Boilerplate
# Provides convenient commands for development, testing, and deployment

.PHONY: help install dev-up dev-down test lint security build deploy clean

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)Agentic Startup Studio Boilerplate$(RESET)"
	@echo "$(CYAN)=====================================$(RESET)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  make install    # Install dependencies"
	@echo "  make dev-up     # Start development environment"
	@echo "  make test       # Run all tests"
	@echo ""

install: ## Install all dependencies
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	npm install
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

install-dev: ## Install development dependencies only
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	npm install
	pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development dependencies installed$(RESET)"

dev-up: ## Start development environment
	@echo "$(GREEN)Starting development environment...$(RESET)"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✓ Development environment started$(RESET)"
	@echo "$(CYAN)Services available at:$(RESET)"
	@echo "  Frontend:    http://localhost:3000"
	@echo "  API:         http://localhost:8000"
	@echo "  API Docs:    http://localhost:8000/docs"
	@echo "  Database:    http://localhost:8081 (Adminer)"
	@echo "  Monitoring:  http://localhost:9090 (Prometheus)"
	@echo "  Dashboards:  http://localhost:3001 (Grafana)"

dev-down: ## Stop development environment
	@echo "$(YELLOW)Stopping development environment...$(RESET)"
	docker-compose -f docker-compose.dev.yml down
	@echo "$(GREEN)✓ Development environment stopped$(RESET)"

dev-logs: ## Show development environment logs
	docker-compose -f docker-compose.dev.yml logs -f

dev-restart: ## Restart development environment
	@echo "$(YELLOW)Restarting development environment...$(RESET)"
	docker-compose -f docker-compose.dev.yml restart
	@echo "$(GREEN)✓ Development environment restarted$(RESET)"

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(RESET)"
	python -m pytest --cov=. --cov-report=html --cov-report=term -v
	@echo "$(GREEN)✓ All tests completed$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	python -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	python -m pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running end-to-end tests...$(RESET)"
	npx playwright test

test-security: ## Run security tests
	@echo "$(GREEN)Running security tests...$(RESET)"
	python -m pytest tests/security/ -v

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(RESET)"
	python -m pytest tests/performance/ -v

test-coverage: ## Generate test coverage report
	@echo "$(GREEN)Generating coverage report...$(RESET)"
	python -m pytest --cov=. --cov-report=html --cov-report=xml
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(RESET)"

lint: ## Run code linting
	@echo "$(GREEN)Running linting...$(RESET)"
	flake8 .
	black --check .
	isort --check-only .
	mypy .
	@echo "$(GREEN)✓ Linting completed$(RESET)"

lint-fix: ## Fix linting issues automatically
	@echo "$(GREEN)Fixing linting issues...$(RESET)"
	black .
	isort .
	@echo "$(GREEN)✓ Linting issues fixed$(RESET)"

security: ## Run security scans
	@echo "$(GREEN)Running security scans...$(RESET)"
	bandit -r . -f json -o bandit-report.json
	safety check
	@echo "$(GREEN)✓ Security scans completed$(RESET)"

security-full: ## Run comprehensive security analysis
	@echo "$(GREEN)Running comprehensive security analysis...$(RESET)"
	bandit -r .
	safety check
	semgrep --config=auto .
	@echo "$(GREEN)✓ Security analysis completed$(RESET)"

build: ## Build production Docker image
	@echo "$(GREEN)Building production Docker image...$(RESET)"
	docker build -t agentic-startup-studio:latest .
	@echo "$(GREEN)✓ Docker image built$(RESET)"

build-dev: ## Build development Docker image
	@echo "$(GREEN)Building development Docker image...$(RESET)"
	docker build -t agentic-startup-studio:dev --target development .
	@echo "$(GREEN)✓ Development Docker image built$(RESET)"

deploy-prod: ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)✓ Production deployment completed$(RESET)"

deploy-staging: ## Deploy to staging
	@echo "$(GREEN)Deploying to staging...$(RESET)"
	docker-compose -f docker-compose.staging.yml up -d
	@echo "$(GREEN)✓ Staging deployment completed$(RESET)"

docs: ## Generate documentation
	@echo "$(GREEN)Generating documentation...$(RESET)"
	pdoc --html --output-dir docs/ .
	@echo "$(GREEN)✓ Documentation generated: docs/$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8082$(RESET)"
	cd docs && python -m http.server 8082

clean: ## Clean up build artifacts and caches
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	docker system prune -f
	@echo "$(GREEN)✓ Cleanup completed$(RESET)"

clean-volumes: ## Clean up Docker volumes
	@echo "$(YELLOW)Cleaning up Docker volumes...$(RESET)"
	docker volume prune -f
	@echo "$(GREEN)✓ Docker volumes cleaned$(RESET)"

validate: ## Validate configuration and setup
	@echo "$(GREEN)Validating configuration...$(RESET)"
	python -c "import sys; print(f'Python: {sys.version}')"
	docker --version
	docker-compose --version
	npm --version
	@echo "$(GREEN)✓ Configuration validated$(RESET)"

status: ## Show service status
	@echo "$(GREEN)Service Status:$(RESET)"
	docker-compose -f docker-compose.dev.yml ps

logs: ## Show logs from all services
	docker-compose -f docker-compose.dev.yml logs

db-shell: ## Connect to database shell
	docker-compose -f docker-compose.dev.yml exec db psql -U postgres -d agentic_startup_dev

redis-shell: ## Connect to Redis CLI
	docker-compose -f docker-compose.dev.yml exec redis redis-cli

migrations: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(RESET)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations completed$(RESET)"

migration-create: ## Create new database migration
	@read -p "Migration name: " name; \
	alembic revision --autogenerate -m "$$name"

seed-data: ## Seed database with sample data
	@echo "$(GREEN)Seeding database with sample data...$(RESET)"
	python scripts/seed_data.py
	@echo "$(GREEN)✓ Database seeded$(RESET)"

backup-db: ## Backup database
	@echo "$(GREEN)Creating database backup...$(RESET)"
	docker-compose -f docker-compose.dev.yml exec db pg_dump -U postgres agentic_startup_dev > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✓ Database backup created$(RESET)"

load-test: ## Run load testing
	@echo "$(GREEN)Running load tests...$(RESET)"
	python -m pytest tests/performance/test_load_testing.py -v
	@echo "$(GREEN)✓ Load testing completed$(RESET)"

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	python scripts/benchmark.py
	@echo "$(GREEN)✓ Benchmarks completed$(RESET)"

release: ## Create a new release
	@echo "$(GREEN)Creating new release...$(RESET)"
	npm run release
	@echo "$(GREEN)✓ Release created$(RESET)"

release-dry: ## Dry run release process
	@echo "$(GREEN)Running release dry run...$(RESET)"
	npm run release:dry
	@echo "$(GREEN)✓ Release dry run completed$(RESET)"

version: ## Show current version
	@echo "$(GREEN)Current version:$(RESET)"
	@cat package.json | grep '"version"' | head -1 | awk -F: '{ print $$2 }' | sed 's/[",]//g' | tr -d '[[:space:]]'

update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	npm update
	pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U
	@echo "$(GREEN)✓ Dependencies updated$(RESET)"

check-deps: ## Check for dependency vulnerabilities
	@echo "$(GREEN)Checking dependencies for vulnerabilities...$(RESET)"
	npm audit
	safety check
	@echo "$(GREEN)✓ Dependency check completed$(RESET)"

format: ## Format code
	@echo "$(GREEN)Formatting code...$(RESET)"
	black .
	isort .
	@echo "$(GREEN)✓ Code formatted$(RESET)"

typecheck: ## Run type checking
	@echo "$(GREEN)Running type checking...$(RESET)"
	mypy .
	@echo "$(GREEN)✓ Type checking completed$(RESET)"

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit hooks completed$(RESET)"

setup-hooks: ## Setup git hooks
	@echo "$(GREEN)Setting up git hooks...$(RESET)"
	pre-commit install
	@echo "$(GREEN)✓ Git hooks setup completed$(RESET)"

ci: lint test security ## Run CI pipeline locally
	@echo "$(GREEN)✓ CI pipeline completed successfully$(RESET)"

quick-test: ## Run quick tests (fast feedback)
	@echo "$(GREEN)Running quick tests...$(RESET)"
	python -m pytest tests/unit/ -x --tb=short
	@echo "$(GREEN)✓ Quick tests completed$(RESET)"

health-check: ## Check service health
	@echo "$(GREEN)Checking service health...$(RESET)"
	curl -f http://localhost:8000/health || echo "$(RED)API health check failed$(RESET)"
	curl -f http://localhost:3000 || echo "$(RED)Frontend health check failed$(RESET)"
	@echo "$(GREEN)✓ Health check completed$(RESET)"

monitor: ## Show real-time logs
	docker-compose -f docker-compose.dev.yml logs -f app

# Development workflow shortcuts
dev: dev-up ## Alias for dev-up
stop: dev-down ## Alias for dev-down
restart: dev-restart ## Alias for dev-restart
shell: ## Open shell in app container
	docker-compose -f docker-compose.dev.yml exec app bash

# Production shortcuts
prod-up: deploy-prod ## Alias for deploy-prod
prod-down: ## Stop production environment
	docker-compose down

prod-logs: ## Show production logs
	docker-compose logs -f

prod-status: ## Show production status
	docker-compose ps