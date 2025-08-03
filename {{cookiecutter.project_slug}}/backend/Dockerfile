# Multi-stage Dockerfile for {{cookiecutter.project_name}} Backend
# Optimized for development and production environments

# ============================================================================
# Base Python image with security updates
# ============================================================================
FROM python:{{cookiecutter.python_version}}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# ============================================================================
# Development stage
# ============================================================================
FROM base as development

# Install Poetry for dependency management
RUN pip install poetry==$POETRY_VERSION

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Development command with hot reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ============================================================================
# Production dependencies stage
# ============================================================================
FROM base as deps

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install production dependencies only
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-interaction --no-ansi

# ============================================================================
# Production stage
# ============================================================================
FROM base as production

# Copy installed dependencies from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=deps /usr/local/bin/ /usr/local/bin/

# Set work directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY docker-entrypoint.sh ./

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create directories for logs and uploads
RUN mkdir -p /app/logs /app/uploads \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ============================================================================
# Testing stage
# ============================================================================
FROM development as testing

# Install test dependencies
RUN poetry install --with=test --no-interaction --no-ansi

# Copy test files
COPY tests/ ./tests/

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=app", "--cov-report=term-missing"]

# ============================================================================
# Build info and labels
# ============================================================================
FROM production

# Add metadata labels
LABEL maintainer="{{cookiecutter.author_name}} <{{cookiecutter.author_email}}>" \
      version="{{cookiecutter.version}}" \
      description="{{cookiecutter.description}}" \
      org.opencontainers.image.title="{{cookiecutter.project_name}} Backend" \
      org.opencontainers.image.description="{{cookiecutter.description}}" \
      org.opencontainers.image.version="{{cookiecutter.version}}" \
      org.opencontainers.image.vendor="{{cookiecutter.author_name}}" \
      org.opencontainers.image.source="https://github.com/{{cookiecutter.github_username}}/{{cookiecutter.project_slug}}" \
      org.opencontainers.image.licenses="{{cookiecutter.license}}"