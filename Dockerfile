# Multi-stage Dockerfile for Agentic Startup Studio
# Optimized for production with security scanning and minimal attack surface

# =============================================================================
# Build stage for Python dependencies
# =============================================================================
FROM python:3.11-slim as python-builder

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Install development dependencies only in development
ARG BUILD_ENV=production
RUN if [ "$BUILD_ENV" = "development" ]; then \
    pip install --no-cache-dir --user -r requirements-dev.txt; \
    fi

# =============================================================================
# Build stage for Node.js dependencies (for frontend)
# =============================================================================
FROM node:18-alpine as node-builder

# Set work directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY frontend/package*.json ./frontend/

# Install dependencies
RUN npm ci --only=production && \
    cd frontend && npm ci --only=production

# Copy frontend source
COPY frontend/ ./frontend/

# Build frontend
RUN cd frontend && npm run build

# =============================================================================
# Production runtime stage
# =============================================================================
FROM python:3.11-slim as production

# Security: Install security updates and minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=python-builder /home/appuser/.local /home/appuser/.local

# Copy built frontend from node builder
COPY --from=node-builder /app/frontend/build ./static/

# Copy application code
COPY --chown=appuser:appuser . .

# Security: Set proper permissions
RUN chmod -R 755 /app && \
    chmod +x /app/docker-entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    USER=appuser \
    HOME=/home/appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security: Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Use dumb-init to handle signals properly
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Default command
CMD ["./docker-entrypoint.sh"]

# =============================================================================
# Development stage
# =============================================================================
FROM production as development

# Switch back to root for installing dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy development dependencies from builder
COPY --from=python-builder /home/appuser/.local /home/appuser/.local

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    debugpy

# Create development directories
RUN mkdir -p /app/logs /app/uploads /app/tmp && \
    chown -R appuser:appuser /app

# Switch back to appuser
USER appuser

# Override command for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# =============================================================================
# Testing stage
# =============================================================================
FROM development as testing

USER root

# Install testing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright browsers
USER appuser
RUN playwright install --with-deps chromium

# Set testing environment variables
ENV TESTING=true \
    PYTEST_ADDOPTS="--tb=short --strict-markers" \
    COVERAGE_PROCESS_START=.coveragerc

# Command for running tests
CMD ["python", "-m", "pytest", "--cov=.", "--cov-report=html", "--cov-report=term", "-v"]

# =============================================================================
# Security scanning stage
# =============================================================================
FROM python:3.11-slim as security-scanner

# Install security scanning tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    semgrep

WORKDIR /app
COPY . .

# Run security scans
RUN bandit -r . -f json -o bandit-report.json || true && \
    safety check --json --output safety-report.json || true && \
    semgrep --config=auto --json --output=semgrep-report.json . || true

# =============================================================================
# Metadata and labels
# =============================================================================
LABEL maintainer="Daniel Schmidt <daniel@terragon.ai>" \
      version="0.2.0" \
      description="Agentic Startup Studio Boilerplate" \
      org.opencontainers.image.title="agentic-startup-studio" \
      org.opencontainers.image.description="A comprehensive boilerplate for building agentic startups" \
      org.opencontainers.image.version="0.2.0" \
      org.opencontainers.image.authors="Daniel Schmidt <daniel@terragon.ai>" \
      org.opencontainers.image.url="https://github.com/danieleschmidt/agentic-startup-studio-boilerplate" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/agentic-startup-studio-boilerplate" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.created="2025-07-27T12:00:00Z" \
      security.scan="enabled" \
      security.non-root="true"