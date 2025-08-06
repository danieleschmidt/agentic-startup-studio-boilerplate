# Multi-stage Dockerfile for Quantum Task Planner
# Optimized for production with security scanning and minimal attack surface

# =============================================================================
# Build stage for Python dependencies
# =============================================================================
FROM python:3.11-slim as python-builder

# Security: Create non-root user
RUN groupadd --gid 1000 quantum && \
    useradd --uid 1000 --gid quantum --shell /bin/bash --create-home quantum

# Set work directory
WORKDIR /app

# Install system dependencies for building Python packages including quantum computing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt ./

# Install Python dependencies for quantum computing
RUN pip install --no-cache-dir --user -r requirements.txt

# Install additional quantum computing dependencies
RUN pip install --no-cache-dir --user \
    uvicorn[standard] \
    gunicorn \
    psycopg2-binary \
    redis \
    prometheus-client \
    structlog

# =============================================================================
# Skip Node.js stage - Quantum Task Planner is API-only
# =============================================================================

# =============================================================================
# Production runtime stage
# =============================================================================
FROM python:3.11-slim as production

# Security: Install security updates and minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    postgresql-client \
    redis-tools \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd --gid 1000 quantum && \
    useradd --uid 1000 --gid quantum --shell /bin/bash --create-home quantum

# Set work directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=python-builder /home/quantum/.local /home/quantum/.local

# Copy application code
COPY --chown=quantum:quantum . .

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/cache /app/data && \
    chown -R quantum:quantum /app && \
    chmod -R 755 /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/quantum/.local/bin:$PATH" \
    USER=quantum \
    HOME=/home/quantum \
    QUANTUM_ENV=production \
    PORT=8000

# Health check for Quantum Task Planner API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Security: Switch to non-root user
USER quantum

# Expose port
EXPOSE 8000

# Use dumb-init to handle signals properly
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Default command for Quantum Task Planner
CMD ["python", "-m", "uvicorn", "quantum_task_planner.api.quantum_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

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

# Install additional development tools for quantum development
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    debugpy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy

# Create development directories
RUN mkdir -p /app/logs /app/cache /app/data /app/tmp && \
    chown -R quantum:quantum /app

# Switch back to quantum user
USER quantum

# Override command for development with hot reload
CMD ["python", "-m", "uvicorn", "quantum_task_planner.api.quantum_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# =============================================================================
# Testing stage
# =============================================================================
FROM development as testing

USER root

# Install additional testing dependencies for quantum testing
RUN pip install --no-cache-dir \
    httpx \
    pytest-benchmark \
    pytest-mock \
    coverage[toml]

USER quantum

# Set testing environment variables for quantum testing
ENV TESTING=true \
    QUANTUM_TEST_MODE=true \
    PYTEST_ADDOPTS="--tb=short --strict-markers" \
    COVERAGE_PROCESS_START=.coveragerc

# Command for running quantum task planner tests
CMD ["python", "-m", "pytest", "tests/", "--cov=quantum_task_planner", "--cov-report=html", "--cov-report=term", "--cov-report=xml", "-v"]

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
LABEL maintainer="Terragon Labs <team@terragon.ai>" \
      version="2.0.0" \
      description="Quantum Task Planner - Production-ready quantum-inspired task planning system" \
      org.opencontainers.image.title="quantum-task-planner" \
      org.opencontainers.image.description="A production-ready quantum-inspired task planning and optimization system with distributed capabilities" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="Terragon Labs <team@terragon.ai>" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.created="2025-08-06T00:00:00Z" \
      quantum.computing="enabled" \
      quantum.optimization="genetic-algorithms" \
      quantum.scheduler="annealing" \
      quantum.entanglement="bell-states" \
      security.scan="enabled" \
      security.non-root="true" \
      performance.caching="quantum-coherence" \
      distributed.sync="quantum-state"