#!/bin/bash
set -e

echo "🚀 Starting Agentic Startup Studio..."

# Environment validation
if [ -z "$ENVIRONMENT" ]; then
    export ENVIRONMENT="production"
fi

echo "📍 Environment: $ENVIRONMENT"
echo "🐍 Python version: $(python --version)"
echo "👤 Running as user: $(whoami)"

# Database migration (if applicable)
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "🔄 Running database migrations..."
    alembic upgrade head
fi

# Collect static files (if applicable)
if [ "$COLLECT_STATIC" = "true" ]; then
    echo "📁 Collecting static files..."
    # This would run Django's collectstatic or similar
    # python manage.py collectstatic --noinput
fi

# Pre-flight checks
echo "🔍 Running pre-flight checks..."

# Check database connectivity
if [ ! -z "$DATABASE_URL" ]; then
    echo "  ✓ Database URL configured"
    # Add actual database connectivity check here
    # python -c "import sys; from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); engine.connect(); print('Database connection successful')"
fi

# Check Redis connectivity
if [ ! -z "$REDIS_URL" ]; then
    echo "  ✓ Redis URL configured"
    # Add actual Redis connectivity check here
    # python -c "import redis; r = redis.from_url('$REDIS_URL'); r.ping(); print('Redis connection successful')"
fi

# Check required environment variables
required_vars=("API_SECRET_KEY" "JWT_SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "  ❌ Required environment variable $var is not set"
        exit 1
    else
        echo "  ✓ $var is configured"
    fi
done

# Security checks
echo "🔒 Security checks..."
if [ "$ENVIRONMENT" = "production" ]; then
    if [ "$DEBUG" = "true" ]; then
        echo "  ⚠️  WARNING: DEBUG is enabled in production"
    fi
    
    if [ -z "$SECURE_SSL_REDIRECT" ] || [ "$SECURE_SSL_REDIRECT" != "true" ]; then
        echo "  ⚠️  WARNING: SSL redirect not enabled in production"
    fi
fi

# Performance optimization
echo "⚡ Optimizing performance..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Set production optimizations
    export PYTHONOPTIMIZE=2
    export PYTHONDONTWRITEBYTECODE=1
    
    # Tune worker processes based on CPU cores
    if [ -z "$WEB_CONCURRENCY" ]; then
        cpu_cores=$(nproc)
        export WEB_CONCURRENCY=$((cpu_cores * 2 + 1))
        echo "  ✓ Set WEB_CONCURRENCY to $WEB_CONCURRENCY based on $cpu_cores CPU cores"
    fi
fi

# Create necessary directories
echo "📁 Creating runtime directories..."
mkdir -p logs uploads tmp
chmod 755 logs uploads tmp

# Health check endpoint setup
echo "🏥 Setting up health checks..."
# The application should expose /health endpoint

# Log configuration
echo "📝 Configuring logging..."
if [ "$ENVIRONMENT" = "production" ]; then
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export LOG_FORMAT="${LOG_FORMAT:-json}"
else
    export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
    export LOG_FORMAT="${LOG_FORMAT:-plain}"
fi

echo "  ✓ Log level: $LOG_LEVEL"
echo "  ✓ Log format: $LOG_FORMAT"

# Determine the command to run
if [ "$#" -eq 0 ]; then
    # Default command based on environment
    case "$ENVIRONMENT" in
        "development")
            echo "🔧 Starting development server..."
            exec uvicorn main:app \
                --host 0.0.0.0 \
                --port 8000 \
                --reload \
                --log-level debug \
                --access-log
            ;;
        "testing")
            echo "🧪 Running tests..."
            exec python -m pytest \
                --cov=. \
                --cov-report=html \
                --cov-report=term \
                --cov-report=xml \
                -v
            ;;
        "production"|*)
            echo "🚀 Starting production server..."
            # Use gunicorn for production with multiple workers
            exec gunicorn main:app \
                --bind 0.0.0.0:8000 \
                --workers ${WEB_CONCURRENCY:-4} \
                --worker-class uvicorn.workers.UvicornWorker \
                --max-requests 1000 \
                --max-requests-jitter 50 \
                --preload \
                --log-level $LOG_LEVEL \
                --access-logfile - \
                --error-logfile - \
                --capture-output \
                --enable-stdio-inheritance
            ;;
    esac
else
    # Execute provided command
    echo "🔧 Executing custom command: $@"
    exec "$@"
fi