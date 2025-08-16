#!/bin/bash
# Optimized Production Deployment Script for Quantum Task Planner

set -e  # Exit on any error

# Configuration
APP_NAME="quantum-task-planner"
VERSION="v3.0.0"
HEALTH_CHECK_TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    # Check Kubernetes tools if needed
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            missing_tools+=("kubectl")
        fi
        if ! command -v helm &> /dev/null; then
            missing_tools+=("helm")
        fi
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Load environment variables
load_environment() {
    log_info "Loading environment configuration..."
    
    local env_file="${PROJECT_ROOT}/.env.${ENVIRONMENT}"
    
    if [[ ! -f "$env_file" ]]; then
        log_error "Environment file not found: $env_file"
        log_info "Creating default environment file..."
        
        cat > "$env_file" << EOF
# Quantum Task Planner Production Configuration
POSTGRES_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16)
RABBITMQ_PASSWORD=$(openssl rand -base64 32)
BACKUP_S3_BUCKET=quantum-tasks-backups
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
DOMAIN_NAME=quantum-tasks.local
SSL_EMAIL=admin@quantum-tasks.local
EOF
        
        log_warning "Default environment file created. Please update with your values: $env_file"
        log_warning "Passwords have been auto-generated for security."
    fi
    
    set -a  # Automatically export all variables
    source "$env_file"
    set +a
    
    log_success "Environment loaded from $env_file"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Build optimized production image
    docker build -f Dockerfile.optimized -t ${APP_NAME}:${VERSION} .
    docker tag ${APP_NAME}:${VERSION} ${APP_NAME}:latest
    
    # Verify image was built successfully
    if docker images | grep -q "${APP_NAME}.*${VERSION}"; then
        log_success "Docker image built successfully: ${APP_NAME}:${VERSION}"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running comprehensive test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Create test environment
    docker run --rm \
        -v "$PROJECT_ROOT:/app" \
        -w /app \
        "quantum-task-planner:${VERSION}" \
        python -m pytest tests/ -v --cov=quantum_task_planner --cov-fail-under=85
    
    log_success "All tests passed"
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p logs/{nginx,application} data database/backups monitoring/grafana/dashboards
    
    # Generate SSL certificates if they don't exist
    if [[ ! -f "nginx/ssl/quantum-tasks.crt" ]]; then
        log_info "Generating self-signed SSL certificates..."
        mkdir -p nginx/ssl
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/quantum-tasks.key \
            -out nginx/ssl/quantum-tasks.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=${DOMAIN_NAME:-quantum-tasks.local}"
    fi
    
    # Deploy services
    docker-compose -f docker-compose.production.yml up -d --remove-orphans
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Run health checks
    check_health_docker_compose
    
    log_success "Docker Compose deployment completed"
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log_info "Deploying with Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Apply namespace first
    kubectl apply -f k8s/namespace.yaml
    
    # Apply secrets (ensure they're properly configured)
    log_warning "Please ensure k8s/secrets.yaml has been configured with real values"
    kubectl apply -f k8s/secrets.yaml
    
    # Apply configurations
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/pvc.yaml
    kubectl apply -f k8s/rbac.yaml
    
    # Deploy applications
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    
    # Wait for rollout
    kubectl rollout status deployment/quantum-api -n quantum-tasks --timeout=300s
    kubectl rollout status deployment/postgres -n quantum-tasks --timeout=300s
    kubectl rollout status deployment/redis -n quantum-tasks --timeout=300s
    kubectl rollout status deployment/nginx -n quantum-tasks --timeout=300s
    
    # Run health checks
    check_health_kubernetes
    
    log_success "Kubernetes deployment completed"
}

# Health checks for Docker Compose
check_health_docker_compose() {
    log_info "Running health checks..."
    
    local services=("quantum-api" "postgres" "redis" "nginx")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if docker-compose -f docker-compose.production.yml ps "$service" | grep -q "Up (healthy)"; then
            log_success "$service is healthy"
        else
            log_error "$service is not healthy"
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "Failed health checks: ${failed_services[*]}"
        show_logs_docker_compose
        exit 1
    fi
    
    # Test API endpoint
    if curl -f -s "http://localhost/api/v1/health" > /dev/null; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        exit 1
    fi
}

# Health checks for Kubernetes
check_health_kubernetes() {
    log_info "Running Kubernetes health checks..."
    
    # Check pod status
    if kubectl get pods -n quantum-tasks | grep -E "(Running|Ready)"; then
        log_success "All pods are running"
    else
        log_error "Some pods are not running"
        kubectl get pods -n quantum-tasks
        exit 1
    fi
    
    # Check services
    local api_service=$(kubectl get service quantum-api-service -n quantum-tasks -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -n "$api_service" ]] && curl -f -s "http://$api_service/api/v1/health" > /dev/null; then
        log_success "API service health check passed"
    else
        log_warning "API service health check failed or still provisioning"
    fi
}

# Show logs for troubleshooting
show_logs_docker_compose() {
    log_info "Recent logs for troubleshooting:"
    docker-compose -f docker-compose.production.yml logs --tail=50
}

# Show logs for Kubernetes
show_logs_kubernetes() {
    log_info "Recent logs for troubleshooting:"
    kubectl logs -n quantum-tasks -l app=quantum-api --tail=50
}

# Backup data
backup_data() {
    log_info "Creating data backup..."
    
    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        # Backup database
        docker-compose -f docker-compose.production.yml exec -T postgres \
            pg_dump -U quantum_user quantum_tasks | gzip > "$backup_dir/database.sql.gz"
        
        # Backup Redis
        docker-compose -f docker-compose.production.yml exec -T redis \
            redis-cli --rdb - | gzip > "$backup_dir/redis.rdb.gz"
        
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        # Backup database from Kubernetes
        kubectl exec -n quantum-tasks deployment/postgres -- \
            pg_dump -U quantum_user quantum_tasks | gzip > "$backup_dir/database.sql.gz"
    fi
    
    log_success "Backup created in $backup_dir"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
        docker-compose -f docker-compose.production.yml down
        log_success "Docker Compose deployment rolled back"
        
    elif [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        kubectl rollout undo deployment/quantum-api -n quantum-tasks
        kubectl rollout undo deployment/nginx -n quantum-tasks
        log_success "Kubernetes deployment rolled back"
    fi
}

# Monitoring setup
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Configure Prometheus
    if [[ ! -f "monitoring/prometheus.yml" ]]; then
        cp monitoring/prometheus-production.yml monitoring/prometheus.yml
    fi
    
    # Import Grafana dashboards
    local dashboard_dir="monitoring/grafana/dashboards"
    mkdir -p "$dashboard_dir"
    
    # Create sample dashboard
    cat > "$dashboard_dir/quantum-overview.json" << 'EOF'
{
  "dashboard": {
    "title": "Quantum Task Planner Overview",
    "panels": [
      {
        "title": "Active Tasks",
        "type": "stat",
        "targets": [{"expr": "quantum_tasks_active"}]
      },
      {
        "title": "Average Coherence",
        "type": "stat",
        "targets": [{"expr": "quantum_average_coherence"}]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [{"expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"}]
      }
    ]
  }
}
EOF
    
    log_success "Monitoring configuration completed"
}

# Main deployment function
main() {
    log_info "Starting Quantum Task Planner deployment..."
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    # Trap errors and provide cleanup
    trap 'log_error "Deployment failed! Check logs above."; exit 1' ERR
    
    check_prerequisites
    load_environment
    build_images
    
    # Skip tests in CI/CD environments if specified
    if [[ "${SKIP_TESTS:-false}" != "true" ]]; then
        run_tests
    fi
    
    setup_monitoring
    
    case "$DEPLOYMENT_TYPE" in
        "docker-compose")
            deploy_docker_compose
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        "swarm")
            log_error "Docker Swarm deployment not implemented yet"
            exit 1
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    log_success "🚀 Quantum Task Planner deployed successfully!"
    log_info "Dashboard: http://localhost:3000 (Grafana)"
    log_info "API Documentation: http://localhost/docs"
    log_info "Health Check: http://localhost/api/v1/health"
}

# Handle command line arguments
case "${1:-}" in
    "docker-compose"|"kubernetes"|"swarm")
        main
        ;;
    "backup")
        load_environment
        backup_data
        ;;
    "rollback")
        load_environment
        rollback_deployment
        ;;
    "logs")
        if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
            show_logs_docker_compose
        else
            show_logs_kubernetes
        fi
        ;;
    "health")
        load_environment
        if [[ "$DEPLOYMENT_TYPE" == "docker-compose" ]]; then
            check_health_docker_compose
        else
            check_health_kubernetes
        fi
        ;;
    *)
        echo "Usage: $0 {docker-compose|kubernetes|swarm} [environment] [version]"
        echo "       $0 {backup|rollback|logs|health}"
        echo ""
        echo "Examples:"
        echo "  $0 docker-compose production v2.0.0"
        echo "  $0 kubernetes staging latest"
        echo "  $0 backup"
        echo "  $0 rollback"
        exit 1
        ;;
esac