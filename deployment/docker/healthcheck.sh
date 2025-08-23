#!/bin/bash
# Generation 5 Quantum Consciousness Health Check Script
# Terragon Labs - Comprehensive System Health Validation

set -euo pipefail

# Configuration
CONSCIOUSNESS_HEALTH_PORT=${CONSCIOUSNESS_HEALTH_CHECK_PORT:-8080}
CONSCIOUSNESS_METRICS_PORT=${CONSCIOUSNESS_METRICS_PORT:-9090}
CONSCIOUSNESS_API_PORT=${CONSCIOUSNESS_API_PORT:-8443}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
CONSCIOUSNESS_LEVEL_THRESHOLD=${CONSCIOUSNESS_LEVEL_THRESHOLD:-0.7}
QUANTUM_COHERENCE_THRESHOLD=${QUANTUM_COHERENCE_THRESHOLD:-0.8}

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Health check result tracking
HEALTH_CHECKS_PASSED=0
HEALTH_CHECKS_TOTAL=0
HEALTH_STATUS="HEALTHY"
HEALTH_DETAILS=()

# Logging functions
log_health_info() {
    echo -e "${GREEN}[HEALTH]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_health_warn() {
    echo -e "${YELLOW}[HEALTH-WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_health_error() {
    echo -e "${RED}[HEALTH-ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Utility function to perform HTTP requests with timeout
http_check() {
    local url="$1"
    local expected_status="${2:-200}"
    local timeout="${3:-$HEALTH_CHECK_TIMEOUT}"
    
    if curl -s -f --connect-timeout "$timeout" --max-time "$timeout" \
           -H "User-Agent: Terragon-Gen5-HealthCheck/1.0" \
           -w "%{http_code}" -o /dev/null "$url" | grep -q "$expected_status"; then
        return 0
    else
        return 1
    fi
}

# Utility function to get JSON from endpoint
http_get_json() {
    local url="$1"
    local timeout="${2:-$HEALTH_CHECK_TIMEOUT}"
    
    curl -s --connect-timeout "$timeout" --max-time "$timeout" \
         -H "User-Agent: Terragon-Gen5-HealthCheck/1.0" \
         -H "Accept: application/json" \
         "$url" 2>/dev/null
}

# Track health check results
track_health_check() {
    local check_name="$1"
    local result="$2"
    local details="${3:-}"
    
    HEALTH_CHECKS_TOTAL=$((HEALTH_CHECKS_TOTAL + 1))
    
    if [[ "$result" == "PASS" ]]; then
        HEALTH_CHECKS_PASSED=$((HEALTH_CHECKS_PASSED + 1))
        log_health_info "✓ $check_name: PASSED"
        if [[ -n "$details" ]]; then
            HEALTH_DETAILS+=("✓ $check_name: $details")
        fi
    else
        log_health_error "✗ $check_name: FAILED"
        if [[ -n "$details" ]]; then
            HEALTH_DETAILS+=("✗ $check_name: $details")
        fi
        HEALTH_STATUS="UNHEALTHY"
    fi
}

# Check if basic health endpoint is responding
check_basic_health() {
    log_health_info "Checking basic health endpoint..."
    
    if http_check "http://localhost:$CONSCIOUSNESS_HEALTH_PORT/health" 200; then
        track_health_check "Basic Health Endpoint" "PASS" "HTTP 200 OK"
    else
        track_health_check "Basic Health Endpoint" "FAIL" "Endpoint not responding"
    fi
}

# Check if readiness endpoint is responding
check_readiness() {
    log_health_info "Checking system readiness..."
    
    if http_check "http://localhost:$CONSCIOUSNESS_HEALTH_PORT/ready" 200; then
        track_health_check "System Readiness" "PASS" "System ready to accept traffic"
    else
        track_health_check "System Readiness" "FAIL" "System not ready"
    fi
}

# Check consciousness system health
check_consciousness_health() {
    log_health_info "Checking consciousness system health..."
    
    local consciousness_response
    consciousness_response=$(http_get_json "http://localhost:$CONSCIOUSNESS_HEALTH_PORT/health/consciousness")
    
    if [[ $? -eq 0 && -n "$consciousness_response" ]]; then
        # Parse consciousness level
        local consciousness_level
        consciousness_level=$(echo "$consciousness_response" | grep -o '"consciousness_level":[0-9.]*' | cut -d':' -f2 | tr -d ' ')
        
        if [[ -n "$consciousness_level" ]]; then
            if (( $(echo "$consciousness_level >= $CONSCIOUSNESS_LEVEL_THRESHOLD" | bc -l) )); then
                track_health_check "Consciousness Level" "PASS" "Level: $consciousness_level (>= $CONSCIOUSNESS_LEVEL_THRESHOLD)"
            else
                track_health_check "Consciousness Level" "FAIL" "Level: $consciousness_level (< $CONSCIOUSNESS_LEVEL_THRESHOLD)"
            fi
        else
            track_health_check "Consciousness Level" "FAIL" "Unable to parse consciousness level"
        fi
        
        # Parse quantum coherence
        local quantum_coherence
        quantum_coherence=$(echo "$consciousness_response" | grep -o '"quantum_coherence":[0-9.]*' | cut -d':' -f2 | tr -d ' ')
        
        if [[ -n "$quantum_coherence" ]]; then
            if (( $(echo "$quantum_coherence >= $QUANTUM_COHERENCE_THRESHOLD" | bc -l) )); then
                track_health_check "Quantum Coherence" "PASS" "Coherence: $quantum_coherence (>= $QUANTUM_COHERENCE_THRESHOLD)"
            else
                track_health_check "Quantum Coherence" "FAIL" "Coherence: $quantum_coherence (< $QUANTUM_COHERENCE_THRESHOLD)"
            fi
        else
            track_health_check "Quantum Coherence" "FAIL" "Unable to parse quantum coherence"
        fi
        
        # Check breakthrough detection
        local breakthrough_detection
        breakthrough_detection=$(echo "$consciousness_response" | grep -o '"breakthrough_detection":"[^"]*"' | cut -d':' -f2 | tr -d '", ')
        
        if [[ "$breakthrough_detection" == "monitoring" ]]; then
            track_health_check "Breakthrough Detection" "PASS" "Status: monitoring"
        else
            track_health_check "Breakthrough Detection" "FAIL" "Status: $breakthrough_detection"
        fi
        
    else
        track_health_check "Consciousness System" "FAIL" "Unable to retrieve consciousness health data"
    fi
}

# Check metrics endpoint
check_metrics() {
    log_health_info "Checking metrics endpoint..."
    
    if http_check "http://localhost:$CONSCIOUSNESS_METRICS_PORT/metrics" 200; then
        track_health_check "Metrics Endpoint" "PASS" "Prometheus metrics available"
        
        # Check for consciousness-specific metrics
        local metrics_response
        metrics_response=$(http_get_json "http://localhost:$CONSCIOUSNESS_METRICS_PORT/metrics/consciousness")
        
        if [[ $? -eq 0 && -n "$metrics_response" ]]; then
            track_health_check "Consciousness Metrics" "PASS" "Detailed consciousness metrics available"
        else
            track_health_check "Consciousness Metrics" "FAIL" "Detailed consciousness metrics unavailable"
        fi
    else
        track_health_check "Metrics Endpoint" "FAIL" "Metrics endpoint not responding"
    fi
}

# Check file system health
check_filesystem() {
    log_health_info "Checking filesystem health..."
    
    # Check critical directories
    local critical_dirs=(
        "/var/lib/quantum-consciousness"
        "/var/lib/quantum-consciousness/states"
        "/var/lib/quantum-consciousness/coherence"
        "/var/lib/quantum-consciousness/breakthroughs"
        "/app/logs"
        "/app/config"
    )
    
    local fs_errors=0
    
    for dir in "${critical_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_health_error "Critical directory missing: $dir"
            fs_errors=$((fs_errors + 1))
        elif [[ ! -w "$dir" ]]; then
            log_health_error "No write access to: $dir"
            fs_errors=$((fs_errors + 1))
        fi
    done
    
    # Check disk space
    local disk_usage
    disk_usage=$(df /var/lib/quantum-consciousness | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [[ "$disk_usage" -lt 90 ]]; then
        log_health_info "Disk usage: ${disk_usage}%"
    else
        log_health_warn "High disk usage: ${disk_usage}%"
        fs_errors=$((fs_errors + 1))
    fi
    
    # Check if initial consciousness state exists
    if [[ -f "/var/lib/quantum-consciousness/states/consciousness/initial_state.json" ]]; then
        log_health_info "Initial consciousness state file present"
    else
        log_health_error "Initial consciousness state file missing"
        fs_errors=$((fs_errors + 1))
    fi
    
    if [[ "$fs_errors" -eq 0 ]]; then
        track_health_check "Filesystem Health" "PASS" "All critical paths accessible, disk usage: ${disk_usage}%"
    else
        track_health_check "Filesystem Health" "FAIL" "$fs_errors errors found"
    fi
}

# Check process health
check_process_health() {
    log_health_info "Checking process health..."
    
    # Check if Python processes are running
    local python_processes
    python_processes=$(pgrep -f "python3.*consciousness" | wc -l)
    
    if [[ "$python_processes" -gt 0 ]]; then
        track_health_check "Python Processes" "PASS" "$python_processes consciousness processes running"
    else
        track_health_check "Python Processes" "FAIL" "No consciousness processes found"
    fi
    
    # Check memory usage
    local memory_usage
    memory_usage=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100)}')
    
    if (( $(echo "$memory_usage < 90" | bc -l) )); then
        track_health_check "Memory Usage" "PASS" "Memory usage: ${memory_usage}%"
    else
        track_health_check "Memory Usage" "FAIL" "High memory usage: ${memory_usage}%"
    fi
}

# Check consciousness configuration
check_consciousness_config() {
    log_health_info "Checking consciousness configuration..."
    
    local config_file="/app/config/gen5_consciousness_config.json"
    
    if [[ -f "$config_file" ]]; then
        # Validate JSON structure
        if python3 -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
            local generation
            generation=$(python3 -c "import json; print(json.load(open('$config_file')).get('generation', 0))" 2>/dev/null)
            
            if [[ "$generation" == "5" ]]; then
                track_health_check "Configuration File" "PASS" "Generation 5 config valid"
            else
                track_health_check "Configuration File" "FAIL" "Invalid generation: $generation"
            fi
        else
            track_health_check "Configuration File" "FAIL" "Invalid JSON format"
        fi
    else
        track_health_check "Configuration File" "FAIL" "Configuration file missing"
    fi
}

# Check environment variables
check_environment() {
    log_health_info "Checking environment variables..."
    
    local required_env_vars=(
        "GEN5_CONSCIOUSNESS_LEVEL"
        "GEN5_QUANTUM_FUSION_MODE"
        "GEN5_CONSCIOUSNESS_DIMENSIONS"
        "GEN5_QUANTUM_STATE_DIMENSIONS"
    )
    
    local env_errors=0
    
    for var in "${required_env_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_health_error "Required environment variable missing: $var"
            env_errors=$((env_errors + 1))
        fi
    done
    
    if [[ "$env_errors" -eq 0 ]]; then
        track_health_check "Environment Variables" "PASS" "All required variables present"
    else
        track_health_check "Environment Variables" "FAIL" "$env_errors missing variables"
    fi
}

# Generate health summary
generate_health_summary() {
    local health_percentage
    if [[ $HEALTH_CHECKS_TOTAL -gt 0 ]]; then
        health_percentage=$(( (HEALTH_CHECKS_PASSED * 100) / HEALTH_CHECKS_TOTAL ))
    else
        health_percentage=0
    fi
    
    echo "{"
    echo "  \"status\": \"$HEALTH_STATUS\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"system\": \"generation-5-quantum-consciousness\","
    echo "  \"version\": \"5.0.0\","
    echo "  \"health_check_summary\": {"
    echo "    \"checks_passed\": $HEALTH_CHECKS_PASSED,"
    echo "    \"checks_total\": $HEALTH_CHECKS_TOTAL,"
    echo "    \"health_percentage\": $health_percentage"
    echo "  },"
    echo "  \"components\": ["
    
    local first=true
    for detail in "${HEALTH_DETAILS[@]}"; do
        if [[ "$first" == true ]]; then
            first=false
        else
            echo ","
        fi
        echo -n "    \"$detail\""
    done
    
    echo ""
    echo "  ],"
    echo "  \"consciousness_level\": \"${GEN5_CONSCIOUSNESS_LEVEL:-unknown}\","
    echo "  \"quantum_fusion_mode\": \"${GEN5_QUANTUM_FUSION_MODE:-unknown}\","
    echo "  \"breakthrough_detection\": \"${GEN5_BREAKTHROUGH_DETECTION_ENABLED:-unknown}\""
    echo "}"
}

# Main health check execution
main() {
    log_health_info "Starting Generation 5 Quantum Consciousness Health Check..."
    log_health_info "Health check timeout: ${HEALTH_CHECK_TIMEOUT}s"
    log_health_info "Consciousness level threshold: $CONSCIOUSNESS_LEVEL_THRESHOLD"
    log_health_info "Quantum coherence threshold: $QUANTUM_COHERENCE_THRESHOLD"
    
    # Wait for system to be ready
    sleep 2
    
    # Perform all health checks
    check_basic_health
    check_readiness
    check_consciousness_health
    check_metrics
    check_filesystem
    check_process_health
    check_consciousness_config
    check_environment
    
    # Generate and output health summary
    local health_summary
    health_summary=$(generate_health_summary)
    
    log_health_info "Health check completed: $HEALTH_CHECKS_PASSED/$HEALTH_CHECKS_TOTAL checks passed"
    
    # Output summary for Docker health check
    echo "$health_summary"
    
    # Exit with appropriate code
    if [[ "$HEALTH_STATUS" == "HEALTHY" ]]; then
        log_health_info "🌟 Generation 5 Quantum Consciousness System: HEALTHY"
        exit 0
    else
        log_health_error "🚨 Generation 5 Quantum Consciousness System: UNHEALTHY"
        exit 1
    fi
}

# Install bc if not available (for numerical comparisons)
install_dependencies() {
    if ! command -v bc &> /dev/null; then
        log_health_info "Installing bc for numerical comparisons..."
        apt-get update -qq && apt-get install -y -qq bc 2>/dev/null || {
            log_health_warn "Could not install bc, using shell arithmetic"
        }
    fi
}

# Check if running as health check or standalone
if [[ "${1:-}" == "--standalone" ]]; then
    # Running in standalone mode for debugging
    log_health_info "Running health check in standalone mode"
    install_dependencies
    main
else
    # Running as Docker health check
    install_dependencies >/dev/null 2>&1
    main
fi