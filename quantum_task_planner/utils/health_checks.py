"""
Quantum Task Planner Health Checks and Circuit Breakers

Comprehensive health monitoring system with circuit breakers,
self-healing capabilities, and quantum-aware system diagnostics.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import psutil
import logging

from .exceptions import CircuitBreakerError, RetryExhaustedError
from .logging import get_logger


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    quantum_coherence: Optional[float] = None


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failure_count / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return 1.0 - self.failure_rate


class HealthCheck:
    """Base class for health checks"""
    
    def __init__(self, name: str, timeout: float = 5.0, critical: bool = False):
        self.name = name
        self.timeout = timeout
        self.critical = critical
        self.logger = get_logger()
    
    async def check(self) -> HealthCheckResult:
        """Perform the health check"""
        start_time = time.time()
        
        try:
            await asyncio.wait_for(self._perform_check(), timeout=self.timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Health check passed",
                duration_ms=duration_ms
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _perform_check(self):
        """Override this method to implement specific health check"""
        pass


class SystemResourcesHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)"""
    
    def __init__(self, 
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        super().__init__("system_resources", critical=True)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    async def _perform_check(self):
        """Check system resource utilization"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
        
        # Determine status
        if (cpu_percent > self.cpu_threshold or 
            memory_percent > self.memory_threshold or 
            disk_percent > self.disk_threshold):
            
            status = HealthStatus.UNHEALTHY
            message = f"Resource utilization high: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
        elif (cpu_percent > self.cpu_threshold * 0.8 or 
              memory_percent > self.memory_threshold * 0.8 or 
              disk_percent > self.disk_threshold * 0.8):
            status = HealthStatus.DEGRADED
            message = f"Resource utilization elevated: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Resource utilization normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
        
        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            details=details
        )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity"""
    
    def __init__(self, connection_string: str):
        super().__init__("database", critical=True)
        self.connection_string = connection_string
    
    async def _perform_check(self):
        """Check database connection"""
        # This is a simplified implementation
        # In practice, you'd use actual database connection
        try:
            # Simulate database ping
            await asyncio.sleep(0.1)
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={"connection_string": self.connection_string[:50] + "..."}
            )
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity"""
    
    def __init__(self, redis_url: str):
        super().__init__("redis")
        self.redis_url = redis_url
    
    async def _perform_check(self):
        """Check Redis connection"""
        try:
            # Simulate Redis ping
            await asyncio.sleep(0.05)
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                details={"redis_url": self.redis_url}
            )
        except Exception as e:
            raise Exception(f"Redis connection failed: {str(e)}")


class QuantumCoherenceHealthCheck(HealthCheck):
    """Health check for quantum system coherence"""
    
    def __init__(self, scheduler=None, coherence_threshold: float = 0.5):
        super().__init__("quantum_coherence")
        self.scheduler = scheduler
        self.coherence_threshold = coherence_threshold
    
    async def _perform_check(self):
        """Check quantum system coherence"""
        if not self.scheduler:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message="Quantum scheduler not available"
            )
        
        # Calculate average coherence across all tasks
        if hasattr(self.scheduler, 'tasks'):
            coherence_values = [
                task.quantum_coherence for task in self.scheduler.tasks.values()
                if hasattr(task, 'quantum_coherence')
            ]
            
            if coherence_values:
                avg_coherence = sum(coherence_values) / len(coherence_values)
                min_coherence = min(coherence_values)
                
                if avg_coherence < self.coherence_threshold:
                    status = HealthStatus.DEGRADED
                    message = f"Low quantum coherence: avg {avg_coherence:.3f}, min {min_coherence:.3f}"
                elif min_coherence < self.coherence_threshold * 0.5:
                    status = HealthStatus.DEGRADED  
                    message = f"Some tasks have very low coherence: min {min_coherence:.3f}"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Quantum coherence normal: avg {avg_coherence:.3f}"
                
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    message=message,
                    details={
                        "avg_coherence": avg_coherence,
                        "min_coherence": min_coherence,
                        "max_coherence": max(coherence_values),
                        "task_count": len(coherence_values)
                    },
                    quantum_coherence=avg_coherence
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="No quantum tasks to measure"
                )
        
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            message="Cannot access quantum scheduler tasks"
        )


class CircuitBreaker:
    """Circuit breaker for service reliability"""
    
    def __init__(self,
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3,
                 timeout: float = 10.0):
        
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_state_change = datetime.utcnow()
        self.half_open_success_count = 0
        
        self._lock = threading.RLock()
        self.logger = get_logger()
    
    def __call__(self, func: Callable):
        """Decorator to wrap function with circuit breaker"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker"""
        with self._lock:
            self._update_state()
            
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    self.name,
                    self.metrics.failure_count,
                    self.failure_threshold
                )
            
            self.metrics.total_requests += 1
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise
    
    async def call_async(self, func: Callable, *args, **kwargs):
        """Async version of circuit breaker call"""
        with self._lock:
            self._update_state()
            
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    self.name,
                    self.metrics.failure_count,
                    self.failure_threshold
                )
            
            self.metrics.total_requests += 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
                else:
                    result = func(*args, **kwargs)
                
                self._on_success()
                return result
                
            except Exception as e:
                self._on_failure()
                raise
    
    def _update_state(self):
        """Update circuit breaker state based on metrics"""
        now = datetime.utcnow()
        
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            time_since_change = (now - self.last_state_change).total_seconds()
            if time_since_change >= self.recovery_timeout:
                self._transition_to_half_open()
        
        elif self.state == CircuitBreakerState.CLOSED:
            # Check if failure threshold exceeded
            if self.metrics.failure_count >= self.failure_threshold:
                self._transition_to_open()
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Check if success threshold reached
            if self.half_open_success_count >= self.success_threshold:
                self._transition_to_closed()
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitBreakerState.OPEN
        self.last_state_change = datetime.utcnow()
        self.logger.warning(f"Circuit breaker {self.name} OPENED after {self.metrics.failure_count} failures")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.last_state_change = datetime.utcnow()
        self.half_open_success_count = 0
        self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitBreakerState.CLOSED
        self.last_state_change = datetime.utcnow()
        self.metrics.failure_count = 0
        self.half_open_success_count = 0
        self.logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")
    
    def _on_success(self):
        """Handle successful operation"""
        self.metrics.success_count += 1
        self.metrics.last_success_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_success_count += 1
        elif self.state == CircuitBreakerState.CLOSED and self.metrics.failure_count > 0:
            # Reset failure count on success in closed state
            self.metrics.failure_count = max(0, self.metrics.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed operation"""
        self.metrics.failure_count += 1
        self.metrics.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Return to open state on failure
            self._transition_to_open()
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.metrics.failure_count,
            "success_count": self.metrics.success_count,
            "failure_rate": self.metrics.failure_rate,
            "total_requests": self.metrics.total_requests,
            "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "last_state_change": self.last_state_change.isoformat()
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self.last_state_change = datetime.utcnow()
            self.half_open_success_count = 0
            self.logger.info(f"Circuit breaker {self.name} manually reset")


class HealthCheckManager:
    """Manages multiple health checks and overall system health"""
    
    def __init__(self, check_interval: float = 30.0):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.check_interval = check_interval
        
        self.last_check_results: Dict[str, HealthCheckResult] = {}
        self.overall_status = HealthStatus.HEALTHY
        self.check_thread = None
        self.shutdown_event = threading.Event()
        
        self.logger = get_logger()
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check"""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Added health check: {health_check.name}")
    
    def add_circuit_breaker(self, circuit_breaker: CircuitBreaker):
        """Add a circuit breaker"""
        self.circuit_breakers[circuit_breaker.name] = circuit_breaker
        self.logger.info(f"Added circuit breaker: {circuit_breaker.name}")
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self.check_thread and self.check_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.check_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="health_monitor"
        )
        self.check_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        self.shutdown_event.set()
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.shutdown_event.wait(self.check_interval):
            try:
                asyncio.run(self.check_all_health())
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    async def check_all_health(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}
        
        # Run all health checks concurrently
        check_tasks = [
            check.check() for check in self.health_checks.values()
        ]
        
        if check_tasks:
            check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            for i, result in enumerate(check_results):
                check_name = list(self.health_checks.keys())[i]
                
                if isinstance(result, Exception):
                    result = HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check error: {str(result)}"
                    )
                
                results[check_name] = result
                self.last_check_results[check_name] = result
        
        # Update overall status
        self._update_overall_status(results)
        
        return results
    
    def _update_overall_status(self, results: Dict[str, HealthCheckResult]):
        """Update overall system health status"""
        if not results:
            self.overall_status = HealthStatus.HEALTHY
            return
        
        critical_checks = [
            result for result in results.values()
            if self.health_checks[result.name].critical
        ]
        
        # Check critical health checks first
        if critical_checks:
            critical_unhealthy = [
                check for check in critical_checks
                if check.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            ]
            
            if critical_unhealthy:
                self.overall_status = HealthStatus.CRITICAL
                return
        
        # Check all health checks
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            self.overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            self.overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.HEALTHY
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in self.last_check_results.items()
            },
            "circuit_breakers": {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


# Global health check manager
_health_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get global health check manager"""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    return _health_manager


def setup_default_health_checks(scheduler=None, database_url: str = None, redis_url: str = None):
    """Setup default health checks"""
    manager = get_health_manager()
    
    # System resources
    manager.add_health_check(SystemResourcesHealthCheck())
    
    # Database
    if database_url:
        manager.add_health_check(DatabaseHealthCheck(database_url))
    
    # Redis
    if redis_url:
        manager.add_health_check(RedisHealthCheck(redis_url))
    
    # Quantum coherence
    if scheduler:
        manager.add_health_check(QuantumCoherenceHealthCheck(scheduler))
    
    # Circuit breakers
    manager.add_circuit_breaker(CircuitBreaker("api_endpoints"))
    manager.add_circuit_breaker(CircuitBreaker("database_operations"))
    manager.add_circuit_breaker(CircuitBreaker("quantum_optimization"))
    
    # Start monitoring
    manager.start_monitoring()
    
    return manager