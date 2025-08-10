"""
Robust Health Monitoring System

Comprehensive health checks with circuit breakers, self-healing, and alerting.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import logging
import psutil
import os
import time

from .exceptions import QuantumTaskPlannerError, CircuitBreakerError


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Health check result with detailed metrics"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    error: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'response_time_ms': self.response_time_ms,
            'error': self.error,
            'recovery_suggestions': self.recovery_suggestions
        }


@dataclass
class CircuitBreaker:
    """Circuit breaker for service protection"""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    
    # Internal state
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = "half_open"
                self.failure_count = 0
                return True
            return False
        elif self.state == "half_open":
            return self.failure_count < self.half_open_max_calls
        return False
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class AdvancedHealthCheck:
    """Advanced health check with circuit breaker and recovery"""
    
    def __init__(self, name: str, check_func: Callable, 
                 circuit_breaker: Optional[CircuitBreaker] = None,
                 timeout: float = 10.0,
                 critical: bool = False):
        self.name = name
        self.check_func = check_func
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name)
        self.timeout = timeout
        self.critical = critical
        self.history: List[HealthCheckResult] = []
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def check(self) -> HealthCheckResult:
        """Execute health check with circuit breaker protection"""
        start_time = time.time()
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL if self.critical else HealthStatus.UNHEALTHY,
                message=f"Circuit breaker open for {self.name}",
                error="Circuit breaker protection activated",
                recovery_suggestions=[
                    f"Wait {self.circuit_breaker.recovery_timeout} seconds for recovery",
                    "Check service dependencies",
                    "Investigate root cause of failures"
                ]
            )
            self._record_result(result)
            return result
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_check(),
                timeout=self.timeout
            )
            
            # Record success in circuit breaker
            self.circuit_breaker.record_success()
            
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"Health check timeout after {self.timeout}s",
                error="Timeout",
                recovery_suggestions=[
                    "Increase timeout threshold",
                    "Check service performance",
                    "Scale resources if needed"
                ]
            )
            self.circuit_breaker.record_failure()
            
        except Exception as e:
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                error=str(e),
                recovery_suggestions=[
                    "Check service configuration",
                    "Verify dependencies are available",
                    "Review service logs"
                ]
            )
            self.circuit_breaker.record_failure()
        
        # Record response time
        result.response_time_ms = (time.time() - start_time) * 1000
        self._record_result(result)
        return result
    
    async def _execute_check(self) -> HealthCheckResult:
        """Execute the actual health check"""
        if asyncio.iscoroutinefunction(self.check_func):
            check_result = await self.check_func()
        else:
            check_result = self.check_func()
        
        if isinstance(check_result, HealthCheckResult):
            return check_result
        elif isinstance(check_result, dict):
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus(check_result.get('status', 'healthy')),
                message=check_result.get('message', 'OK'),
                metrics=check_result.get('metrics', {}),
                recovery_suggestions=check_result.get('recovery_suggestions', [])
            )
        else:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="OK"
            )
    
    def _record_result(self, result: HealthCheckResult):
        """Record health check result in history"""
        self.history.append(result)
        # Keep only last 100 results
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Log result
        if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            self.logger.error(f"Health check failed: {result.to_dict()}")
        elif result.status == HealthStatus.DEGRADED:
            self.logger.warning(f"Health check degraded: {result.to_dict()}")
        else:
            self.logger.info(f"Health check passed: {result.name}")


class RobustHealthManager:
    """Robust health management with monitoring and alerting"""
    
    def __init__(self, check_interval: float = 30.0):
        self.checks: Dict[str, AdvancedHealthCheck] = {}
        self.check_interval = check_interval
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alert_callbacks: List[Callable] = []
        self.system_metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def add_check(self, health_check: AdvancedHealthCheck):
        """Add health check"""
        self.checks[health_check.name] = health_check
        self.logger.info(f"Added health check: {health_check.name}")
    
    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add alert callback for failed health checks"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Started health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped health monitoring")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            try:
                await self.check_all_health()
                await self._collect_system_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_all_health(self) -> Dict[str, HealthCheckResult]:
        """Check all registered health checks"""
        results = {}
        
        # Run all checks concurrently
        check_tasks = []
        for name, health_check in self.checks.items():
            check_tasks.append(self._safe_check(health_check))
        
        if check_tasks:
            check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            for i, result in enumerate(check_results):
                check_name = list(self.checks.keys())[i]
                
                if isinstance(result, Exception):
                    # Handle check execution errors
                    result = HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.CRITICAL,
                        message=f"Health check execution failed: {result}",
                        error=str(result)
                    )
                
                results[check_name] = result
                
                # Trigger alerts for failed checks
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    await self._trigger_alerts(result)
        
        return results
    
    async def _safe_check(self, health_check: AdvancedHealthCheck) -> HealthCheckResult:
        """Safely execute a health check"""
        try:
            return await health_check.check()
        except Exception as e:
            return HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check exception: {e}",
                error=str(e)
            )
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            network = psutil.net_io_counters()
            
            self.system_metrics = {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                'memory': {
                    'percent': memory.percent,
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                },
                'disk': {
                    'percent': (disk.used / disk.total) * 100,
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3)
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                } if network else None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    async def _trigger_alerts(self, result: HealthCheckResult):
        """Trigger alerts for failed health checks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        if not self.checks:
            return {
                'status': HealthStatus.HEALTHY.value,
                'message': 'No health checks configured',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        # Get latest results
        latest_results = {}
        overall_status = HealthStatus.HEALTHY
        critical_issues = []
        degraded_services = []
        
        for name, health_check in self.checks.items():
            if health_check.history:
                latest_result = health_check.history[-1]
                latest_results[name] = latest_result.to_dict()
                
                # Determine overall status
                if latest_result.status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                    critical_issues.append(latest_result.name)
                elif latest_result.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                elif latest_result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    degraded_services.append(latest_result.name)
        
        # Build status message
        if overall_status == HealthStatus.CRITICAL:
            message = f"Critical issues detected: {', '.join(critical_issues)}"
        elif overall_status == HealthStatus.UNHEALTHY:
            message = "System unhealthy - some services failing"
        elif overall_status == HealthStatus.DEGRADED:
            message = f"System degraded: {', '.join(degraded_services)}"
        else:
            message = "All systems operational"
        
        return {
            'status': overall_status.value,
            'message': message,
            'timestamp': datetime.utcnow().isoformat(),
            'health_checks': latest_results,
            'system_metrics': self.system_metrics,
            'circuit_breakers': {
                name: {
                    'state': check.circuit_breaker.state,
                    'failure_count': check.circuit_breaker.failure_count,
                    'last_failure': check.circuit_breaker.last_failure_time.isoformat() 
                                   if check.circuit_breaker.last_failure_time else None
                }
                for name, check in self.checks.items()
            }
        }


# Pre-built health checks
def create_quantum_coherence_check(scheduler) -> AdvancedHealthCheck:
    """Create quantum coherence health check"""
    
    async def check_coherence():
        if not scheduler.tasks:
            return HealthCheckResult(
                name="quantum_coherence",
                status=HealthStatus.HEALTHY,
                message="No tasks to check",
                metrics={'task_count': 0}
            )
        
        coherences = [task.quantum_coherence for task in scheduler.tasks.values()]
        avg_coherence = sum(coherences) / len(coherences)
        min_coherence = min(coherences)
        
        metrics = {
            'average_coherence': avg_coherence,
            'minimum_coherence': min_coherence,
            'task_count': len(scheduler.tasks),
            'low_coherence_tasks': sum(1 for c in coherences if c < 0.3)
        }
        
        if min_coherence < 0.1:
            status = HealthStatus.CRITICAL
            message = f"Critical: Minimum coherence {min_coherence:.3f} below threshold"
            suggestions = [
                "Apply quantum error correction",
                "Reduce entanglement complexity",
                "Restart affected quantum processes"
            ]
        elif avg_coherence < 0.5:
            status = HealthStatus.DEGRADED
            message = f"Degraded: Average coherence {avg_coherence:.3f} below optimal"
            suggestions = [
                "Monitor quantum decoherence rate",
                "Consider coherence restoration procedures"
            ]
        else:
            status = HealthStatus.HEALTHY
            message = f"Optimal: Average coherence {avg_coherence:.3f}"
            suggestions = []
        
        return HealthCheckResult(
            name="quantum_coherence",
            status=status,
            message=message,
            metrics=metrics,
            recovery_suggestions=suggestions
        )
    
    return AdvancedHealthCheck(
        name="quantum_coherence",
        check_func=check_coherence,
        critical=True
    )


def create_system_resources_check() -> AdvancedHealthCheck:
    """Create system resources health check"""
    
    def check_resources():
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'disk_percent': (disk.used / disk.total) * 100,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3)
        }
        
        # Determine status based on thresholds
        critical_issues = []
        warnings = []
        suggestions = []
        
        if cpu_percent > 90:
            critical_issues.append(f"CPU usage {cpu_percent:.1f}%")
            suggestions.append("Scale CPU resources or optimize workload")
        elif cpu_percent > 70:
            warnings.append(f"High CPU usage {cpu_percent:.1f}%")
        
        if memory.percent > 90:
            critical_issues.append(f"Memory usage {memory.percent:.1f}%")
            suggestions.append("Scale memory or restart services")
        elif memory.percent > 70:
            warnings.append(f"High memory usage {memory.percent:.1f}%")
        
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > 90:
            critical_issues.append(f"Disk usage {disk_percent:.1f}%")
            suggestions.append("Clean up disk space or add storage")
        elif disk_percent > 80:
            warnings.append(f"High disk usage {disk_percent:.1f}%")
        
        # Determine overall status
        if critical_issues:
            status = HealthStatus.CRITICAL
            message = f"Critical resource issues: {', '.join(critical_issues)}"
        elif warnings:
            status = HealthStatus.DEGRADED
            message = f"Resource warnings: {', '.join(warnings)}"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources optimal"
        
        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            metrics=metrics,
            recovery_suggestions=suggestions
        )
    
    return AdvancedHealthCheck(
        name="system_resources",
        check_func=check_resources,
        critical=True
    )