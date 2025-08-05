"""
Quantum Task Planner Logging and Monitoring

Advanced logging system with quantum state tracking, performance metrics,
structured logging, and real-time monitoring capabilities.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import threading
from collections import defaultdict, deque
import traceback

import structlog
from structlog.processors import JSONRenderer, TimeStamper
from pythonjsonlogger import jsonlogger


@dataclass
class QuantumMetric:
    """Quantum-specific metric data"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    task_id: Optional[str] = None
    quantum_coherence: Optional[float] = None
    entanglement_count: Optional[int] = None
    measurement_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetric:
    """Performance measurement data"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    error_type: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


class QuantumLogger:
    """Advanced logging system for quantum task planner"""
    
    def __init__(self, name: str, log_level: str = "INFO", log_dir: Optional[Path] = None):
        self.name = name
        self.log_level = log_level.upper()
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.quantum_metrics: deque = deque(maxlen=10000)
        self.performance_metrics: deque = deque(maxlen=10000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize structured logger
        self._setup_structured_logging()
        
        # Start background metrics collector
        self._metrics_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self._metrics_thread.start()
    
    def _setup_structured_logging(self):
        """Setup structured logging with JSON format"""
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Create main logger
        self.logger = structlog.get_logger(self.name)
        
        # Setup file handlers
        self._setup_file_handlers()
        
        # Setup console handler
        self._setup_console_handler()
    
    def _setup_file_handlers(self):
        """Setup file-based log handlers"""
        
        # Main application log
        app_handler = logging.FileHandler(
            self.log_dir / f"{self.name}.log"
        )
        app_handler.setFormatter(
            jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
        )
        
        # Quantum metrics log
        quantum_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_quantum.log"
        )
        quantum_handler.setFormatter(
            jsonlogger.JsonFormatter(
                '%(asctime)s %(message)s'
            )
        )
        
        # Performance metrics log
        perf_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_performance.log"
        )
        perf_handler.setFormatter(
            jsonlogger.JsonFormatter(
                '%(asctime)s %(message)s'
            )
        )
        
        # Error log
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s %(exc_info)s'
            )
        )
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))
        root_logger.addHandler(app_handler)
        root_logger.addHandler(quantum_handler)
        root_logger.addHandler(perf_handler)
        root_logger.addHandler(error_handler)
    
    def _setup_console_handler(self):
        """Setup console logging"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
        )
        
        # Add to specific logger to avoid duplicate logs
        specific_logger = logging.getLogger(self.name)
        specific_logger.addHandler(console_handler)
        specific_logger.setLevel(getattr(logging, self.log_level))
    
    def _collect_system_metrics(self):
        """Background thread to collect system metrics"""
        try:
            import psutil
        except ImportError:
            return  # psutil not available
        
        while True:
            try:
                # Collect system metrics every 30 seconds
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_metrics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
                
                self.logger.info("system_metrics", **system_metrics)
                time.sleep(30)
                
            except Exception as e:
                self.logger.error("metrics_collection_error", error=str(e))
                time.sleep(60)  # Wait longer on error
    
    def log_quantum_metric(self, metric: QuantumMetric):
        """Log quantum-specific metric"""
        self.quantum_metrics.append(metric)
        
        self.logger.info(
            "quantum_metric",
            metric_name=metric.metric_name,
            value=metric.value,
            unit=metric.unit,
            task_id=metric.task_id,
            quantum_coherence=metric.quantum_coherence,
            entanglement_count=metric.entanglement_count,
            measurement_type=metric.measurement_type,
            context=metric.context or {}
        )
    
    def log_performance_metric(self, metric: PerformanceMetric):
        """Log performance metric"""
        self.performance_metrics.append(metric)
        self.operation_stats[metric.operation].append(metric.duration_ms)
        
        self.logger.info(
            "performance_metric",
            operation=metric.operation,
            duration_ms=metric.duration_ms,
            success=metric.success,
            error_type=metric.error_type,
            memory_usage_mb=metric.memory_usage_mb,
            cpu_percent=metric.cpu_percent,
            context=metric.context or {}
        )
    
    def log_task_creation(self, task_id: str, title: str, priority: str, 
                         quantum_coherence: float, complexity: float):
        """Log task creation event"""
        self.logger.info(
            "task_created",
            task_id=task_id,
            title=title,
            priority=priority,
            quantum_coherence=quantum_coherence,
            complexity_factor=complexity,
            event_type="task_lifecycle"
        )
    
    def log_task_state_change(self, task_id: str, old_state: str, new_state: str,
                             probability: float, observer_effect: float):
        """Log quantum state change"""
        self.logger.info(
            "quantum_state_change",
            task_id=task_id,
            old_state=old_state,
            new_state=new_state,
            probability=probability,
            observer_effect=observer_effect,
            event_type="quantum_measurement"
        )
        
        # Log as quantum metric
        metric = QuantumMetric(
            metric_name="state_transition",
            value=probability,
            unit="probability",
            timestamp=datetime.utcnow(),
            task_id=task_id,
            measurement_type="state_collapse",
            context={
                "old_state": old_state,
                "new_state": new_state,
                "observer_effect": observer_effect
            }
        )
        self.log_quantum_metric(metric)
    
    def log_entanglement_creation(self, bond_id: str, task_ids: List[str], 
                                 entanglement_type: str, strength: float):
        """Log entanglement creation"""
        self.logger.info(
            "entanglement_created",
            bond_id=bond_id,
            task_ids=task_ids,
            entanglement_type=entanglement_type,
            strength=strength,
            task_count=len(task_ids),
            event_type="quantum_entanglement"
        )
        
        # Log as quantum metric
        metric = QuantumMetric(
            metric_name="entanglement_strength",
            value=strength,
            unit="correlation",
            timestamp=datetime.utcnow(),
            entanglement_count=len(task_ids),
            measurement_type="entanglement_creation",
            context={
                "bond_id": bond_id,
                "entanglement_type": entanglement_type,
                "task_ids": task_ids
            }
        )
        self.log_quantum_metric(metric)
    
    def log_optimization_iteration(self, iteration: int, energy: float, 
                                  temperature: float, accepted: bool):
        """Log optimization iteration"""
        self.logger.debug(
            "optimization_iteration",
            iteration=iteration,
            energy=energy,
            temperature=temperature,
            accepted=accepted,
            event_type="quantum_optimization"
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        
        self.logger.error(
            "application_error",
            error_type=error_type,
            error_message=str(error),
            error_count=self.error_counts[error_type],
            context=context or {},
            stack_trace=traceback.format_exc(),
            event_type="error"
        )
    
    def get_performance_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if operation:
            durations = self.operation_stats.get(operation, [])
            if not durations:
                return {"operation": operation, "stats": "No data"}
            
            return {
                "operation": operation,
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if durations else 0,
                "p99_ms": sorted(durations)[int(len(durations) * 0.99)] if durations else 0
            }
        else:
            return {
                operation: {
                    "count": len(durations),
                    "avg_ms": sum(durations) / len(durations) if durations else 0
                }
                for operation, durations in self.operation_stats.items()
            }
    
    def get_quantum_metrics_summary(self) -> Dict[str, Any]:
        """Get quantum metrics summary"""
        if not self.quantum_metrics:
            return {"status": "No quantum metrics available"}
        
        coherence_values = [
            m.quantum_coherence for m in self.quantum_metrics 
            if m.quantum_coherence is not None
        ]
        
        entanglement_counts = [
            m.entanglement_count for m in self.quantum_metrics
            if m.entanglement_count is not None
        ]
        
        return {
            "total_metrics": len(self.quantum_metrics),
            "coherence_stats": {
                "avg": sum(coherence_values) / len(coherence_values) if coherence_values else 0,
                "min": min(coherence_values) if coherence_values else 0,
                "max": max(coherence_values) if coherence_values else 0
            },
            "entanglement_stats": {
                "avg_count": sum(entanglement_counts) / len(entanglement_counts) if entanglement_counts else 0,
                "max_count": max(entanglement_counts) if entanglement_counts else 0
            },
            "error_counts": dict(self.error_counts)
        }


# Performance monitoring decorators
def monitor_performance(operation_name: str = None, log_args: bool = False):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                # Get memory usage
                try:
                    import psutil
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    memory_before = None
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate metrics
                duration_ms = (time.time() - start_time) * 1000
                
                try:
                    import psutil
                    process = psutil.Process()
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage = memory_after - memory_before if memory_before else None
                    cpu_percent = process.cpu_percent()
                except ImportError:
                    memory_usage = None
                    cpu_percent = None
                
                # Log performance metric
                metric = PerformanceMetric(
                    operation=op_name,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=True,
                    memory_usage_mb=memory_usage,
                    cpu_percent=cpu_percent,
                    context={
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()) if log_args else None
                    }
                )
                
                # Get logger instance (this is a simplified approach)
                logger = logging.getLogger("quantum_task_planner")
                if hasattr(logger, 'log_performance_metric'):
                    logger.log_performance_metric(metric)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                metric = PerformanceMetric(
                    operation=op_name,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=False,
                    error_type=type(e).__name__,
                    context={
                        "error_message": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()) if log_args else None
                    }
                )
                
                logger = logging.getLogger("quantum_task_planner")
                if hasattr(logger, 'log_performance_metric'):
                    logger.log_performance_metric(metric)
                
                raise
        
        return wrapper
    return decorator


def monitor_async_performance(operation_name: str = None, log_args: bool = False):
    """Async version of performance monitoring decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                metric = PerformanceMetric(
                    operation=op_name,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=True,
                    context={
                        "async": True,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()) if log_args else None
                    }
                )
                
                logger = logging.getLogger("quantum_task_planner")
                if hasattr(logger, 'log_performance_metric'):
                    logger.log_performance_metric(metric)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                metric = PerformanceMetric(
                    operation=op_name,
                    duration_ms=duration_ms,
                    timestamp=datetime.utcnow(),
                    success=False,
                    error_type=type(e).__name__,
                    context={
                        "async": True,
                        "error_message": str(e),
                        "args_count": len(args)
                    }
                )
                
                logger = logging.getLogger("quantum_task_planner")
                if hasattr(logger, 'log_performance_metric'):
                    logger.log_performance_metric(metric)
                
                raise
        
        return wrapper
    return decorator


@contextmanager
def quantum_operation_context(logger: QuantumLogger, operation: str, 
                             task_id: str = None, context: Dict[str, Any] = None):
    """Context manager for quantum operations"""
    start_time = time.time()
    logger.logger.info(
        "quantum_operation_start",
        operation=operation,
        task_id=task_id,
        context=context or {}
    )
    
    try:
        yield
        duration_ms = (time.time() - start_time) * 1000
        logger.logger.info(
            "quantum_operation_complete",
            operation=operation,
            task_id=task_id,
            duration_ms=duration_ms,
            success=True
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.logger.error(
            "quantum_operation_failed",
            operation=operation,
            task_id=task_id,
            duration_ms=duration_ms,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise


@asynccontextmanager
async def async_quantum_operation_context(logger: QuantumLogger, operation: str,
                                         task_id: str = None, context: Dict[str, Any] = None):
    """Async context manager for quantum operations"""
    start_time = time.time()
    logger.logger.info(
        "quantum_operation_start",
        operation=operation,
        task_id=task_id,
        context=context or {}
    )
    
    try:
        yield
        duration_ms = (time.time() - start_time) * 1000
        logger.logger.info(
            "quantum_operation_complete",
            operation=operation,
            task_id=task_id,
            duration_ms=duration_ms,
            success=True
        )
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.logger.error(
            "quantum_operation_failed",
            operation=operation,
            task_id=task_id,
            duration_ms=duration_ms,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise


# Global logger instance
_global_logger: Optional[QuantumLogger] = None


def get_logger(name: str = "quantum_task_planner") -> QuantumLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = QuantumLogger(name)
    return _global_logger


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None) -> QuantumLogger:
    """Setup global logging configuration"""
    global _global_logger
    _global_logger = QuantumLogger("quantum_task_planner", log_level, log_dir)
    return _global_logger