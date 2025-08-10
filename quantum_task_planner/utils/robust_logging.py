"""
Robust Logging System

Advanced logging with structured formats, correlation IDs, metrics, and alerting.
"""

import logging
import json
import sys
import uuid
import time
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import contextvars
import threading
from pathlib import Path
import gzip
import os


# Context variables for request tracking
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id')
user_id: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='system')
request_start_time: contextvars.ContextVar[float] = contextvars.ContextVar('request_start_time')


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60  # Custom level for security events
    AUDIT = 70     # Custom level for audit events


@dataclass
class LogContext:
    """Structured logging context"""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "system"
    component: str = "quantum_planner"
    operation: str = ""
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'component': self.component,
            'operation': self.operation,
            'duration_ms': (time.time() - self.start_time) * 1000,
            'metadata': self.metadata
        }


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def __init__(self, include_trace: bool = True):
        self.include_trace = include_trace
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'thread_name': record.threadName
        }
        
        # Add correlation context if available
        try:
            log_entry['correlation_id'] = correlation_id.get()
        except LookupError:
            pass
        
        try:
            log_entry['user_id'] = user_id.get()
        except LookupError:
            pass
        
        try:
            start_time = request_start_time.get()
            log_entry['request_duration_ms'] = (time.time() - start_time) * 1000
        except LookupError:
            pass
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if self.include_trace else None
            }
        
        # Add custom fields from extra
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Performance metrics
        if hasattr(record, 'performance'):
            log_entry['performance'] = record.performance
        
        # Security context
        if hasattr(record, 'security'):
            log_entry['security'] = record.security
        
        # Quantum-specific fields
        if hasattr(record, 'quantum'):
            log_entry['quantum'] = record.quantum
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class QuantumLoggerAdapter(logging.LoggerAdapter):
    """Enhanced logger adapter for quantum task planner"""
    
    def __init__(self, logger: logging.Logger, component: str = "quantum_planner"):
        self.component = component
        super().__init__(logger, {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message with context"""
        
        # Get or create correlation ID
        try:
            corr_id = correlation_id.get()
        except LookupError:
            corr_id = str(uuid.uuid4())
            correlation_id.set(corr_id)
        
        # Build structured extra fields
        extra_fields = {
            'component': self.component,
            'correlation_id': corr_id
        }
        
        # Add user context
        try:
            extra_fields['user_id'] = user_id.get()
        except LookupError:
            extra_fields['user_id'] = 'system'
        
        # Merge with provided extra
        if 'extra' in kwargs:
            if isinstance(kwargs['extra'], dict):
                extra_fields.update(kwargs['extra'])
            kwargs['extra'] = extra_fields
        else:
            kwargs['extra'] = {'extra_fields': extra_fields}
        
        return msg, kwargs
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        performance_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow().isoformat()
        }
        performance_data.update(kwargs)
        
        self.info(f"Performance: {operation} completed in {duration_ms:.2f}ms", 
                 extra={'performance': performance_data})
    
    def security(self, event: str, severity: str = "info", **kwargs):
        """Log security events"""
        security_data = {
            'event': event,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat()
        }
        security_data.update(kwargs)
        
        level = getattr(logging, severity.upper(), logging.INFO)
        self.log(level, f"Security: {event}", extra={'security': security_data})
    
    def audit(self, action: str, resource: str, outcome: str = "success", **kwargs):
        """Log audit events"""
        audit_data = {
            'action': action,
            'resource': resource,
            'outcome': outcome,
            'timestamp': datetime.utcnow().isoformat()
        }
        audit_data.update(kwargs)
        
        self.info(f"Audit: {action} on {resource} - {outcome}", 
                 extra={'audit': audit_data})
    
    def quantum(self, event: str, task_id: str = None, coherence: float = None, 
                entanglement_count: int = None, **kwargs):
        """Log quantum-specific events"""
        quantum_data = {
            'event': event,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if task_id:
            quantum_data['task_id'] = task_id
        if coherence is not None:
            quantum_data['coherence'] = coherence
        if entanglement_count is not None:
            quantum_data['entanglement_count'] = entanglement_count
        
        quantum_data.update(kwargs)
        
        self.info(f"Quantum: {event}", extra={'quantum': quantum_data})


class LogRotatingHandler(logging.handlers.RotatingFileHandler):
    """Enhanced rotating file handler with compression"""
    
    def __init__(self, filename: str, maxBytes: int = 10*1024*1024, 
                 backupCount: int = 10, compress: bool = True, **kwargs):
        self.compress = compress
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, **kwargs)
    
    def doRollover(self):
        """Perform log rotation with optional compression"""
        super().doRollover()
        
        if self.compress and self.backupCount > 0:
            # Compress the most recent backup
            backup_name = f"{self.baseFilename}.1"
            compressed_name = f"{backup_name}.gz"
            
            if os.path.exists(backup_name):
                try:
                    with open(backup_name, 'rb') as f_in:
                        with gzip.open(compressed_name, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove uncompressed file
                    os.remove(backup_name)
                    
                    # Update backup numbering for compressed files
                    for i in range(2, self.backupCount + 1):
                        old_name = f"{backup_name}.{i-1}.gz"
                        new_name = f"{backup_name}.{i}.gz"
                        if os.path.exists(old_name):
                            os.rename(old_name, new_name)
                    
                except Exception as e:
                    self.handleError(logging.LogRecord("", 0, "", 0, f"Log compression failed: {e}", (), None))


class MetricsCollector:
    """Collect and aggregate log metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'log_counts': {},
            'error_rates': {},
            'response_times': [],
            'quantum_events': {},
            'security_events': [],
            'last_reset': time.time()
        }
        self._lock = threading.Lock()
    
    def record_log(self, level: str, component: str):
        """Record log entry"""
        with self._lock:
            key = f"{component}:{level}"
            self.metrics['log_counts'][key] = self.metrics['log_counts'].get(key, 0) + 1
    
    def record_performance(self, operation: str, duration_ms: float):
        """Record performance metric"""
        with self._lock:
            self.metrics['response_times'].append({
                'operation': operation,
                'duration_ms': duration_ms,
                'timestamp': time.time()
            })
            
            # Keep only last 1000 entries
            if len(self.metrics['response_times']) > 1000:
                self.metrics['response_times'] = self.metrics['response_times'][-1000:]
    
    def record_quantum_event(self, event: str):
        """Record quantum event"""
        with self._lock:
            self.metrics['quantum_events'][event] = self.metrics['quantum_events'].get(event, 0) + 1
    
    def record_security_event(self, event: str, severity: str):
        """Record security event"""
        with self._lock:
            self.metrics['security_events'].append({
                'event': event,
                'severity': severity,
                'timestamp': time.time()
            })
            
            # Keep only last 100 security events
            if len(self.metrics['security_events']) > 100:
                self.metrics['security_events'] = self.metrics['security_events'][-100:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            # Calculate error rates
            error_rates = {}
            total_logs = sum(self.metrics['log_counts'].values())
            for key, count in self.metrics['log_counts'].items():
                component, level = key.split(':', 1)
                if level in ['ERROR', 'CRITICAL']:
                    error_rates[component] = error_rates.get(component, 0) + count
            
            # Calculate average response times
            avg_response_times = {}
            for entry in self.metrics['response_times']:
                op = entry['operation']
                if op not in avg_response_times:
                    avg_response_times[op] = []
                avg_response_times[op].append(entry['duration_ms'])
            
            for op in avg_response_times:
                times = avg_response_times[op]
                avg_response_times[op] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
            
            return {
                'log_counts': self.metrics['log_counts'].copy(),
                'error_rates': error_rates,
                'total_error_rate': sum(error_rates.values()) / max(1, total_logs),
                'response_times': avg_response_times,
                'quantum_events': self.metrics['quantum_events'].copy(),
                'security_events_count': len(self.metrics['security_events']),
                'uptime_seconds': time.time() - self.metrics['last_reset']
            }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            self.metrics = {
                'log_counts': {},
                'error_rates': {},
                'response_times': [],
                'quantum_events': {},
                'security_events': [],
                'last_reset': time.time()
            }


class MetricsHandler(logging.Handler):
    """Log handler that collects metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        super().__init__()
        self.metrics_collector = metrics_collector
    
    def emit(self, record: logging.LogRecord):
        """Emit log record and collect metrics"""
        # Record basic log
        component = getattr(record, 'component', 'unknown')
        self.metrics_collector.record_log(record.levelname, component)
        
        # Record performance metrics
        if hasattr(record, 'performance'):
            perf_data = record.performance
            self.metrics_collector.record_performance(
                perf_data.get('operation', 'unknown'),
                perf_data.get('duration_ms', 0)
            )
        
        # Record quantum events
        if hasattr(record, 'quantum'):
            quantum_data = record.quantum
            self.metrics_collector.record_quantum_event(
                quantum_data.get('event', 'unknown')
            )
        
        # Record security events
        if hasattr(record, 'security'):
            security_data = record.security
            self.metrics_collector.record_security_event(
                security_data.get('event', 'unknown'),
                security_data.get('severity', 'info')
            )


# Global metrics collector
_metrics_collector = MetricsCollector()


def setup_robust_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    enable_metrics: bool = True,
    max_file_size: int = 10*1024*1024,
    backup_count: int = 10,
    compress_logs: bool = True
) -> QuantumLoggerAdapter:
    """Setup robust logging system"""
    
    # Create logger
    logger = logging.getLogger("quantum_task_planner")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if structured:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = LogRotatingHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            compress=compress_logs
        )
        if structured:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        logger.addHandler(file_handler)
    
    # Metrics handler (optional)
    if enable_metrics:
        metrics_handler = MetricsHandler(_metrics_collector)
        logger.addHandler(metrics_handler)
    
    # Return adapter
    return QuantumLoggerAdapter(logger)


def get_metrics() -> Dict[str, Any]:
    """Get current logging metrics"""
    return _metrics_collector.get_metrics()


def reset_metrics():
    """Reset logging metrics"""
    _metrics_collector.reset_metrics()


def performance_logger(operation_name: str = None):
    """Decorator for automatic performance logging"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                logger = QuantumLoggerAdapter(logging.getLogger(func.__module__))
                logger.performance(op_name, duration, status="success")
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                logger = QuantumLoggerAdapter(logging.getLogger(func.__module__))
                logger.performance(op_name, duration, status="error", error=str(e))
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                
                logger = QuantumLoggerAdapter(logging.getLogger(func.__module__))
                logger.performance(op_name, duration, status="success")
                
                return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                logger = QuantumLoggerAdapter(logging.getLogger(func.__module__))
                logger.performance(op_name, duration, status="error", error=str(e))
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def correlation_context(corr_id: str = None, user: str = None):
    """Context manager for correlation tracking"""
    class CorrelationContext:
        def __init__(self):
            self.correlation_token = None
            self.user_token = None
            self.time_token = None
        
        def __enter__(self):
            self.correlation_token = correlation_id.set(corr_id or str(uuid.uuid4()))
            if user:
                self.user_token = user_id.set(user)
            self.time_token = request_start_time.set(time.time())
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            correlation_id.reset(self.correlation_token)
            if self.user_token:
                user_id.reset(self.user_token)
            request_start_time.reset(self.time_token)
    
    return CorrelationContext()