"""
Simple error handling and resilience features for Generation 2
"""

import traceback
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from functools import wraps
from datetime import datetime
import asyncio

# Type variable for generic functions
T = TypeVar('T')

logger = logging.getLogger(__name__)


class QuantumTaskError(Exception):
    """Base exception for quantum task operations"""
    pass


class TaskNotFoundError(QuantumTaskError):
    """Task not found exception"""
    pass


class QuantumStateError(QuantumTaskError):
    """Quantum state operation error"""
    pass


class OptimizationError(QuantumTaskError):
    """Optimization process error"""
    pass


class CircuitBreakerOpen(Exception):
    """Circuit breaker is open"""
    pass


class SimpleCircuitBreaker:
    """Simple circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).seconds
        return time_since_failure >= self.reset_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RetryHandler:
    """Simple retry logic with exponential backoff"""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Decorator for retry logic"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                attempt = 1
                current_delay = delay
                
                while attempt <= max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts:
                            logger.error(f"Final attempt failed for {func.__name__}: {e}")
                            raise e
                        
                        logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay} seconds...")
                        
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                        attempt += 1
                
                raise Exception("Retry logic error")  # Should never reach here
            
            return wrapper
        return decorator
    
    @staticmethod
    async def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """Async decorator for retry logic"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                attempt = 1
                current_delay = delay
                
                while attempt <= max_attempts:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts:
                            logger.error(f"Final async attempt failed for {func.__name__}: {e}")
                            raise e
                        
                        logger.warning(f"Async attempt {attempt} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay} seconds...")
                        
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                        attempt += 1
                
                raise Exception("Async retry logic error")  # Should never reach here
            
            return wrapper
        return decorator


class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    def handle_gracefully(default_return: Any = None, log_errors: bool = True):
        """Graceful error handling decorator"""
        def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Union[T, Any]:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_errors:
                        logger.error(f"Error in {func.__name__}: {e}")
                        logger.debug(f"Full traceback: {traceback.format_exc()}")
                    
                    return default_return
            
            return wrapper
        return decorator
    
    @staticmethod
    def log_errors(func: Callable[..., T]) -> Callable[..., T]:
        """Simple error logging decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                raise e
        
        return wrapper


class HealthChecker:
    """Simple health checking for system components"""
    
    def __init__(self):
        self.health_status: Dict[str, Dict[str, Any]] = {}
    
    def check_component_health(self, component_name: str, check_func: Callable[[], bool]) -> Dict[str, Any]:
        """Check health of a specific component"""
        try:
            is_healthy = check_func()
            status = {
                "healthy": is_healthy,
                "last_check": datetime.now().isoformat(),
                "error": None
            }
        except Exception as e:
            status = {
                "healthy": False,
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
        
        self.health_status[component_name] = status
        return status
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.health_status:
            return {"healthy": False, "message": "No health checks performed"}
        
        all_healthy = all(status["healthy"] for status in self.health_status.values())
        unhealthy_components = [
            name for name, status in self.health_status.items() 
            if not status["healthy"]
        ]
        
        return {
            "healthy": all_healthy,
            "components": self.health_status,
            "unhealthy_components": unhealthy_components,
            "total_components": len(self.health_status),
            "healthy_components": len(self.health_status) - len(unhealthy_components)
        }


# Global instances
circuit_breaker = SimpleCircuitBreaker()
health_checker = HealthChecker()


def safe_quantum_operation(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator combining circuit breaker, retry, and error handling"""
    @RetryHandler.retry(max_attempts=3, delay=0.5)
    @ErrorHandler.log_errors
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        return circuit_breaker.call(func, *args, **kwargs)
    
    return wrapper