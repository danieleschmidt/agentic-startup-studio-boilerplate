"""
Quantum Task Planner Exceptions

Custom exception classes for quantum task planning operations
with detailed error context and recovery suggestions.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class QuantumTaskPlannerError(Exception):
    """Base exception for all quantum task planner errors"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or "QUANTUM_ERROR"
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


# Task-related exceptions
class TaskNotFoundError(QuantumTaskPlannerError):
    """Raised when a task cannot be found"""
    
    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            error_code="TASK_NOT_FOUND",
            context={"task_id": task_id}
        )


class TaskValidationError(QuantumTaskPlannerError):
    """Raised when task validation fails"""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Invalid {field}: {reason}",
            error_code="TASK_VALIDATION_ERROR",
            context={"field": field, "value": str(value), "reason": reason}
        )


class TaskStateError(QuantumTaskPlannerError):
    """Raised when task state operation is invalid"""
    
    def __init__(self, task_id: str, current_state: str, attempted_operation: str):
        super().__init__(
            f"Cannot perform '{attempted_operation}' on task {task_id} in state '{current_state}'",
            error_code="INVALID_TASK_STATE",
            context={
                "task_id": task_id,
                "current_state": current_state,
                "attempted_operation": attempted_operation
            }
        )


# Quantum-specific exceptions
class QuantumCoherenceError(QuantumTaskPlannerError):
    """Raised when quantum coherence is lost or invalid"""
    
    def __init__(self, task_id: str, coherence: float, threshold: float):
        super().__init__(
            f"Quantum coherence {coherence:.3f} below threshold {threshold:.3f} for task {task_id}",
            error_code="COHERENCE_LOST",
            context={
                "task_id": task_id,
                "coherence": coherence,
                "threshold": threshold
            }
        )


class EntanglementError(QuantumTaskPlannerError):
    """Raised when quantum entanglement operations fail"""
    
    def __init__(self, message: str, bond_id: str = None, task_ids: List[str] = None):
        super().__init__(
            message,
            error_code="ENTANGLEMENT_ERROR",
            context={
                "bond_id": bond_id,
                "task_ids": task_ids or []
            }
        )


class QuantumMeasurementError(QuantumTaskPlannerError):
    """Raised when quantum measurement fails"""
    
    def __init__(self, task_id: str, reason: str):
        super().__init__(
            f"Quantum measurement failed for task {task_id}: {reason}",
            error_code="MEASUREMENT_FAILED",
            context={"task_id": task_id, "reason": reason}
        )


# Scheduling exceptions
class SchedulingError(QuantumTaskPlannerError):
    """Raised when scheduling operations fail"""
    
    def __init__(self, message: str, tasks_affected: List[str] = None):
        super().__init__(
            message,
            error_code="SCHEDULING_ERROR",
            context={"tasks_affected": tasks_affected or []}
        )


class OptimizationError(QuantumTaskPlannerError):
    """Raised when optimization algorithms fail"""
    
    def __init__(self, algorithm: str, iteration: int, reason: str):
        super().__init__(
            f"Optimization failed in {algorithm} at iteration {iteration}: {reason}",
            error_code="OPTIMIZATION_FAILED",
            context={
                "algorithm": algorithm,
                "iteration": iteration,
                "reason": reason
            }
        )


class ResourceConstraintError(QuantumTaskPlannerError):
    """Raised when resource constraints are violated"""
    
    def __init__(self, resource_type: str, required: float, available: float):
        super().__init__(
            f"Insufficient {resource_type}: required {required}, available {available}",
            error_code="RESOURCE_CONSTRAINT_VIOLATION",
            context={
                "resource_type": resource_type,
                "required": required,
                "available": available,
                "deficit": required - available
            }
        )


# API exceptions
class APIError(QuantumTaskPlannerError):
    """Base class for API-related errors"""
    
    def __init__(self, message: str, status_code: int, error_code: str = None):
        super().__init__(message, error_code or "API_ERROR")
        self.status_code = status_code


class AuthenticationError(APIError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401, "AUTHENTICATION_FAILED")


class AuthorizationError(APIError):
    """Raised when authorization fails"""
    
    def __init__(self, resource: str, action: str):
        super().__init__(
            f"Not authorized to {action} {resource}",
            403,
            "AUTHORIZATION_FAILED"
        )


class RateLimitError(APIError):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, limit: int, window: int, retry_after: int):
        super().__init__(
            f"Rate limit exceeded: {limit} requests per {window}s",
            429,
            "RATE_LIMIT_EXCEEDED"
        )
        self.context.update({
            "limit": limit,
            "window": window,
            "retry_after": retry_after
        })


# Storage exceptions
class StorageError(QuantumTaskPlannerError):
    """Raised when storage operations fail"""
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Storage {operation} failed: {reason}",
            error_code="STORAGE_ERROR",
            context={"operation": operation, "reason": reason}
        )


class DatabaseConnectionError(StorageError):
    """Raised when database connection fails"""
    
    def __init__(self, connection_string: str, reason: str):
        super().__init__(
            "connect",
            f"Failed to connect to database: {reason}"
        )
        self.context.update({
            "connection_string": connection_string,
            "database_error": reason
        })


# Configuration exceptions
class ConfigurationError(QuantumTaskPlannerError):
    """Raised when configuration is invalid"""
    
    def __init__(self, parameter: str, value: Any, reason: str):
        super().__init__(
            f"Invalid configuration for {parameter}: {reason}",
            error_code="CONFIGURATION_ERROR",
            context={
                "parameter": parameter,
                "value": str(value),
                "reason": reason
            }
        )


# Quantum algorithm specific exceptions
class QuantumAnnealingError(OptimizationError):
    """Raised when quantum annealing fails"""
    
    def __init__(self, temperature: float, energy: float, reason: str):
        super().__init__(
            "quantum_annealing",
            0,  # iteration not applicable
            reason
        )
        self.context.update({
            "temperature": temperature,
            "energy": energy
        })


class SuperpositionCollapseError(QuantumTaskPlannerError):
    """Raised when quantum superposition collapses unexpectedly"""
    
    def __init__(self, task_id: str, expected_states: List[str], collapsed_state: str):
        super().__init__(
            f"Unexpected superposition collapse for task {task_id}",
            error_code="SUPERPOSITION_COLLAPSE",
            context={
                "task_id": task_id,
                "expected_states": expected_states,
                "collapsed_state": collapsed_state
            }
        )


# Recovery and circuit breaker exceptions
class CircuitBreakerError(QuantumTaskPlannerError):
    """Raised when circuit breaker is open"""
    
    def __init__(self, service: str, failure_count: int, threshold: int):
        super().__init__(
            f"Circuit breaker open for {service}: {failure_count}/{threshold} failures",
            error_code="CIRCUIT_BREAKER_OPEN",
            context={
                "service": service,
                "failure_count": failure_count,
                "threshold": threshold
            }
        )


class RetryExhaustedError(QuantumTaskPlannerError):
    """Raised when retry attempts are exhausted"""
    
    def __init__(self, operation: str, attempts: int, last_error: str):
        super().__init__(
            f"Retry exhausted for {operation} after {attempts} attempts: {last_error}",
            error_code="RETRY_EXHAUSTED",
            context={
                "operation": operation,
                "attempts": attempts,
                "last_error": last_error
            }
        )


# Exception utility functions
def handle_quantum_exception(func):
    """Decorator to handle and log quantum task planner exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QuantumTaskPlannerError as e:
            # Log the exception with context
            import logging
            logger = logging.getLogger(func.__module__)
            logger.error(f"Quantum exception in {func.__name__}: {e.to_dict()}")
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            import logging
            logger = logging.getLogger(func.__module__)
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise QuantumTaskPlannerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__, "original_error": str(e)}
            )
    return wrapper


async def handle_async_quantum_exception(func):
    """Async version of quantum exception handler"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except QuantumTaskPlannerError as e:
            import logging
            logger = logging.getLogger(func.__module__)
            logger.error(f"Quantum exception in {func.__name__}: {e.to_dict()}")
            raise
        except Exception as e:
            import logging
            logger = logging.getLogger(func.__module__)
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise QuantumTaskPlannerError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                context={"function": func.__name__, "original_error": str(e)}
            )
    return wrapper