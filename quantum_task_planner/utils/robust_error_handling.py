"""
Robust Error Handling System - Comprehensive error management with quantum resilience

This module implements advanced error handling, recovery mechanisms, and system resilience
with quantum-inspired fault tolerance and consciousness-aware error recovery.
"""

import asyncio
import traceback
import functools
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels"""
    TRACE = ("trace", 0.1, "debug")
    DEBUG = ("debug", 0.2, "info")
    INFO = ("info", 0.3, "info")
    WARNING = ("warning", 0.5, "warning")
    ERROR = ("error", 0.7, "error")
    CRITICAL = ("critical", 0.9, "critical")
    QUANTUM_CATASTROPHE = ("quantum_catastrophe", 1.0, "critical")
    
    def __init__(self, name: str, impact: float, log_level: str):
        self.impact = impact
        self.log_level = log_level


class ErrorCategory(Enum):
    """Categories of errors for classification"""
    SYSTEM = "system"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    COMPUTATION = "computation"
    QUANTUM_STATE = "quantum_state"
    CONSCIOUSNESS = "consciousness"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"


class RecoveryStrategy(Enum):
    """Error recovery strategies"""
    IGNORE = "ignore"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    QUANTUM_HEALING = "quantum_healing"
    CONSCIOUSNESS_RESET = "consciousness_reset"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    exception_type: str
    message: str
    stack_trace: str
    function_name: str
    module_name: str
    quantum_state: Optional[Dict[str, Any]] = None
    consciousness_level: Optional[float] = None
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    user_context: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action specification"""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    fallback_function: Optional[Callable] = None
    quantum_healing_factor: float = 0.1
    consciousness_threshold: float = 0.5


class QuantumCircuitBreaker:
    """Quantum-enhanced circuit breaker for error resilience"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.quantum_coherence = 1.0
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half-open"
                    logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                self._on_success()
                return result
            
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = "closed"
        # Restore quantum coherence
        self.quantum_coherence = min(1.0, self.quantum_coherence + 0.1)
        
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Reduce quantum coherence
        self.quantum_coherence = max(0.0, self.quantum_coherence - 0.2)
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker OPENED - {self.failure_count} failures")


class ErrorAnalyzer:
    """Advanced error analysis with pattern recognition"""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[ErrorContext]] = {}
        self.error_frequency: Dict[str, int] = {}
        self.recovery_success_rates: Dict[RecoveryStrategy, float] = {}
        self.quantum_error_correlation = 0.0
        
    def analyze_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Analyze error and provide insights"""
        
        error_signature = self._generate_error_signature(error_context)
        
        # Record error pattern
        if error_signature not in self.error_patterns:
            self.error_patterns[error_signature] = []
        
        self.error_patterns[error_signature].append(error_context)
        self.error_frequency[error_signature] = self.error_frequency.get(error_signature, 0) + 1
        
        # Analyze patterns
        analysis = {
            "error_signature": error_signature,
            "frequency": self.error_frequency[error_signature],
            "first_occurrence": self.error_patterns[error_signature][0].timestamp.isoformat(),
            "recent_occurrence": error_context.timestamp.isoformat(),
            "pattern_analysis": self._analyze_error_pattern(error_signature),
            "recommended_strategy": self._recommend_recovery_strategy(error_context),
            "quantum_correlation": self._calculate_quantum_correlation(error_context),
            "consciousness_impact": self._assess_consciousness_impact(error_context)
        }
        
        return analysis
    
    def _generate_error_signature(self, error_context: ErrorContext) -> str:
        """Generate unique signature for error pattern matching"""
        
        signature_components = [
            error_context.category.value,
            error_context.exception_type,
            error_context.function_name,
            error_context.module_name
        ]
        
        # Add quantum state signature if available
        if error_context.quantum_state:
            quantum_hash = hash(str(sorted(error_context.quantum_state.items())))
            signature_components.append(f"quantum_{quantum_hash}")
        
        return "|".join(signature_components)
    
    def _analyze_error_pattern(self, error_signature: str) -> Dict[str, Any]:
        """Analyze error pattern for insights"""
        
        errors = self.error_patterns[error_signature]
        
        if len(errors) < 2:
            return {"pattern_type": "isolated", "confidence": 0.1}
        
        # Time pattern analysis
        time_intervals = []
        for i in range(1, len(errors)):
            interval = (errors[i].timestamp - errors[i-1].timestamp).total_seconds()
            time_intervals.append(interval)
        
        avg_interval = np.mean(time_intervals) if time_intervals else 0
        interval_variance = np.var(time_intervals) if len(time_intervals) > 1 else 0
        
        # Determine pattern type
        if len(errors) > 5 and interval_variance < (avg_interval * 0.1):
            pattern_type = "periodic"
            confidence = 0.9
        elif len(errors) > 3 and avg_interval < 300:  # Less than 5 minutes
            pattern_type = "cascading"
            confidence = 0.8
        elif len(errors) > 10:
            pattern_type = "chronic"
            confidence = 0.7
        else:
            pattern_type = "sporadic"
            confidence = 0.4
        
        return {
            "pattern_type": pattern_type,
            "confidence": confidence,
            "frequency": len(errors),
            "average_interval_seconds": avg_interval,
            "interval_variance": interval_variance
        }
    
    def _recommend_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Recommend optimal recovery strategy based on error analysis"""
        
        # Strategy selection based on error characteristics
        if error_context.category == ErrorCategory.NETWORK:
            return RecoveryStrategy.RETRY
        
        elif error_context.category == ErrorCategory.QUANTUM_STATE:
            return RecoveryStrategy.QUANTUM_HEALING
        
        elif error_context.category == ErrorCategory.CONSCIOUSNESS:
            return RecoveryStrategy.CONSCIOUSNESS_RESET
        
        elif error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        elif error_context.category == ErrorCategory.RESOURCE_EXHAUSTION:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        elif error_context.category == ErrorCategory.EXTERNAL_SERVICE:
            return RecoveryStrategy.FALLBACK
        
        else:
            return RecoveryStrategy.RETRY
    
    def _calculate_quantum_correlation(self, error_context: ErrorContext) -> float:
        """Calculate correlation with quantum states"""
        
        if not error_context.quantum_state:
            return 0.0
        
        # Simple quantum correlation metric
        quantum_values = [
            v for v in error_context.quantum_state.values() 
            if isinstance(v, (int, float))
        ]
        
        if not quantum_values:
            return 0.0
        
        # Calculate correlation with error severity
        quantum_mean = np.mean(quantum_values)
        correlation = abs(quantum_mean - error_context.severity.impact)
        
        return min(1.0, correlation)
    
    def _assess_consciousness_impact(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Assess impact on consciousness systems"""
        
        impact = {
            "consciousness_degradation": 0.0,
            "recovery_potential": 1.0,
            "meditation_required": False
        }
        
        if error_context.consciousness_level is not None:
            if error_context.consciousness_level < 0.3:
                impact["consciousness_degradation"] = 0.8
                impact["recovery_potential"] = 0.2
                impact["meditation_required"] = True
            elif error_context.consciousness_level < 0.6:
                impact["consciousness_degradation"] = 0.4
                impact["recovery_potential"] = 0.6
        
        # Category-specific impacts
        if error_context.category == ErrorCategory.CONSCIOUSNESS:
            impact["consciousness_degradation"] = 0.9
            impact["meditation_required"] = True
        
        return impact


class RobustErrorHandler:
    """Comprehensive error handling system with quantum resilience"""
    
    def __init__(self):
        self.error_analyzer = ErrorAnalyzer()
        self.circuit_breakers: Dict[str, QuantumCircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, RecoveryAction] = {}
        self.quantum_healing_enabled = True
        self.consciousness_recovery_enabled = True
        
        # Error handling metrics
        self.total_errors = 0
        self.recovered_errors = 0
        self.critical_errors = 0
        self.quantum_healing_successes = 0
        
        # Configure default recovery strategies
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies"""
        
        self.recovery_strategies = {
            "network_retry": RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                max_attempts=3,
                initial_delay=1.0,
                backoff_multiplier=2.0
            ),
            "quantum_healing": RecoveryAction(
                strategy=RecoveryStrategy.QUANTUM_HEALING,
                max_attempts=2,
                quantum_healing_factor=0.2,
                consciousness_threshold=0.3
            ),
            "consciousness_reset": RecoveryAction(
                strategy=RecoveryStrategy.CONSCIOUSNESS_RESET,
                max_attempts=1,
                consciousness_threshold=0.1
            ),
            "graceful_degradation": RecoveryAction(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_attempts=1
            ),
            "circuit_breaker": RecoveryAction(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                max_attempts=5
            )
        }
    
    def handle_error(self, 
                    exception: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    recovery_strategy: Optional[RecoveryStrategy] = None) -> ErrorContext:
        """Handle error with comprehensive analysis and recovery"""
        
        self.total_errors += 1
        
        # Create error context
        error_context = self._create_error_context(
            exception, context, severity, category, recovery_strategy
        )
        
        # Record error
        self.error_history.append(error_context)
        
        # Analyze error
        analysis = self.error_analyzer.analyze_error(error_context)
        
        # Log error with appropriate level
        self._log_error(error_context, analysis)
        
        # Determine recovery strategy
        if not recovery_strategy:
            recovery_strategy = analysis["recommended_strategy"]
        
        error_context.recovery_strategy = recovery_strategy
        
        # Track critical errors
        if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.QUANTUM_CATASTROPHE]:
            self.critical_errors += 1
        
        return error_context
    
    async def attempt_recovery(self, 
                             error_context: ErrorContext,
                             target_function: Callable,
                             *args, **kwargs) -> Any:
        """Attempt error recovery using specified strategy"""
        
        recovery_action = self._get_recovery_action(error_context.recovery_strategy)
        
        for attempt in range(recovery_action.max_attempts):
            error_context.recovery_attempts = attempt + 1
            
            try:
                if recovery_action.strategy == RecoveryStrategy.RETRY:
                    result = await self._retry_with_backoff(
                        target_function, recovery_action, attempt, *args, **kwargs
                    )
                
                elif recovery_action.strategy == RecoveryStrategy.QUANTUM_HEALING:
                    result = await self._quantum_healing_recovery(
                        target_function, recovery_action, error_context, *args, **kwargs
                    )
                
                elif recovery_action.strategy == RecoveryStrategy.CONSCIOUSNESS_RESET:
                    result = await self._consciousness_reset_recovery(
                        target_function, recovery_action, error_context, *args, **kwargs
                    )
                
                elif recovery_action.strategy == RecoveryStrategy.FALLBACK:
                    result = await self._fallback_recovery(
                        target_function, recovery_action, *args, **kwargs
                    )
                
                elif recovery_action.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    result = await self._graceful_degradation_recovery(
                        target_function, error_context, *args, **kwargs
                    )
                
                else:
                    # Default retry
                    result = await self._retry_with_backoff(
                        target_function, recovery_action, attempt, *args, **kwargs
                    )
                
                # Recovery successful
                self.recovered_errors += 1
                
                if recovery_action.strategy == RecoveryStrategy.QUANTUM_HEALING:
                    self.quantum_healing_successes += 1
                
                logger.info(f"Error recovery successful using {recovery_action.strategy.value} after {attempt + 1} attempts")
                
                return result
            
            except Exception as e:
                logger.warning(f"Recovery attempt {attempt + 1} failed: {e}")
                
                if attempt < recovery_action.max_attempts - 1:
                    # Wait before next attempt
                    delay = recovery_action.initial_delay * (recovery_action.backoff_multiplier ** attempt)
                    await asyncio.sleep(delay)
        
        # All recovery attempts failed
        logger.error(f"All recovery attempts failed for error {error_context.error_id}")
        raise Exception(f"Error recovery failed after {recovery_action.max_attempts} attempts")
    
    async def _retry_with_backoff(self, 
                                target_function: Callable,
                                recovery_action: RecoveryAction,
                                attempt: int,
                                *args, **kwargs) -> Any:
        """Retry with exponential backoff"""
        
        if attempt > 0:
            delay = recovery_action.initial_delay * (recovery_action.backoff_multiplier ** (attempt - 1))
            await asyncio.sleep(delay)
        
        if asyncio.iscoroutinefunction(target_function):
            return await target_function(*args, **kwargs)
        else:
            return target_function(*args, **kwargs)
    
    async def _quantum_healing_recovery(self,
                                      target_function: Callable,
                                      recovery_action: RecoveryAction,
                                      error_context: ErrorContext,
                                      *args, **kwargs) -> Any:
        """Quantum healing recovery mechanism"""
        
        if not self.quantum_healing_enabled:
            raise Exception("Quantum healing is disabled")
        
        # Apply quantum healing
        healing_factor = recovery_action.quantum_healing_factor
        
        # Enhance consciousness if available
        if error_context.consciousness_level is not None:
            enhanced_consciousness = min(1.0, error_context.consciousness_level + healing_factor)
            
            # Update context if possible
            if "consciousness_level" in kwargs:
                kwargs["consciousness_level"] = enhanced_consciousness
            elif error_context.user_context:
                error_context.user_context["consciousness_level"] = enhanced_consciousness
        
        # Apply quantum state healing
        if error_context.quantum_state:
            # Restore quantum coherence
            for key, value in error_context.quantum_state.items():
                if isinstance(value, (int, float)):
                    error_context.quantum_state[key] = min(1.0, value + healing_factor)
        
        logger.info(f"Applied quantum healing with factor {healing_factor}")
        
        # Retry function with healed parameters
        if asyncio.iscoroutinefunction(target_function):
            return await target_function(*args, **kwargs)
        else:
            return target_function(*args, **kwargs)
    
    async def _consciousness_reset_recovery(self,
                                          target_function: Callable,
                                          recovery_action: RecoveryAction,
                                          error_context: ErrorContext,
                                          *args, **kwargs) -> Any:
        """Consciousness reset recovery mechanism"""
        
        if not self.consciousness_recovery_enabled:
            raise Exception("Consciousness recovery is disabled")
        
        # Reset consciousness to baseline
        baseline_consciousness = recovery_action.consciousness_threshold
        
        if "consciousness_level" in kwargs:
            kwargs["consciousness_level"] = baseline_consciousness
        elif error_context.user_context:
            error_context.user_context["consciousness_level"] = baseline_consciousness
        
        logger.info(f"Reset consciousness to baseline {baseline_consciousness}")
        
        # Retry function with reset consciousness
        if asyncio.iscoroutinefunction(target_function):
            return await target_function(*args, **kwargs)
        else:
            return target_function(*args, **kwargs)
    
    async def _fallback_recovery(self,
                               target_function: Callable,
                               recovery_action: RecoveryAction,
                               *args, **kwargs) -> Any:
        """Fallback to alternative function"""
        
        if recovery_action.fallback_function:
            logger.info("Using fallback function for recovery")
            
            if asyncio.iscoroutinefunction(recovery_action.fallback_function):
                return await recovery_action.fallback_function(*args, **kwargs)
            else:
                return recovery_action.fallback_function(*args, **kwargs)
        else:
            # Default fallback - return safe default value
            logger.info("Using default fallback recovery")
            return None
    
    async def _graceful_degradation_recovery(self,
                                           target_function: Callable,
                                           error_context: ErrorContext,
                                           *args, **kwargs) -> Any:
        """Graceful degradation recovery"""
        
        # Reduce complexity or functionality
        simplified_kwargs = kwargs.copy()
        
        # Remove or simplify complex parameters
        if "complexity" in simplified_kwargs:
            simplified_kwargs["complexity"] = 0.1
        
        if "quantum_enhancement" in simplified_kwargs:
            simplified_kwargs["quantum_enhancement"] = False
        
        if "consciousness_required" in simplified_kwargs:
            simplified_kwargs["consciousness_required"] = False
        
        logger.info("Attempting graceful degradation recovery")
        
        if asyncio.iscoroutinefunction(target_function):
            return await target_function(*args, **simplified_kwargs)
        else:
            return target_function(*args, **simplified_kwargs)
    
    def _create_error_context(self,
                            exception: Exception,
                            context: Optional[Dict[str, Any]],
                            severity: ErrorSeverity,
                            category: ErrorCategory,
                            recovery_strategy: Optional[RecoveryStrategy]) -> ErrorContext:
        """Create comprehensive error context"""
        
        import uuid
        import inspect
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Go up two frames to get actual caller
        
        function_name = caller_frame.f_code.co_name if caller_frame else "unknown"
        module_name = caller_frame.f_globals.get("__name__", "unknown") if caller_frame else "unknown"
        
        error_context = ErrorContext(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            exception_type=type(exception).__name__,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            function_name=function_name,
            module_name=module_name,
            recovery_strategy=recovery_strategy
        )
        
        # Add context information
        if context:
            error_context.quantum_state = context.get("quantum_state")
            error_context.consciousness_level = context.get("consciousness_level")
            error_context.user_context = context.get("user_context")
            error_context.correlation_id = context.get("correlation_id")
            error_context.custom_data = context.get("custom_data", {})
            
            # Extract system metrics
            error_context.system_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "memory_usage": context.get("memory_usage", 0),
                "cpu_usage": context.get("cpu_usage", 0),
                "active_sessions": context.get("active_sessions", 0)
            }
        
        return error_context
    
    def _get_recovery_action(self, strategy: RecoveryStrategy) -> RecoveryAction:
        """Get recovery action for strategy"""
        
        strategy_map = {
            RecoveryStrategy.RETRY: "network_retry",
            RecoveryStrategy.QUANTUM_HEALING: "quantum_healing",
            RecoveryStrategy.CONSCIOUSNESS_RESET: "consciousness_reset",
            RecoveryStrategy.GRACEFUL_DEGRADATION: "graceful_degradation",
            RecoveryStrategy.CIRCUIT_BREAKER: "circuit_breaker"
        }
        
        action_name = strategy_map.get(strategy, "network_retry")
        return self.recovery_strategies[action_name]
    
    def _log_error(self, error_context: ErrorContext, analysis: Dict[str, Any]):
        """Log error with appropriate level and formatting"""
        
        log_message = f"[{error_context.error_id}] {error_context.category.value.upper()}: {error_context.message}"
        
        log_data = {
            "error_id": error_context.error_id,
            "category": error_context.category.value,
            "severity": error_context.severity.name,
            "function": error_context.function_name,
            "module": error_context.module_name,
            "frequency": analysis["frequency"],
            "pattern": analysis["pattern_analysis"]["pattern_type"],
            "recommended_strategy": analysis["recommended_strategy"].value
        }
        
        if error_context.severity.log_level == "critical":
            logger.critical(log_message, extra=log_data)
        elif error_context.severity.log_level == "error":
            logger.error(log_message, extra=log_data)
        elif error_context.severity.log_level == "warning":
            logger.warning(log_message, extra=log_data)
        else:
            logger.info(log_message, extra=log_data)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics"""
        
        recent_errors = [
            error for error in self.error_history
            if (datetime.utcnow() - error.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        category_stats = {}
        severity_stats = {}
        
        for error in recent_errors:
            category_stats[error.category.value] = category_stats.get(error.category.value, 0) + 1
            severity_stats[error.severity.name] = severity_stats.get(error.severity.name, 0) + 1
        
        recovery_rate = self.recovered_errors / max(1, self.total_errors)
        
        return {
            "total_errors": self.total_errors,
            "recovered_errors": self.recovered_errors,
            "critical_errors": self.critical_errors,
            "quantum_healing_successes": self.quantum_healing_successes,
            "recovery_rate": recovery_rate,
            "recent_errors_1h": len(recent_errors),
            "error_categories": category_stats,
            "error_severities": severity_stats,
            "circuit_breakers_active": len(self.circuit_breakers),
            "quantum_healing_enabled": self.quantum_healing_enabled,
            "consciousness_recovery_enabled": self.consciousness_recovery_enabled,
            "top_error_patterns": [
                {
                    "signature": sig,
                    "frequency": freq
                }
                for sig, freq in sorted(
                    self.error_analyzer.error_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ]
        }


# Decorator for robust error handling
def robust_error_handler(
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    recovery_strategy: Optional[RecoveryStrategy] = None,
    max_retries: int = 3,
    circuit_breaker: bool = False
):
    """Decorator for robust error handling with automatic recovery"""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        
        # Add circuit breaker if requested
        if circuit_breaker:
            func = QuantumCircuitBreaker()(func)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            handler = RobustErrorHandler()
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Handle error
                    context = {
                        "quantum_state": kwargs.get("quantum_state"),
                        "consciousness_level": kwargs.get("consciousness_level"),
                        "user_context": kwargs.get("user_context"),
                        "attempt": attempt + 1,
                        "max_attempts": max_retries
                    }
                    
                    error_context = handler.handle_error(e, context, severity, category, recovery_strategy)
                    
                    if attempt < max_retries - 1:
                        # Attempt recovery
                        try:
                            return await handler.attempt_recovery(error_context, func, *args, **kwargs)
                        except Exception:
                            continue  # Try next attempt
                    else:
                        # Final attempt failed
                        raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            handler = RobustErrorHandler()
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    context = {
                        "attempt": attempt + 1,
                        "max_attempts": max_retries
                    }
                    
                    error_context = handler.handle_error(e, context, severity, category, recovery_strategy)
                    
                    if attempt < max_retries - 1:
                        # Simple retry for sync functions
                        time.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global error handler instance
global_error_handler = RobustErrorHandler()


# Export main components
__all__ = [
    "RobustErrorHandler",
    "ErrorContext",
    "ErrorSeverity",
    "ErrorCategory", 
    "RecoveryStrategy",
    "RecoveryAction",
    "QuantumCircuitBreaker",
    "ErrorAnalyzer",
    "robust_error_handler",
    "global_error_handler"
]