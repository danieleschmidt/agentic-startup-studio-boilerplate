"""
Robust Quantum Task Implementation with Enhanced Error Handling and Validation
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field, validator

from .quantum_task import QuantumTask, TaskState, TaskPriority
from ..utils.simple_validation import (
    validate_task_creation_input, 
    TaskValidationError,
    SecurityValidator,
    InputSanitizer
)
from ..utils.simple_error_handling import (
    safe_quantum_operation,
    QuantumTaskError,
    TaskNotFoundError,
    ErrorHandler
)


class RobustQuantumTask(QuantumTask):
    """Enhanced quantum task with robust error handling and validation"""
    
    # Enhanced validation
    @validator('title')
    def validate_title(cls, v):
        """Validate task title"""
        if not v or not v.strip():
            raise ValueError("Task title cannot be empty")
        
        return SecurityValidator.validate_safe_content(v, 'title')
    
    @validator('description')
    def validate_description(cls, v):
        """Validate task description"""
        if not v or not v.strip():
            raise ValueError("Task description cannot be empty")
        
        return SecurityValidator.validate_safe_content(v, 'description')
    
    @validator('task_id')
    def validate_task_id(cls, v):
        """Validate task ID format"""
        return InputSanitizer.validate_task_id(v)
    
    @validator('quantum_coherence')
    def validate_quantum_coherence(cls, v):
        """Validate quantum coherence value"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quantum coherence must be between 0.0 and 1.0")
        return v
    
    @validator('success_probability')
    def validate_success_probability(cls, v):
        """Validate success probability"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Success probability must be between 0.0 and 1.0")
        return v
    
    @safe_quantum_operation
    def get_completion_probability(self) -> float:
        """Calculate task completion probability with error handling"""
        try:
            # Base probability from success_probability
            base_prob = self.success_probability
            
            # Apply quantum coherence factor
            coherence_factor = self.quantum_coherence * 0.1
            
            # Apply priority factor
            priority_factor = self.priority.probability_weight * 0.2
            
            # Calculate final probability
            completion_prob = min(1.0, base_prob + coherence_factor + priority_factor)
            
            return completion_prob
            
        except Exception as e:
            # Fallback to a reasonable default
            return 0.75
    
    @safe_quantum_operation
    def get_superposition_probability(self) -> float:
        """Calculate superposition probability with enhanced error handling"""
        try:
            if self.state == TaskState.SUPERPOSITION:
                return 1.0
            elif self.state in [TaskState.COMPLETED, TaskState.FAILED]:
                return 0.0
            else:
                # Calculate based on quantum coherence and current state
                base_superposition = self.quantum_coherence * 0.5
                
                # State-specific adjustments
                state_adjustments = {
                    TaskState.PENDING: 0.8,
                    TaskState.IN_PROGRESS: 0.6,
                    TaskState.RUNNING: 0.6,
                    TaskState.BLOCKED: 0.3,
                    TaskState.PAUSED: 0.4,
                }
                
                adjustment = state_adjustments.get(self.state, 0.5)
                return min(1.0, base_superposition * adjustment)
                
        except Exception:
            # Safe fallback
            return 0.5
    
    @safe_quantum_operation
    def transition_state(self, new_state: TaskState, measurement_probability: float = 1.0) -> bool:
        """Safely transition to a new state with validation"""
        try:
            if not isinstance(new_state, TaskState):
                raise QuantumTaskError(f"Invalid state type: {type(new_state)}")
            
            if not 0.0 <= measurement_probability <= 1.0:
                raise QuantumTaskError("Measurement probability must be between 0.0 and 1.0")
            
            # Validate state transition
            valid_transitions = self._get_valid_transitions()
            if new_state not in valid_transitions:
                raise QuantumTaskError(f"Invalid transition from {self.state} to {new_state}")
            
            # Record measurement
            measurement_time = datetime.utcnow()
            self.measurement_history.append((measurement_time, new_state, measurement_probability))
            
            # Update state
            old_state = self.state
            self.state = new_state
            
            # Adjust quantum coherence based on measurement
            coherence_decay = (1.0 - measurement_probability) * 0.1
            self.quantum_coherence = max(0.0, self.quantum_coherence - coherence_decay)
            
            return True
            
        except Exception as e:
            raise QuantumTaskError(f"State transition failed: {e}")
    
    def _get_valid_transitions(self) -> Set[TaskState]:
        """Get valid state transitions from current state"""
        transition_map = {
            TaskState.PENDING: {
                TaskState.IN_PROGRESS, TaskState.RUNNING, TaskState.BLOCKED, 
                TaskState.CANCELLED, TaskState.DEFERRED
            },
            TaskState.IN_PROGRESS: {
                TaskState.COMPLETED, TaskState.FAILED, TaskState.BLOCKED, 
                TaskState.PAUSED, TaskState.CANCELLED
            },
            TaskState.RUNNING: {
                TaskState.COMPLETED, TaskState.FAILED, TaskState.BLOCKED, 
                TaskState.PAUSED, TaskState.CANCELLED
            },
            TaskState.BLOCKED: {
                TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.CANCELLED, TaskState.FAILED
            },
            TaskState.PAUSED: {
                TaskState.IN_PROGRESS, TaskState.RUNNING, TaskState.CANCELLED
            },
            TaskState.DEFERRED: {
                TaskState.PENDING, TaskState.CANCELLED
            },
            TaskState.SUPERPOSITION: {
                TaskState.PENDING, TaskState.IN_PROGRESS, TaskState.RUNNING,
                TaskState.COMPLETED, TaskState.FAILED
            }
        }
        
        return transition_map.get(self.state, set())
    
    @ErrorHandler.handle_gracefully(default_return={})
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get task health and performance metrics"""
        return {
            "task_id": self.task_id,
            "state": self.state.value,
            "quantum_coherence": self.quantum_coherence,
            "completion_probability": self.get_completion_probability(),
            "superposition_probability": self.get_superposition_probability(),
            "measurement_count": len(self.measurement_history),
            "priority_weight": self.priority.probability_weight,
            "created_at": self.created_at.isoformat(),
            "is_healthy": self.quantum_coherence > 0.3 and self.state != TaskState.FAILED,
            "performance_score": self._calculate_performance_score()
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall task performance score"""
        try:
            coherence_score = self.quantum_coherence * 0.3
            completion_score = self.get_completion_probability() * 0.4
            priority_score = self.priority.probability_weight * 0.2
            
            # State-based score
            state_scores = {
                TaskState.COMPLETED: 1.0,
                TaskState.IN_PROGRESS: 0.7,
                TaskState.RUNNING: 0.7,
                TaskState.PENDING: 0.5,
                TaskState.BLOCKED: 0.2,
                TaskState.FAILED: 0.0,
                TaskState.CANCELLED: 0.0
            }
            state_score = state_scores.get(self.state, 0.3) * 0.1
            
            return coherence_score + completion_score + priority_score + state_score
            
        except Exception:
            return 0.5  # Safe fallback


class RobustTaskManager:
    """Enhanced task manager with error handling and validation"""
    
    def __init__(self):
        self.tasks: Dict[str, RobustQuantumTask] = {}
        self.error_history: List[Dict[str, Any]] = []
    
    @safe_quantum_operation
    def create_task(self, task_data: Dict[str, Any]) -> RobustQuantumTask:
        """Create a new task with comprehensive validation"""
        try:
            # Validate input data
            validated_data = validate_task_creation_input(task_data)
            
            # Create task
            task = RobustQuantumTask(**validated_data)
            
            # Store task
            self.tasks[task.task_id] = task
            
            return task
            
        except TaskValidationError as e:
            self._record_error("validation_error", str(e), task_data)
            raise e
        except Exception as e:
            self._record_error("creation_error", str(e), task_data)
            raise QuantumTaskError(f"Task creation failed: {e}")
    
    @safe_quantum_operation
    def get_task(self, task_id: str) -> RobustQuantumTask:
        """Get task with error handling"""
        if task_id not in self.tasks:
            raise TaskNotFoundError(f"Task not found: {task_id}")
        
        return self.tasks[task_id]
    
    @safe_quantum_operation
    def update_task_state(self, task_id: str, new_state: TaskState) -> bool:
        """Update task state with validation"""
        task = self.get_task(task_id)
        return task.transition_state(new_state)
    
    @ErrorHandler.handle_gracefully(default_return=[])
    def get_all_tasks(self) -> List[RobustQuantumTask]:
        """Get all tasks safely"""
        return list(self.tasks.values())
    
    @ErrorHandler.handle_gracefully(default_return={})
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        if not self.tasks:
            return {
                "healthy": True,
                "total_tasks": 0,
                "message": "No tasks in system"
            }
        
        task_metrics = [task.get_health_metrics() for task in self.tasks.values()]
        healthy_tasks = sum(1 for metrics in task_metrics if metrics.get("is_healthy", False))
        
        avg_coherence = np.mean([metrics.get("quantum_coherence", 0) for metrics in task_metrics])
        avg_performance = np.mean([metrics.get("performance_score", 0) for metrics in task_metrics])
        
        return {
            "healthy": len(self.error_history) < 10 and avg_coherence > 0.3,
            "total_tasks": len(self.tasks),
            "healthy_tasks": healthy_tasks,
            "unhealthy_tasks": len(self.tasks) - healthy_tasks,
            "average_coherence": float(avg_coherence),
            "average_performance": float(avg_performance),
            "error_count": len(self.error_history),
            "recent_errors": self.error_history[-5:] if self.error_history else []
        }
    
    def _record_error(self, error_type: str, error_message: str, context: Any = None):
        """Record error for debugging and monitoring"""
        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": str(context) if context else None
        }
        
        self.error_history.append(error_record)
        
        # Keep only recent errors (last 100)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]