"""
Quantum Task Representation

Implements quantum-inspired task models with superposition states,
probability amplitudes, and quantum measurement capabilities.
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel, Field


class TaskState(Enum):
    """Quantum task states representing superposition possibilities"""
    SUPERPOSITION = "superposition"  # Task exists in multiple states
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    RUNNING = "running"  # Alias for in_progress
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"
    PAUSED = "paused"
    FAILED = "failed"


class TaskPriority(Enum):
    """Task priority levels with quantum probability weights"""
    CRITICAL = ("critical", 0.95)
    HIGH = ("high", 0.80)
    MEDIUM = ("medium", 0.60)
    LOW = ("low", 0.40)
    MINIMAL = ("minimal", 0.20)
    
    def __init__(self, name: str, probability_weight: float):
        self.probability_weight = probability_weight


@dataclass
class QuantumAmplitude:
    """Represents quantum probability amplitude for task states"""
    state: TaskState
    amplitude: complex
    probability: float = field(init=False)
    
    def __post_init__(self):
        self.probability = abs(self.amplitude) ** 2
    
    @property
    def phase(self) -> float:
        """Extract phase information from complex amplitude"""
        return np.angle(self.amplitude)


@dataclass 
class TaskResource:
    """Resource requirements with quantum uncertainty"""
    resource_type: str
    min_required: float
    max_required: float
    uncertainty_factor: float = 0.1
    
    def get_expected_requirement(self) -> float:
        """Calculate expected resource requirement accounting for uncertainty"""
        return (self.min_required + self.max_required) / 2 * (1 + self.uncertainty_factor)


class QuantumTask(BaseModel):
    """
    Quantum-inspired task representation with superposition states,
    entanglement capabilities, and probability-based optimization.
    """
    
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    
    # Quantum properties
    state_amplitudes: Dict[TaskState, QuantumAmplitude] = Field(default_factory=dict)
    quantum_coherence: float = Field(default=1.0, ge=0.0, le=1.0)
    entangled_tasks: Set[str] = Field(default_factory=set)
    measurement_history: List[Tuple[datetime, TaskState, float]] = Field(default_factory=list)
    
    # Traditional properties enhanced with quantum concepts
    priority: TaskPriority = TaskPriority.MEDIUM
    tags: List[str] = Field(default_factory=list)
    resources: List[TaskResource] = Field(default_factory=list)
    dependencies: Set[str] = Field(default_factory=set)
    assignee: Optional[str] = None
    
    # Performance metrics
    success_probability: float = Field(default=0.8, ge=0.0, le=1.0)
    complexity_factor: float = Field(default=1.0, ge=0.1, le=10.0)
    
    # Current state (for API compatibility)
    state: TaskState = TaskState.PENDING
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds(),
            complex: lambda v: {"real": v.real, "imag": v.imag},
            np.ndarray: lambda v: v.tolist()
        }
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.state_amplitudes:
            self._initialize_superposition()
    
    def _initialize_superposition(self):
        """Initialize task in quantum superposition of all possible states"""
        n_states = len(TaskState) - 1  # Exclude SUPERPOSITION itself
        equal_probability = 1.0 / n_states
        
        for state in TaskState:
            if state != TaskState.SUPERPOSITION:
                amplitude = np.sqrt(equal_probability) * np.exp(1j * np.random.uniform(0, 2*np.pi))
                self.state_amplitudes[state] = QuantumAmplitude(state, amplitude)
        
        self._normalize_amplitudes()
    
    def _normalize_amplitudes(self):
        """Ensure quantum state amplitudes are properly normalized"""
        total_probability = sum(amp.probability for amp in self.state_amplitudes.values())
        if total_probability > 0:
            normalization_factor = np.sqrt(1.0 / total_probability)
            for state_amp in self.state_amplitudes.values():
                state_amp.amplitude *= normalization_factor
                state_amp.probability = abs(state_amp.amplitude) ** 2
    
    def measure_state(self, observer_effect: float = 0.1) -> TaskState:
        """
        Perform quantum measurement to collapse superposition into definite state
        
        Args:
            observer_effect: Degree to which measurement affects the system (0-1)
        
        Returns:
            Measured task state
        """
        if not self.state_amplitudes:
            return TaskState.PENDING
        
        # Calculate probabilities for measurement
        probabilities = [amp.probability for amp in self.state_amplitudes.values()]
        states = list(self.state_amplitudes.keys())
        
        # Perform quantum measurement
        measured_state = np.random.choice(states, p=probabilities)
        
        # Apply observer effect - reduce coherence
        self.quantum_coherence *= (1.0 - observer_effect)
        
        # Record measurement
        self.measurement_history.append((
            datetime.utcnow(),
            measured_state,
            self.state_amplitudes[measured_state].probability
        ))
        
        return measured_state
    
    def entangle_with(self, other_task: 'QuantumTask', entanglement_strength: float = 0.5):
        """
        Create quantum entanglement between tasks
        
        Args:
            other_task: Task to entangle with
            entanglement_strength: Strength of entanglement (0-1)
        """
        self.entangled_tasks.add(other_task.task_id)
        other_task.entangled_tasks.add(self.task_id)
        
        # Synchronize quantum states based on entanglement
        self._apply_entanglement_effects(other_task, entanglement_strength)
    
    def _apply_entanglement_effects(self, other_task: 'QuantumTask', strength: float):
        """Apply quantum entanglement effects between tasks"""
        for state in self.state_amplitudes:
            if state in other_task.state_amplitudes:
                # Create correlated amplitudes
                self_amp = self.state_amplitudes[state].amplitude
                other_amp = other_task.state_amplitudes[state].amplitude
                
                # Apply entanglement correlation
                correlation_factor = strength * np.exp(1j * np.pi / 4)
                
                new_self_amp = (1 - strength) * self_amp + strength * other_amp * correlation_factor
                new_other_amp = (1 - strength) * other_amp + strength * self_amp * np.conj(correlation_factor)
                
                self.state_amplitudes[state] = QuantumAmplitude(state, new_self_amp)
                other_task.state_amplitudes[state] = QuantumAmplitude(state, new_other_amp)
        
        # Renormalize both tasks
        self._normalize_amplitudes()
        other_task._normalize_amplitudes()
    
    def get_completion_probability(self) -> float:
        """Calculate probability of successful task completion"""
        if TaskState.COMPLETED in self.state_amplitudes:
            base_prob = self.state_amplitudes[TaskState.COMPLETED].probability
        else:
            base_prob = 0.5
        
        # Factor in task properties
        priority_boost = self.priority.probability_weight * 0.2
        complexity_penalty = (self.complexity_factor - 1.0) * 0.1
        coherence_bonus = self.quantum_coherence * 0.1
        
        final_probability = base_prob + priority_boost - complexity_penalty + coherence_bonus
        return max(0.0, min(1.0, final_probability))
    
    def update_state_probability(self, state: TaskState, new_probability: float):
        """Update the probability amplitude for a specific state"""
        if 0 <= new_probability <= 1:
            # Calculate new amplitude maintaining phase
            if state in self.state_amplitudes:
                old_phase = self.state_amplitudes[state].phase
                new_amplitude = np.sqrt(new_probability) * np.exp(1j * old_phase)
            else:
                new_amplitude = np.sqrt(new_probability) * np.exp(1j * np.random.uniform(0, 2*np.pi))
            
            self.state_amplitudes[state] = QuantumAmplitude(state, new_amplitude)
            self._normalize_amplitudes()
    
    def get_quantum_state_vector(self) -> np.ndarray:
        """Get quantum state vector representation"""
        states = list(TaskState)
        state_vector = np.zeros(len(states), dtype=complex)
        
        for i, state in enumerate(states):
            if state in self.state_amplitudes:
                state_vector[i] = self.state_amplitudes[state].amplitude
        
        return state_vector
    
    def calculate_expected_completion_time(self) -> Optional[datetime]:
        """Calculate expected completion time using quantum probability"""
        if not self.estimated_duration:
            return None
        
        completion_prob = self.get_completion_probability()
        uncertainty_factor = 1.0 / self.quantum_coherence if self.quantum_coherence > 0 else 2.0
        
        expected_duration = self.estimated_duration * uncertainty_factor / completion_prob
        return self.created_at + expected_duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None,
            "priority": self.priority.name,
            "quantum_coherence": self.quantum_coherence,
            "success_probability": self.success_probability,
            "complexity_factor": self.complexity_factor,
            "completion_probability": self.get_completion_probability(),
            "entangled_tasks": list(self.entangled_tasks),
            "state_probabilities": {
                state.value: amp.probability 
                for state, amp in self.state_amplitudes.items()
            },
            "current_state": self.state.value
        }
    
    def start_execution(self):
        """Start task execution"""
        self.state = TaskState.IN_PROGRESS
    
    def complete_execution(self):
        """Complete task execution"""
        self.state = TaskState.COMPLETED
    
    def set_state(self, new_state: TaskState):
        """Set task state"""
        self.state = new_state
    
    def _update_completion_probability(self, progress: float):
        """Update completion probability based on progress"""
        self.success_probability = min(1.0, self.success_probability + progress * 0.1)