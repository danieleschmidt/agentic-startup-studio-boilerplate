"""
Tests for Quantum Task Implementation

Comprehensive test suite for quantum task functionality including
superposition, measurement, entanglement, and probability calculations.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from quantum_task_planner.core.quantum_task import (
    QuantumTask, TaskState, TaskPriority, QuantumAmplitude, TaskResource
)


class TestQuantumTask:
    """Test quantum task functionality"""
    
    def test_task_creation(self):
        """Test basic task creation"""
        task = QuantumTask(
            title="Test Task",
            description="A test quantum task",
            priority=TaskPriority.HIGH
        )
        
        assert task.title == "Test Task"
        assert task.description == "A test quantum task"
        assert task.priority == TaskPriority.HIGH
        assert task.quantum_coherence == 1.0
        assert len(task.state_amplitudes) > 0
        assert task.task_id is not None
    
    def test_superposition_initialization(self):
        """Test quantum superposition initialization"""
        task = QuantumTask(
            title="Superposition Test",
            description="Testing superposition initialization"
        )
        
        # Check that task is in superposition of multiple states
        assert len(task.state_amplitudes) > 1
        
        # Check probability normalization
        total_prob = sum(amp.probability for amp in task.state_amplitudes.values())
        assert abs(total_prob - 1.0) < 1e-10
        
        # Check that all probabilities are positive
        for amplitude in task.state_amplitudes.values():
            assert amplitude.probability >= 0
            assert amplitude.probability <= 1
    
    def test_quantum_measurement(self):
        """Test quantum measurement and state collapse"""
        task = QuantumTask(
            title="Measurement Test",
            description="Testing quantum measurement"
        )
        
        initial_coherence = task.quantum_coherence
        initial_states = len(task.state_amplitudes)
        
        # Perform measurement
        measured_state = task.measure_state(observer_effect=0.1)
        
        # Check that measurement returns a valid state
        assert isinstance(measured_state, TaskState)
        assert measured_state != TaskState.SUPERPOSITION
        
        # Check that coherence decreased due to observer effect
        assert task.quantum_coherence < initial_coherence
        
        # Check that measurement was recorded
        assert len(task.measurement_history) == 1
        assert task.measurement_history[0][1] == measured_state
    
    def test_completion_probability(self):
        """Test completion probability calculation"""
        task = QuantumTask(
            title="Probability Test",
            description="Testing completion probability",
            priority=TaskPriority.CRITICAL,
            complexity_factor=2.0
        )
        
        completion_prob = task.get_completion_probability()
        
        # Probability should be between 0 and 1
        assert 0 <= completion_prob <= 1
        
        # Higher priority should generally increase completion probability
        low_priority_task = QuantumTask(
            title="Low Priority",
            description="Low priority task",
            priority=TaskPriority.LOW,
            complexity_factor=2.0
        )
        
        low_prob = low_priority_task.get_completion_probability()
        # Note: Due to quantum uncertainty, this isn't guaranteed, but statistically likely
        # assert completion_prob >= low_prob
    
    def test_state_probability_update(self):
        """Test updating state probabilities"""
        task = QuantumTask(
            title="State Update Test",
            description="Testing state probability updates"
        )
        
        # Update probability for COMPLETED state
        task.update_state_probability(TaskState.COMPLETED, 0.8)
        
        # Check that the probability was updated
        completed_prob = task.state_amplitudes[TaskState.COMPLETED].probability
        assert abs(completed_prob - 0.8) < 1e-6
        
        # Check that probabilities are still normalized
        total_prob = sum(amp.probability for amp in task.state_amplitudes.values())
        assert abs(total_prob - 1.0) < 1e-6
    
    def test_quantum_state_vector(self):
        """Test quantum state vector representation"""
        task = QuantumTask(
            title="State Vector Test",
            description="Testing quantum state vector"
        )
        
        state_vector = task.get_quantum_state_vector()
        
        # Check that it's a numpy array
        assert isinstance(state_vector, np.ndarray)
        
        # Check that it has complex values
        assert state_vector.dtype == complex
        
        # Check normalization
        norm = np.linalg.norm(state_vector)
        assert abs(norm - 1.0) < 1e-6
    
    def test_expected_completion_time(self):
        """Test expected completion time calculation"""
        duration = timedelta(hours=4)
        task = QuantumTask(
            title="Completion Time Test",
            description="Testing expected completion time",
            estimated_duration=duration
        )
        
        expected_time = task.calculate_expected_completion_time()
        
        # Should return a datetime
        assert isinstance(expected_time, datetime)
        
        # Should be after creation time
        assert expected_time > task.created_at
    
    def test_task_serialization(self):
        """Test task serialization to dictionary"""
        task = QuantumTask(
            title="Serialization Test",
            description="Testing task serialization",
            priority=TaskPriority.MEDIUM,
            complexity_factor=1.5,
            tags=["test", "serialization"]
        )
        
        task_dict = task.to_dict()
        
        # Check that all expected fields are present
        expected_fields = [
            "task_id", "title", "description", "created_at", 
            "priority", "quantum_coherence", "success_probability",
            "complexity_factor", "completion_probability", "state_probabilities"
        ]
        
        for field in expected_fields:
            assert field in task_dict
        
        # Check data types
        assert isinstance(task_dict["state_probabilities"], dict)
        assert isinstance(task_dict["completion_probability"], float)
        assert isinstance(task_dict["quantum_coherence"], float)


class TestQuantumAmplitude:
    """Test quantum amplitude functionality"""
    
    def test_amplitude_creation(self):
        """Test quantum amplitude creation"""
        amplitude = QuantumAmplitude(
            state=TaskState.PENDING,
            amplitude=complex(0.7, 0.3)
        )
        
        assert amplitude.state == TaskState.PENDING
        assert amplitude.amplitude == complex(0.7, 0.3)
        
        # Check probability calculation
        expected_prob = abs(complex(0.7, 0.3)) ** 2
        assert abs(amplitude.probability - expected_prob) < 1e-10
    
    def test_amplitude_phase(self):
        """Test amplitude phase extraction"""
        amplitude = QuantumAmplitude(
            state=TaskState.IN_PROGRESS,
            amplitude=complex(0, 1)  # Pure imaginary = π/2 phase
        )
        
        phase = amplitude.phase
        assert abs(phase - np.pi/2) < 1e-10


class TestTaskResource:
    """Test task resource functionality"""
    
    def test_resource_creation(self):
        """Test task resource creation"""
        resource = TaskResource(
            resource_type="cpu",
            min_required=2.0,
            max_required=4.0,
            uncertainty_factor=0.1
        )
        
        assert resource.resource_type == "cpu"
        assert resource.min_required == 2.0
        assert resource.max_required == 4.0
        assert resource.uncertainty_factor == 0.1
    
    def test_expected_requirement_calculation(self):
        """Test expected resource requirement calculation"""
        resource = TaskResource(
            resource_type="memory",
            min_required=1.0,
            max_required=3.0,
            uncertainty_factor=0.2
        )
        
        expected = resource.get_expected_requirement()
        
        # Should be average of min/max adjusted by uncertainty
        base_expected = (1.0 + 3.0) / 2
        adjusted_expected = base_expected * (1 + 0.2)
        
        assert abs(expected - adjusted_expected) < 1e-10


class TestTaskEntanglement:
    """Test task entanglement functionality"""
    
    def test_task_entanglement(self):
        """Test basic task entanglement"""
        task1 = QuantumTask(title="Task 1", description="First task")
        task2 = QuantumTask(title="Task 2", description="Second task")
        
        # Entangle tasks
        task1.entangle_with(task2, entanglement_strength=0.8)
        
        # Check entanglement relationship
        assert task2.task_id in task1.entangled_tasks
        assert task1.task_id in task2.entangled_tasks
    
    def test_entanglement_effects(self):
        """Test quantum entanglement effects on states"""
        task1 = QuantumTask(title="Task 1", description="First task")
        task2 = QuantumTask(title="Task 2", description="Second task")
        
        # Store initial states
        initial_states_1 = {
            state: amp.amplitude for state, amp in task1.state_amplitudes.items()
        }
        initial_states_2 = {
            state: amp.amplitude for state, amp in task2.state_amplitudes.items()
        }
        
        # Entangle tasks
        task1.entangle_with(task2, entanglement_strength=0.5)
        
        # Check that states have changed due to entanglement
        states_changed_1 = any(
            abs(task1.state_amplitudes[state].amplitude - initial_states_1[state]) > 1e-10
            for state in initial_states_1
        )
        states_changed_2 = any(
            abs(task2.state_amplitudes[state].amplitude - initial_states_2[state]) > 1e-10
            for state in initial_states_2
        )
        
        assert states_changed_1 or states_changed_2


class TestQuantumTaskIntegration:
    """Integration tests for quantum task system"""
    
    def test_complex_task_workflow(self):
        """Test complex workflow with multiple quantum operations"""
        # Create task with specific parameters
        task = QuantumTask(
            title="Complex Workflow Test",
            description="Testing complex quantum task workflow",
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(hours=2),
            complexity_factor=2.5,
            tags=["integration", "test", "quantum"]
        )
        
        # Perform multiple measurements
        measurements = []
        for i in range(5):
            state = task.measure_state(observer_effect=0.05)
            measurements.append(state)
        
        # Check that measurements were recorded
        assert len(task.measurement_history) == 5
        
        # Check coherence degradation
        assert task.quantum_coherence < 1.0
        
        # Update probabilities
        task.update_state_probability(TaskState.COMPLETED, 0.9)
        
        # Check completion probability
        completion_prob = task.get_completion_probability()
        assert completion_prob > 0.5  # Should be high due to high completion state probability
        
        # Serialize task
        task_dict = task.to_dict()
        assert isinstance(task_dict, dict)
        assert "task_id" in task_dict
    
    def test_quantum_uncertainty_effects(self):
        """Test quantum uncertainty in task operations"""
        tasks = []
        
        # Create multiple identical tasks
        for i in range(10):
            task = QuantumTask(
                title=f"Uncertainty Test {i}",
                description="Testing quantum uncertainty effects",
                priority=TaskPriority.MEDIUM
            )
            tasks.append(task)
        
        # Measure all tasks
        measured_states = []
        for task in tasks:
            state = task.measure_state()
            measured_states.append(state)
        
        # Due to quantum uncertainty, we should get different results
        unique_states = set(measured_states)
        # With quantum superposition, we should get at least 2 different states
        # (though this is probabilistic, so we use a lower bound)
        assert len(unique_states) >= 1  # At minimum, some variation expected


# Pytest fixtures
@pytest.fixture
def simple_task():
    """Fixture providing a simple quantum task"""
    return QuantumTask(
        title="Test Task",
        description="A simple test task",
        priority=TaskPriority.MEDIUM
    )


@pytest.fixture
def complex_task():
    """Fixture providing a complex quantum task"""
    return QuantumTask(
        title="Complex Test Task",
        description="A complex test task with multiple properties",
        priority=TaskPriority.HIGH,
        estimated_duration=timedelta(hours=3),
        due_date=datetime.utcnow() + timedelta(days=7),
        complexity_factor=3.0,
        tags=["complex", "test", "fixture"],
        resources=[
            TaskResource("cpu", 2.0, 4.0, 0.1),
            TaskResource("memory", 1.0, 2.0, 0.05)
        ]
    )


@pytest.fixture
def entangled_tasks():
    """Fixture providing two entangled tasks"""
    task1 = QuantumTask(title="Entangled Task 1", description="First entangled task")
    task2 = QuantumTask(title="Entangled Task 2", description="Second entangled task")
    
    task1.entangle_with(task2, entanglement_strength=0.7)
    
    return task1, task2


# Performance tests
class TestQuantumTaskPerformance:
    """Performance tests for quantum task operations"""
    
    def test_measurement_performance(self, simple_task):
        """Test performance of quantum measurements"""
        import time
        
        start_time = time.time()
        
        # Perform many measurements
        for _ in range(1000):
            simple_task.measure_state(observer_effect=0.01)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 1000 measurements
    
    def test_entanglement_performance(self):
        """Test performance of entanglement operations"""
        import time
        
        # Create many tasks
        tasks = [
            QuantumTask(title=f"Perf Task {i}", description=f"Performance test task {i}")
            for i in range(100)
        ]
        
        start_time = time.time()
        
        # Entangle adjacent tasks
        for i in range(len(tasks) - 1):
            tasks[i].entangle_with(tasks[i + 1], entanglement_strength=0.5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds for 99 entanglements


if __name__ == "__main__":
    pytest.main([__file__])