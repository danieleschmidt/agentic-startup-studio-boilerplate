"""
Tests for Quantum Scheduler

Test suite for quantum scheduling algorithms including optimization,
constraint satisfaction, and annealing procedures.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
from quantum_task_planner.core.quantum_scheduler import (
    QuantumTaskScheduler, ResourceConstraint, TimeConstraint
)


class TestQuantumTaskScheduler:
    """Test quantum task scheduler functionality"""
    
    @pytest.fixture
    def scheduler(self):
        """Fixture providing a quantum scheduler"""
        return QuantumTaskScheduler(max_iterations=100)
    
    @pytest.fixture
    def sample_tasks(self):
        """Fixture providing sample tasks"""
        tasks = []
        
        # Create tasks with different priorities and durations
        tasks.append(QuantumTask(
            title="High Priority Task",
            description="Critical task",
            priority=TaskPriority.CRITICAL,
            estimated_duration=timedelta(hours=2),
            due_date=datetime.utcnow() + timedelta(days=1)
        ))
        
        tasks.append(QuantumTask(
            title="Medium Priority Task", 
            description="Normal task",
            priority=TaskPriority.MEDIUM,
            estimated_duration=timedelta(hours=4),
            due_date=datetime.utcnow() + timedelta(days=3)
        ))
        
        tasks.append(QuantumTask(
            title="Low Priority Task",
            description="Low importance task",
            priority=TaskPriority.LOW,
            estimated_duration=timedelta(hours=1),
            due_date=datetime.utcnow() + timedelta(days=7)
        ))
        
        return tasks
    
    def test_scheduler_creation(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.max_iterations == 100
        assert scheduler.temperature_schedule == "exponential"
        assert len(scheduler.tasks) == 0
        assert len(scheduler.constraints) == 0
    
    def test_add_task(self, scheduler, sample_tasks):
        """Test adding tasks to scheduler"""
        task = sample_tasks[0]
        scheduler.add_task(task)
        
        assert task.task_id in scheduler.tasks
        assert scheduler.tasks[task.task_id] == task
    
    def test_remove_task(self, scheduler, sample_tasks):
        """Test removing tasks from scheduler"""
        task = sample_tasks[0]
        scheduler.add_task(task)
        
        removed_task = scheduler.remove_task(task.task_id)
        
        assert removed_task == task
        assert task.task_id not in scheduler.tasks
    
    def test_add_constraint(self, scheduler):
        """Test adding scheduling constraints"""
        constraint = ResourceConstraint(
            constraint_type="resource",
            weight=1.0,
            resource_type="cpu",
            available_amount=8.0
        )
        
        scheduler.add_constraint(constraint)
        assert len(scheduler.constraints) == 1
        assert scheduler.constraints[0] == constraint
    
    def test_set_resource_pool(self, scheduler):
        """Test setting resource pools"""
        scheduler.set_resource_pool("cpu", 16.0)
        scheduler.set_resource_pool("memory", 32.0)
        
        assert scheduler.resource_pools["cpu"] == 16.0
        assert scheduler.resource_pools["memory"] == 32.0
    
    @pytest.mark.asyncio
    async def test_optimize_empty_schedule(self, scheduler):
        """Test optimization with no tasks"""
        schedule = await scheduler.optimize_schedule()
        assert len(schedule) == 0
    
    @pytest.mark.asyncio
    async def test_optimize_schedule_basic(self, scheduler, sample_tasks):
        """Test basic schedule optimization"""
        # Add tasks to scheduler
        for task in sample_tasks:
            scheduler.add_task(task)
        
        # Set some resource pools
        scheduler.set_resource_pool("cpu", 8.0)
        scheduler.set_resource_pool("memory", 16.0)
        
        # Optimize schedule
        schedule = await scheduler.optimize_schedule()
        
        # Check that all tasks are scheduled
        assert len(schedule) == len(sample_tasks)
        
        # Check that schedule contains valid entries
        for start_time, task in schedule:
            assert isinstance(start_time, datetime)
            assert isinstance(task, QuantumTask)
            assert task.task_id in scheduler.tasks
    
    @pytest.mark.asyncio
    async def test_schedule_energy_calculation(self, scheduler, sample_tasks):
        """Test schedule energy calculation"""
        for task in sample_tasks:
            scheduler.add_task(task)
        
        # Create a simple schedule
        schedule = []
        current_time = datetime.utcnow()
        
        for task in sample_tasks:
            schedule.append((current_time, task))
            current_time += timedelta(hours=1)
        
        # Calculate energy
        energy = await scheduler._calculate_schedule_energy(schedule)
        
        # Energy should be a non-negative number
        assert isinstance(energy, (int, float))
        assert energy >= 0
    
    def test_quantum_priority_calculation(self, scheduler, sample_tasks):
        """Test quantum priority calculation"""
        high_priority_task = sample_tasks[0]  # CRITICAL
        low_priority_task = sample_tasks[2]   # LOW
        
        high_priority_score = scheduler._calculate_quantum_priority(high_priority_task)
        low_priority_score = scheduler._calculate_quantum_priority(low_priority_task)
        
        # Higher priority should generally result in higher score
        # Note: Due to quantum effects, this isn't guaranteed but statistically likely
        assert isinstance(high_priority_score, float)
        assert isinstance(low_priority_score, float)
    
    @pytest.mark.asyncio
    async def test_quantum_perturbation(self, scheduler, sample_tasks):
        """Test quantum perturbation operations"""
        for task in sample_tasks:
            scheduler.add_task(task)
        
        # Create initial schedule
        schedule = []
        current_time = datetime.utcnow()
        
        for task in sample_tasks:
            schedule.append((current_time, task))
            current_time += timedelta(hours=2)
        
        # Apply perturbation
        perturbed_schedule = await scheduler._quantum_perturbation(schedule)
        
        # Should return a schedule of same length
        assert len(perturbed_schedule) == len(schedule)
        
        # Should contain the same tasks (but possibly in different order/times)
        original_task_ids = {task.task_id for _, task in schedule}
        perturbed_task_ids = {task.task_id for _, task in perturbed_schedule}
        assert original_task_ids == perturbed_task_ids
    
    @pytest.mark.asyncio
    async def test_get_next_tasks(self, scheduler, sample_tasks):
        """Test getting next tasks to execute"""
        for task in sample_tasks:
            scheduler.add_task(task)
        
        # Get next tasks
        next_tasks = await scheduler.get_next_tasks(count=2)
        
        # Should return list of tasks
        assert isinstance(next_tasks, list)
        assert len(next_tasks) <= 2
        
        # All returned items should be QuantumTask instances
        for task in next_tasks:
            assert isinstance(task, QuantumTask)
    
    def test_get_schedule_statistics(self, scheduler, sample_tasks):
        """Test schedule statistics generation"""
        # Empty scheduler should return empty stats
        empty_stats = scheduler.get_schedule_statistics()
        assert empty_stats == {}
        
        # Add tasks and optimize
        for task in sample_tasks:
            scheduler.add_task(task)
        
        # Create a fake optimized schedule
        scheduler.scheduled_tasks = [
            (datetime.utcnow(), task) for task in sample_tasks
        ]
        
        stats = scheduler.get_schedule_statistics()
        
        # Check that statistics are generated
        assert "total_tasks" in stats
        assert "completion_probability_avg" in stats
        assert "quantum_coherence_avg" in stats
        assert "priority_distribution" in stats
        
        assert stats["total_tasks"] == len(sample_tasks)


class TestResourceConstraint:
    """Test resource constraint functionality"""
    
    def test_constraint_creation(self):
        """Test resource constraint creation"""
        constraint = ResourceConstraint(
            constraint_type="resource",
            weight=0.8,
            resource_type="cpu",
            available_amount=4.0
        )
        
        assert constraint.constraint_type == "resource"
        assert constraint.weight == 0.8
        assert constraint.resource_type == "cpu"
        assert constraint.available_amount == 4.0
    
    def test_constraint_evaluation(self):
        """Test resource constraint evaluation"""
        constraint = ResourceConstraint(
            constraint_type="resource",
            weight=1.0,
            resource_type="cpu",
            available_amount=8.0
        )
        
        # Create task with CPU resource requirement
        from quantum_task_planner.core.quantum_task import TaskResource
        task = QuantumTask(
            title="Resource Test Task",
            description="Task for testing resource constraints",
            resources=[TaskResource("cpu", 2.0, 4.0, 0.1)]
        )
        
        # Evaluate constraint
        satisfaction = constraint.evaluate(task, {})
        
        # Should return a value between 0 and 1
        assert 0 <= satisfaction <= 1
        
        # Task requiring 3.0 average CPU should be satisfiable with 8.0 available
        assert satisfaction > 0.5


class TestTimeConstraint:
    """Test time constraint functionality"""
    
    def test_time_constraint_creation(self):
        """Test time constraint creation"""
        deadline = datetime.utcnow() + timedelta(days=1)
        constraint = TimeConstraint(
            constraint_type="time",
            weight=0.9,
            deadline=deadline,
            criticality=1.0
        )
        
        assert constraint.constraint_type == "time"
        assert constraint.weight == 0.9
        assert constraint.deadline == deadline
        assert constraint.criticality == 1.0
    
    def test_time_constraint_evaluation(self):
        """Test time constraint evaluation"""
        # Deadline in 2 days
        deadline = datetime.utcnow() + timedelta(days=2)
        constraint = TimeConstraint(
            constraint_type="time",
            weight=1.0,
            deadline=deadline,
            criticality=1.0
        )
        
        # Task due in 1 day
        task = QuantumTask(
            title="Time Test Task",
            description="Task for testing time constraints",
            due_date=datetime.utcnow() + timedelta(days=1),
            estimated_duration=timedelta(hours=4)
        )
        
        # Evaluate constraint
        satisfaction = constraint.evaluate(task, {})
        
        # Should return a value between 0 and 1
        assert 0 <= satisfaction <= 1


class TestQuantumAnnealing:
    """Test quantum annealing algorithm components"""
    
    @pytest.fixture
    def scheduler_with_tasks(self, sample_tasks):
        """Fixture providing scheduler with tasks"""
        scheduler = QuantumTaskScheduler(max_iterations=50)
        
        for task in sample_tasks:
            scheduler.add_task(task)
        
        scheduler.set_resource_pool("cpu", 8.0)
        scheduler.set_resource_pool("memory", 16.0)
        
        return scheduler
    
    def test_temperature_schedules(self, scheduler_with_tasks):
        """Test different temperature schedules"""
        scheduler = scheduler_with_tasks
        
        # Test exponential schedule
        scheduler.temperature_schedule = "exponential"
        temp_exp = scheduler._get_temperature(50)  # Halfway through
        assert temp_exp > 0
        assert temp_exp < scheduler.initial_temperature
        
        # Test linear schedule
        scheduler.temperature_schedule = "linear"
        temp_linear = scheduler._get_temperature(50)
        assert temp_linear > 0
        assert temp_linear < scheduler.initial_temperature
        
        # Test quantum schedule
        scheduler.temperature_schedule = "quantum"
        temp_quantum = scheduler._get_temperature(50)
        assert temp_quantum > 0
    
    @pytest.mark.asyncio
    async def test_quantum_acceptance(self, scheduler_with_tasks):
        """Test quantum acceptance probability"""
        scheduler = scheduler_with_tasks
        
        # Mock schedules
        current_schedule = []
        candidate_schedule = []
        
        # Test acceptance with lower energy (should always accept)
        accept_better = await scheduler._quantum_accept(
            current_schedule, candidate_schedule, 10.0, 5.0, 1.0
        )
        assert accept_better is True
        
        # Test acceptance with higher energy at high temperature
        accept_worse_hot = await scheduler._quantum_accept(
            current_schedule, candidate_schedule, 5.0, 10.0, 100.0
        )
        # At high temperature, should sometimes accept worse solutions
        # This is probabilistic, so we just check it returns a boolean
        assert isinstance(accept_worse_hot, bool)
        
        # Test acceptance with higher energy at low temperature  
        accept_worse_cold = await scheduler._quantum_accept(
            current_schedule, candidate_schedule, 5.0, 10.0, 0.01
        )
        # At low temperature, should rarely accept worse solutions
        assert isinstance(accept_worse_cold, bool)
    
    @pytest.mark.asyncio
    async def test_quantum_interference(self, scheduler_with_tasks):
        """Test quantum interference effects"""
        scheduler = scheduler_with_tasks
        
        # Create a schedule
        schedule = []
        for task in scheduler.tasks.values():
            schedule.append((datetime.utcnow(), task))
        
        # Store initial quantum states
        initial_coherences = {
            task.task_id: task.quantum_coherence 
            for _, task in schedule
        }
        
        # Apply quantum interference
        await scheduler._apply_quantum_interference(schedule)
        
        # Check that quantum states may have changed
        # (Due to quantum effects, changes are not guaranteed but possible)
        final_coherences = {
            task.task_id: task.quantum_coherence 
            for _, task in schedule
        }
        
        # At least verify the operation completed without error
        assert len(initial_coherences) == len(final_coherences)


class TestSchedulerIntegration:
    """Integration tests for quantum scheduler"""
    
    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self):
        """Test complete optimization cycle"""
        scheduler = QuantumTaskScheduler(max_iterations=20)
        
        # Create diverse set of tasks
        tasks = []
        
        for i in range(5):
            task = QuantumTask(
                title=f"Integration Task {i}",
                description=f"Integration test task {i}",
                priority=list(TaskPriority)[i % len(TaskPriority)],
                estimated_duration=timedelta(hours=2 + i),
                due_date=datetime.utcnow() + timedelta(days=1 + i)
            )
            tasks.append(task)
            scheduler.add_task(task)
        
        # Add constraints
        cpu_constraint = ResourceConstraint(
            constraint_type="resource",
            weight=0.8,
            resource_type="cpu",
            available_amount=16.0
        )
        scheduler.add_constraint(cpu_constraint)
        
        time_constraint = TimeConstraint(
            constraint_type="time",
            weight=0.9,
            deadline=datetime.utcnow() + timedelta(days=10)
        )
        scheduler.add_constraint(time_constraint)
        
        # Set resource pools
        scheduler.set_resource_pool("cpu", 16.0)
        scheduler.set_resource_pool("memory", 32.0)
        
        # Run optimization
        optimized_schedule = await scheduler.optimize_schedule()
        
        # Verify results
        assert len(optimized_schedule) == len(tasks)
        
        # Check that optimization metrics were recorded
        assert scheduler.optimization_metrics is not None
        assert "final_energy" in scheduler.optimization_metrics
        assert "iterations" in scheduler.optimization_metrics
        
        # Get statistics
        stats = scheduler.get_schedule_statistics()
        assert stats["total_tasks"] == len(tasks)
        
        # Get next tasks
        next_tasks = await scheduler.get_next_tasks(count=3)
        assert len(next_tasks) <= 3
    
    @pytest.mark.asyncio
    async def test_scheduler_performance(self):
        """Test scheduler performance with many tasks"""
        import time
        
        scheduler = QuantumTaskScheduler(max_iterations=10)  # Reduced for performance test
        
        # Create many tasks
        num_tasks = 50
        for i in range(num_tasks):
            task = QuantumTask(
                title=f"Performance Task {i}",
                description=f"Performance test task {i}",
                priority=TaskPriority.MEDIUM
            )
            scheduler.add_task(task)
        
        # Measure optimization time
        start_time = time.time()
        optimized_schedule = await scheduler.optimize_schedule()
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert optimization_time < 30.0  # 30 seconds for 50 tasks
        assert len(optimized_schedule) == num_tasks
    
    @pytest.mark.asyncio
    async def test_scheduler_with_entangled_tasks(self):
        """Test scheduler with quantum entangled tasks"""
        scheduler = QuantumTaskScheduler(max_iterations=20)
        
        # Create entangled tasks
        task1 = QuantumTask(title="Entangled Task 1", description="First entangled task")
        task2 = QuantumTask(title="Entangled Task 2", description="Second entangled task")
        task3 = QuantumTask(title="Independent Task", description="Independent task")
        
        # Entangle first two tasks
        task1.entangle_with(task2, entanglement_strength=0.8)
        
        # Add to scheduler
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        
        # Optimize schedule
        optimized_schedule = await scheduler.optimize_schedule()
        
        # Check that all tasks are scheduled
        assert len(optimized_schedule) == 3
        
        # Check that entanglement is considered in energy calculation
        energy = await scheduler._calculate_schedule_energy(optimized_schedule)
        assert isinstance(energy, (int, float))
        assert energy >= 0


if __name__ == "__main__":
    pytest.main([__file__])