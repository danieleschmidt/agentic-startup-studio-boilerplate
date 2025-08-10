#!/usr/bin/env python3
"""
Simple Test Runner

A dependency-free test runner for the Quantum Task Planner system.
"""

import sys
import traceback
import time
import asyncio
from pathlib import Path

# Add quantum_task_planner to path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_task_planner.core.quantum_task import QuantumTask, TaskState, TaskPriority
from quantum_task_planner.core.quantum_scheduler import QuantumTaskScheduler
from quantum_task_planner.core.simple_optimizer import SimpleQuantumOptimizer
from quantum_task_planner.core.simple_entanglement import SimpleEntanglementManager, SimpleEntanglementType
from quantum_task_planner.utils.robust_validation import QuantumValidator, validate_task_data

from datetime import datetime, timedelta


class TestRunner:
    """Simple test runner"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test"""
        print(f"🧪 {test_name}...", end=" ")
        try:
            result = test_func()
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            
            print("✅ PASSED")
            self.passed += 1
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            self.failed += 1
            self.errors.append((test_name, str(e), traceback.format_exc()))
    
    def assert_true(self, condition, message="Assertion failed"):
        """Simple assertion"""
        if not condition:
            raise AssertionError(message)
    
    def assert_equal(self, a, b, message=None):
        """Assert equality"""
        if a != b:
            msg = message or f"Expected {a} == {b}"
            raise AssertionError(msg)
    
    def assert_not_equal(self, a, b, message=None):
        """Assert inequality"""
        if a == b:
            msg = message or f"Expected {a} != {b}"
            raise AssertionError(msg)
    
    def assert_greater(self, a, b, message=None):
        """Assert greater than"""
        if not a > b:
            msg = message or f"Expected {a} > {b}"
            raise AssertionError(msg)
    
    def assert_in(self, item, container, message=None):
        """Assert item in container"""
        if item not in container:
            msg = message or f"Expected {item} in {container}"
            raise AssertionError(msg)
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"Test Results: {self.passed} passed, {self.failed} failed out of {total} tests")
        
        if self.failed > 0:
            print("\nFailure Details:")
            for test_name, error, trace in self.errors:
                print(f"\n{test_name}:")
                print(f"  Error: {error}")
        
        return self.failed == 0


def test_task_creation(runner):
    """Test basic task creation"""
    task = QuantumTask(
        title="Test Task",
        description="A test task for validation",
        priority=TaskPriority.MEDIUM,
        complexity_factor=1.5,
        estimated_duration=timedelta(hours=2)
    )
    
    runner.assert_true(task.task_id is not None, "Task should have ID")
    runner.assert_equal(task.title, "Test Task", "Task title should match")
    runner.assert_equal(task.priority, TaskPriority.MEDIUM, "Task priority should match")
    runner.assert_greater(task.quantum_coherence, 0.0, "Quantum coherence should be positive")
    runner.assert_greater(task.get_completion_probability(), 0.0, "Completion probability should be positive")


def test_task_measurement(runner):
    """Test quantum measurement"""
    task = QuantumTask(
        title="Measurement Test",
        description="Test quantum measurement",
        priority=TaskPriority.HIGH
    )
    
    initial_coherence = task.quantum_coherence
    measured_state = task.measure_state(observer_effect=0.1)
    
    runner.assert_true(measured_state in TaskState, "Measured state should be valid")
    runner.assert_greater(initial_coherence, task.quantum_coherence, "Coherence should decrease after measurement")


def test_scheduler_operations(runner):
    """Test scheduler operations"""
    scheduler = QuantumTaskScheduler()
    
    # Create test tasks
    tasks = [
        QuantumTask(
            title=f"Task {i}",
            description=f"Test task {i}",
            priority=TaskPriority.MEDIUM if i % 2 else TaskPriority.HIGH,
            complexity_factor=1.0 + i * 0.1
        )
        for i in range(5)
    ]
    
    # Add tasks to scheduler
    for task in tasks:
        scheduler.add_task(task)
    
    runner.assert_equal(len(scheduler.tasks), 5, "Should have 5 tasks")
    
    # Test task removal
    task_id = tasks[0].task_id
    removed_task = scheduler.remove_task(task_id)
    
    runner.assert_true(removed_task is not None, "Should return removed task")
    runner.assert_equal(len(scheduler.tasks), 4, "Should have 4 tasks after removal")


async def test_async_optimization(runner):
    """Test asynchronous optimization"""
    optimizer = SimpleQuantumOptimizer()
    
    tasks = [
        QuantumTask(
            title=f"Optimization Task {i}",
            description=f"Task for optimization testing {i}",
            priority=TaskPriority.HIGH if i < 2 else TaskPriority.MEDIUM,
            complexity_factor=1.0 + i * 0.2
        )
        for i in range(3)
    ]
    
    resources = {"cpu": 100.0, "memory": 16.0}
    result = await optimizer.optimize_task_allocation(tasks, resources)
    
    runner.assert_equal(result['status'], 'success', "Optimization should succeed")
    runner.assert_in('optimized_allocations', result, "Should contain allocations")
    runner.assert_equal(len(result['optimized_allocations']), 3, "Should optimize all tasks")


async def test_entanglement_operations(runner):
    """Test entanglement operations"""
    entanglement_manager = SimpleEntanglementManager()
    
    # Create tasks for entanglement
    tasks = [
        QuantumTask(
            title=f"Entangled Task {i}",
            description=f"Task for entanglement {i}",
            priority=TaskPriority.MEDIUM
        )
        for i in range(3)
    ]
    
    # Create entanglement
    bond_id = await entanglement_manager.create_entanglement(
        tasks, SimpleEntanglementType.BELL_STATE, 0.8
    )
    
    runner.assert_true(bond_id is not None, "Should create entanglement bond")
    runner.assert_in(bond_id, entanglement_manager.entanglement_bonds, "Bond should be registered")
    
    # Check entanglement effects
    for task in tasks:
        runner.assert_greater(len(task.entangled_tasks), 0, "Task should have entangled partners")
    
    # Test statistics
    stats = entanglement_manager.get_entanglement_statistics()
    runner.assert_equal(stats['active_bonds'], 1, "Should have one active bond")
    runner.assert_equal(stats['total_entangled_tasks'], 3, "Should have three entangled tasks")


def test_validation_system(runner):
    """Test validation system"""
    validator = QuantumValidator()
    
    # Test valid task data
    valid_data = {
        'title': 'Valid Task Title',
        'description': 'A properly formatted task description',
        'priority': 'high',
        'complexity_factor': 2.0
    }
    
    validation_results = validate_task_data(valid_data)
    
    # Should have no validation errors
    for field_results in validation_results.values():
        for result in field_results:
            runner.assert_true(result.valid, f"Validation should pass: {result.message}")
    
    # Test invalid task data
    invalid_data = {
        'title': '',  # Empty title
        'description': 'x' * 3000,  # Too long
        'priority': 'invalid',  # Invalid priority
        'complexity_factor': -1.0  # Invalid negative value
    }
    
    validation_results = validate_task_data(invalid_data)
    
    # Should have validation errors
    has_errors = False
    for field_results in validation_results.values():
        for result in field_results:
            if not result.valid:
                has_errors = True
                break
    
    runner.assert_true(has_errors, "Invalid data should produce validation errors")


def test_performance_benchmark(runner):
    """Test system performance"""
    scheduler = QuantumTaskScheduler()
    optimizer = SimpleQuantumOptimizer()
    
    # Create larger task set
    num_tasks = 50
    tasks = [
        QuantumTask(
            title=f"Performance Task {i}",
            description=f"Task for performance testing {i}",
            priority=TaskPriority.MEDIUM if i % 3 else TaskPriority.HIGH,
            complexity_factor=1.0 + (i % 5) * 0.2
        )
        for i in range(num_tasks)
    ]
    
    # Benchmark task addition
    start_time = time.time()
    for task in tasks:
        scheduler.add_task(task)
    creation_time = time.time() - start_time
    
    runner.assert_greater(1.0, creation_time, "Task creation should be fast")
    runner.assert_equal(len(scheduler.tasks), num_tasks, f"Should have {num_tasks} tasks")
    
    # Benchmark measurements
    start_time = time.time()
    measurements = [task.measure_state() for task in tasks]
    measurement_time = time.time() - start_time
    
    runner.assert_greater(0.5, measurement_time, "Measurements should be fast")
    runner.assert_equal(len(measurements), num_tasks, f"Should have {num_tasks} measurements")


def test_real_world_scenario(runner):
    """Test realistic development workflow"""
    scheduler = QuantumTaskScheduler()
    entanglement_manager = SimpleEntanglementManager()
    
    # Create development project tasks
    project_tasks = [
        ("Requirements Analysis", TaskPriority.HIGH, 1.0),
        ("System Design", TaskPriority.HIGH, 2.0),
        ("Database Schema", TaskPriority.HIGH, 1.5),
        ("API Development", TaskPriority.HIGH, 3.0),
        ("Frontend Development", TaskPriority.MEDIUM, 2.5),
        ("Testing", TaskPriority.MEDIUM, 2.0),
        ("Deployment", TaskPriority.CRITICAL, 1.0),
        ("Documentation", TaskPriority.LOW, 1.0)
    ]
    
    tasks = []
    for title, priority, complexity in project_tasks:
        task = QuantumTask(
            title=title,
            description=f"Development task: {title}",
            priority=priority,
            complexity_factor=complexity,
            estimated_duration=timedelta(hours=complexity * 8)
        )
        tasks.append(task)
        scheduler.add_task(task)
    
    runner.assert_equal(len(scheduler.tasks), len(project_tasks), "All tasks should be added")
    
    # Create logical dependencies (entanglements)
    design_task = next(t for t in tasks if "Design" in t.title)
    api_task = next(t for t in tasks if "API" in t.title)
    
    # Create dependency entanglement
    bond_id = asyncio.run(
        entanglement_manager.create_entanglement(
            [design_task, api_task], SimpleEntanglementType.DEPENDENCY, 0.9
        )
    )
    
    runner.assert_true(bond_id is not None, "Should create dependency entanglement")
    
    # Test schedule optimization
    schedule = asyncio.run(scheduler.optimize_schedule())
    
    runner.assert_equal(len(schedule), len(tasks), "Schedule should include all tasks")
    
    # Critical tasks should be scheduled with high priority
    critical_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]
    if critical_tasks:
        critical_positions = [
            i for i, (_, task) in enumerate(schedule)
            if task.priority == TaskPriority.CRITICAL
        ]
        # Critical tasks should be in first half of schedule
        for pos in critical_positions:
            runner.assert_greater(len(schedule) / 2, pos, "Critical tasks should be prioritized")


def main():
    """Run all tests"""
    print("🌌 Quantum Task Planner - Test Suite")
    print("=" * 60)
    
    runner = TestRunner()
    
    # Run all tests
    runner.run_test("Task Creation", lambda: test_task_creation(runner))
    runner.run_test("Task Measurement", lambda: test_task_measurement(runner))
    runner.run_test("Scheduler Operations", lambda: test_scheduler_operations(runner))
    runner.run_test("Async Optimization", lambda: test_async_optimization(runner))
    runner.run_test("Entanglement Operations", lambda: test_entanglement_operations(runner))
    runner.run_test("Validation System", lambda: test_validation_system(runner))
    runner.run_test("Performance Benchmark", lambda: test_performance_benchmark(runner))
    runner.run_test("Real-world Scenario", lambda: test_real_world_scenario(runner))
    
    # Print summary
    success = runner.print_summary()
    
    if success:
        print("\n✨ All tests passed! System is ready for deployment.")
    else:
        print("\n❌ Some tests failed. Please review and fix issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)