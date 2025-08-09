#!/usr/bin/env python3
"""
Comprehensive Quantum Task Planner Testing Suite

Tests all components of the quantum task planning system without requiring
external dependencies like pytest or numpy.
"""

import sys
import asyncio
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

class TestResult:
    """Simple test result container"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.start_time = time.time()
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def summary(self):
        duration = time.time() - self.start_time
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Duration: {duration:.2f}s")
        print(f"Success rate: {(self.passed/total*100):.1f}%" if total > 0 else "No tests run")
        
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        
        return self.failed == 0


def test_basic_imports():
    """Test basic module imports"""
    result = TestResult()
    
    try:
        # Test core imports without dependencies
        import json
        import uuid
        import hashlib
        from datetime import datetime, timedelta
        from typing import Dict, List, Optional
        result.add_pass("Basic Python imports")
    except Exception as e:
        result.add_fail("Basic Python imports", str(e))
    
    return result


def test_quantum_task_basic():
    """Test basic QuantumTask functionality"""
    result = TestResult()
    
    try:
        # Mock numpy for testing
        import sys
        
        class MockNumpy:
            def random(self):
                import random
                return random.random()
            
            def exp(self, x):
                import math
                return math.exp(x)
                
            def sqrt(self, x):
                import math
                return math.sqrt(x)
                
            def mean(self, values):
                return sum(values) / len(values)
                
            def std(self, values):
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                return variance ** 0.5
        
        sys.modules['numpy'] = MockNumpy()
        
        # Now test quantum task creation
        from quantum_task_planner.core.quantum_task import QuantumTask, TaskState, TaskPriority
        
        # Test task creation
        task = QuantumTask(
            title="Test Task",
            description="A test quantum task",
            priority=TaskPriority.HIGH,
            complexity_factor=2.5
        )
        
        assert task.title == "Test Task"
        assert task.quantum_coherence > 0
        assert task.state == TaskState.PENDING
        result.add_pass("QuantumTask creation")
        
        # Test task state changes
        task.set_state(TaskState.RUNNING)
        assert task.state == TaskState.RUNNING
        result.add_pass("QuantumTask state changes")
        
        # Test quantum measurement
        measured_state = task.measure_state(0.1)
        assert measured_state in [TaskState.PENDING, TaskState.RUNNING, TaskState.COMPLETED, TaskState.PAUSED]
        result.add_pass("Quantum state measurement")
        
    except Exception as e:
        result.add_fail("QuantumTask basic functionality", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_task_scheduling():
    """Test quantum task scheduling"""
    result = TestResult()
    
    try:
        from quantum_task_planner.core.quantum_scheduler import QuantumTaskScheduler
        from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
        
        # Create scheduler
        scheduler = QuantumTaskScheduler(max_iterations=10)  # Reduced for testing
        
        # Create test tasks
        task1 = QuantumTask("Task 1", "First task", TaskPriority.HIGH)
        task2 = QuantumTask("Task 2", "Second task", TaskPriority.MEDIUM)
        task3 = QuantumTask("Task 3", "Third task", TaskPriority.LOW)
        
        # Add tasks to scheduler
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        scheduler.add_task(task3)
        
        assert len(scheduler.tasks) == 3
        result.add_pass("Task scheduling - add tasks")
        
        # Test task removal
        removed_task = scheduler.remove_task(task2.task_id)
        assert removed_task.task_id == task2.task_id
        assert len(scheduler.tasks) == 2
        result.add_pass("Task scheduling - remove tasks")
        
    except Exception as e:
        result.add_fail("Task scheduling", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_async_execution():
    """Test async task execution system"""
    result = TestResult()
    
    try:
        async def run_async_tests():
            from quantum_task_planner.core.async_executor import get_quantum_executor, ExecutorType
            from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
            
            # Get executor
            executor = get_quantum_executor()
            
            # Create test task
            task = QuantumTask("Async Test Task", "Test async execution", TaskPriority.MEDIUM)
            
            # Execute task
            context = await executor.execute_task(task, ExecutorType.ASYNC_COROUTINE)
            
            assert context.task.task_id == task.task_id
            result.add_pass("Async execution - task execution")
            
            # Check execution metrics
            metrics = executor.get_system_metrics()
            assert "executor_metrics" in metrics
            result.add_pass("Async execution - metrics collection")
        
        # Run async tests
        asyncio.run(run_async_tests())
        
    except Exception as e:
        result.add_fail("Async execution", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_security_system():
    """Test quantum security components"""
    result = TestResult()
    
    try:
        from quantum_task_planner.security.quantum_security import (
            QuantumSecurityManager, SecurityLevel, SecurityPolicy
        )
        from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
        
        # Create security manager
        policy = SecurityPolicy(SecurityLevel.ENHANCED)
        security_manager = QuantumSecurityManager("test_node", policy)
        
        # Create test task
        task = QuantumTask("Secure Task", "Test security", TaskPriority.HIGH)
        
        # Create quantum signature
        signature = asyncio.run(security_manager.create_quantum_signature(task))
        assert signature.task_id == task.task_id
        assert signature.verify_integrity()
        result.add_pass("Security - quantum signature creation")
        
        # Validate task security
        is_valid = asyncio.run(security_manager.validate_task_security(task, signature))
        assert is_valid
        result.add_pass("Security - task validation")
        
        # Get security status
        status = security_manager.get_security_status()
        assert "node_id" in status
        assert "security_level" in status
        result.add_pass("Security - status reporting")
        
    except Exception as e:
        result.add_fail("Security system", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_ml_optimization():
    """Test ML optimization components (without PyTorch)"""
    result = TestResult()
    
    try:
        from quantum_task_planner.ml.quantum_ml_optimizer import (
            QuantumMLOptimizer, QuantumStateEncoder
        )
        from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
        
        # Create ML optimizer
        ml_optimizer = QuantumMLOptimizer(state_encoder_dim=16)  # Smaller for testing
        
        # Create test tasks
        tasks = [
            QuantumTask("ML Task 1", "First ML test", TaskPriority.HIGH),
            QuantumTask("ML Task 2", "Second ML test", TaskPriority.MEDIUM),
        ]
        
        # Test state encoding
        state_encoder = QuantumStateEncoder(feature_dim=16)
        encoded_state = state_encoder.encode_task(tasks[0])
        assert len(encoded_state) == 16
        result.add_pass("ML optimization - state encoding")
        
        # Test task scheduling optimization
        system_metrics = {"resource_utilization": {"cpu": 0.5, "memory": 0.3}}
        optimized_schedule = asyncio.run(
            ml_optimizer.optimize_task_scheduling(tasks, system_metrics)
        )
        assert len(optimized_schedule) == len(tasks)
        result.add_pass("ML optimization - task scheduling")
        
        # Test completion time prediction
        completion_time = asyncio.run(
            ml_optimizer.predict_task_completion_time(tasks[0], system_metrics)
        )
        assert completion_time > 0
        result.add_pass("ML optimization - completion time prediction")
        
    except Exception as e:
        result.add_fail("ML optimization", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_scaling_system():
    """Test auto-scaling and load balancing"""
    result = TestResult()
    
    try:
        from quantum_task_planner.performance.scaling import (
            get_load_balancer, get_auto_scaler, get_cluster_orchestrator
        )
        
        # Test load balancer
        load_balancer = get_load_balancer(coherence_weight=0.3)
        
        # Register test worker
        load_balancer.register_worker("worker1", "http://worker1:8000", 0.8)
        assert "worker1" in load_balancer.worker_stats
        result.add_pass("Scaling - load balancer worker registration")
        
        # Test worker selection
        selected_worker = load_balancer.select_worker(0.7)
        assert selected_worker == "worker1"
        result.add_pass("Scaling - worker selection")
        
        # Test auto-scaler
        auto_scaler = get_auto_scaler(check_interval=1.0)
        scaling_status = auto_scaler.get_scaling_status()
        assert "current_instances" in scaling_status
        result.add_pass("Scaling - auto-scaler status")
        
        # Test cluster orchestrator
        orchestrator = get_cluster_orchestrator()
        cluster_status = orchestrator.get_cluster_status()
        assert "total_instances" in cluster_status
        result.add_pass("Scaling - cluster orchestrator")
        
        # Stop monitoring to clean up
        auto_scaler.stop_monitoring()
        
    except Exception as e:
        result.add_fail("Scaling system", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_api_functionality():
    """Test API components"""
    result = TestResult()
    
    try:
        # Test API module imports
        from quantum_task_planner.api.quantum_api import app, scheduler, optimizer
        
        assert app is not None
        assert scheduler is not None
        assert optimizer is not None
        result.add_pass("API - module imports")
        
        # Test dashboard components
        from quantum_task_planner.api.quantum_dashboard import get_dashboard_manager
        
        dashboard_manager = get_dashboard_manager()
        assert dashboard_manager is not None
        result.add_pass("API - dashboard manager")
        
    except Exception as e:
        result.add_fail("API functionality", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def test_integration():
    """Test end-to-end integration"""
    result = TestResult()
    
    try:
        async def integration_test():
            from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
            from quantum_task_planner.core.quantum_scheduler import QuantumTaskScheduler
            from quantum_task_planner.core.async_executor import get_quantum_executor
            
            # Create integration test scenario
            scheduler = QuantumTaskScheduler(max_iterations=5)
            executor = get_quantum_executor()
            
            # Create test tasks
            tasks = [
                QuantumTask("Integration Task 1", "First integration test", TaskPriority.HIGH),
                QuantumTask("Integration Task 2", "Second integration test", TaskPriority.MEDIUM),
                QuantumTask("Integration Task 3", "Third integration test", TaskPriority.LOW)
            ]
            
            # Add tasks to scheduler
            for task in tasks:
                scheduler.add_task(task)
            
            # Optimize schedule
            optimized_schedule = await scheduler.optimize_schedule()
            assert len(optimized_schedule) == len(tasks)
            result.add_pass("Integration - schedule optimization")
            
            # Execute first task
            if optimized_schedule:
                first_task = optimized_schedule[0][1]  # (time, task) tuple
                context = await executor.execute_task(first_task)
                assert context.task.task_id == first_task.task_id
                result.add_pass("Integration - task execution")
            
            # Get system metrics
            metrics = executor.get_system_metrics()
            assert "executor_metrics" in metrics
            result.add_pass("Integration - system metrics")
        
        asyncio.run(integration_test())
        
    except Exception as e:
        result.add_fail("Integration test", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def run_performance_benchmark():
    """Run performance benchmarks"""
    result = TestResult()
    
    try:
        import time
        from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
        
        # Benchmark task creation
        start_time = time.time()
        tasks = []
        for i in range(100):
            task = QuantumTask(f"Benchmark Task {i}", f"Description {i}", TaskPriority.MEDIUM)
            tasks.append(task)
        
        creation_time = time.time() - start_time
        assert creation_time < 1.0  # Should create 100 tasks in under 1 second
        result.add_pass(f"Performance - task creation (100 tasks in {creation_time:.3f}s)")
        
        # Benchmark quantum measurement
        start_time = time.time()
        for task in tasks[:10]:  # Measure first 10 tasks
            task.measure_state(0.1)
        measurement_time = time.time() - start_time
        result.add_pass(f"Performance - quantum measurements (10 tasks in {measurement_time:.3f}s)")
        
        # Memory usage check (basic)
        import sys
        memory_usage = sys.getsizeof(tasks)
        assert memory_usage < 50000  # Should be reasonable memory usage
        result.add_pass(f"Performance - memory usage ({memory_usage} bytes for 100 tasks)")
        
    except Exception as e:
        result.add_fail("Performance benchmark", str(e))
        print(f"Detailed error: {traceback.format_exc()}")
    
    return result


def main():
    """Main test runner"""
    print("🌌 Quantum Task Planner - Comprehensive Test Suite")
    print("="*60)
    
    # List of all test functions
    test_functions = [
        test_basic_imports,
        test_quantum_task_basic,
        test_task_scheduling,
        test_async_execution,
        test_security_system,
        test_ml_optimization,
        test_scaling_system,
        test_api_functionality,
        test_integration,
        run_performance_benchmark
    ]
    
    # Aggregate results
    total_passed = 0
    total_failed = 0
    all_errors = []
    
    # Run all tests
    for test_func in test_functions:
        print(f"\n📋 Running {test_func.__name__}...")
        test_result = test_func()
        
        total_passed += test_result.passed
        total_failed += test_result.failed
        all_errors.extend(test_result.errors)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {total_passed + total_failed}")
    print(f"Total passed: {total_passed}")
    print(f"Total failed: {total_failed}")
    
    if total_passed + total_failed > 0:
        success_rate = (total_passed / (total_passed + total_failed)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if all_errors:
        print(f"\n❌ FAILED TESTS:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
    
    if total_failed == 0:
        print(f"\n✅ ALL TESTS PASSED! Quantum Task Planner is ready for production.")
        return True
    else:
        print(f"\n⚠️  {total_failed} tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)