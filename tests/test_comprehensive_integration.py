"""
Comprehensive Integration Tests

End-to-end integration tests for the Quantum Task Planner system
with real-world scenarios and performance benchmarks.
"""

import asyncio
import pytest
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum_task_planner.core.quantum_task import QuantumTask, TaskState, TaskPriority
from quantum_task_planner.core.quantum_scheduler import QuantumTaskScheduler
from quantum_task_planner.core.simple_optimizer import SimpleQuantumOptimizer
from quantum_task_planner.core.simple_entanglement import SimpleEntanglementManager, SimpleEntanglementType
from quantum_task_planner.utils.robust_validation import QuantumValidator, validate_task_data
from quantum_task_planner.utils.robust_logging import setup_robust_logging
from quantum_task_planner.performance.advanced_cache import QuantumAwareCache, quantum_cached


class TestQuantumTaskIntegration:
    """Integration tests for quantum task operations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.scheduler = QuantumTaskScheduler()
        self.optimizer = SimpleQuantumOptimizer()
        self.entanglement_manager = SimpleEntanglementManager()
        self.validator = QuantumValidator()
        self.cache = QuantumAwareCache("test_cache", max_size_mb=10)
        
        # Setup logging
        self.logger = setup_robust_logging("INFO", structured=False, enable_metrics=False)
    
    def create_test_task(self, title: str = "Test Task", priority: str = "medium", 
                        complexity: float = 1.0) -> QuantumTask:
        """Create test task"""
        return QuantumTask(
            title=title,
            description=f"Test task: {title}",
            priority=TaskPriority.MEDIUM if priority == "medium" else TaskPriority.HIGH,
            complexity_factor=complexity,
            estimated_duration=timedelta(hours=2)
        )
    
    def test_task_lifecycle(self):
        """Test complete task lifecycle"""
        # Create task
        task = self.create_test_task("Integration Test Task", "high")
        
        # Validate task data
        task_data = {
            'title': task.title,
            'description': task.description,
            'priority': task.priority.name.lower(),
            'complexity_factor': task.complexity_factor
        }
        
        validation_results = validate_task_data(task_data)
        assert all(all(r.valid for r in results) for results in validation_results.values())
        
        # Add to scheduler
        self.scheduler.add_task(task)
        assert task.task_id in self.scheduler.tasks
        
        # Test quantum measurement
        initial_coherence = task.quantum_coherence
        measured_state = task.measure_state(observer_effect=0.1)
        assert task.quantum_coherence < initial_coherence  # Should decrease due to measurement
        
        # Test state updates
        task.start_execution()
        assert task.state == TaskState.IN_PROGRESS
        
        task.complete_execution()
        assert task.state == TaskState.COMPLETED
    
    def test_entanglement_operations(self):
        """Test quantum entanglement operations"""
        # Create multiple tasks
        tasks = [
            self.create_test_task(f"Task {i}", "medium", 1.0 + i * 0.1)
            for i in range(3)
        ]
        
        for task in tasks:
            self.scheduler.add_task(task)
        
        # Create entanglement
        bond_id = asyncio.run(
            self.entanglement_manager.create_entanglement(
                tasks, SimpleEntanglementType.BELL_STATE, 0.8
            )
        )
        
        assert bond_id in self.entanglement_manager.entanglement_bonds
        
        # Check entanglement effects
        for task in tasks:
            assert len(task.entangled_tasks) == 2  # Entangled with 2 other tasks
        
        # Test entanglement measurement
        measurements = asyncio.run(
            self.entanglement_manager.measure_entanglement(bond_id)
        )
        
        assert len(measurements) == len(tasks)
        
        # Test statistics
        stats = self.entanglement_manager.get_entanglement_statistics()
        assert stats['active_bonds'] == 1
        assert stats['total_entangled_tasks'] == 3
    
    @pytest.mark.asyncio
    async def test_scheduling_optimization(self):
        """Test scheduling and optimization integration"""
        # Create diverse task set
        tasks = [
            self.create_test_task("Critical Deploy", "high", 2.0),
            self.create_test_task("Database Migration", "high", 3.0),
            self.create_test_task("Unit Tests", "medium", 1.5),
            self.create_test_task("Documentation", "low", 1.0),
            self.create_test_task("Code Review", "medium", 1.2)
        ]
        
        for task in tasks:
            self.scheduler.add_task(task)
        
        # Test optimization
        resources = {"cpu": 100.0, "memory": 16.0}
        optimization_result = await self.optimizer.optimize_task_allocation(tasks, resources)
        
        assert optimization_result['status'] == 'success'
        assert 'optimized_allocations' in optimization_result
        assert len(optimization_result['optimized_allocations']) == len(tasks)
        
        # Test scheduling
        optimized_schedule = await self.scheduler.optimize_schedule()
        
        assert len(optimized_schedule) == len(tasks)
        assert all(isinstance(entry, tuple) and len(entry) == 2 for entry in optimized_schedule)
        
        # Verify high-priority tasks are scheduled earlier
        task_priorities = [(start_time, task.priority.probability_weight) 
                          for start_time, task in optimized_schedule]
        
        # First task should have high priority
        assert task_priorities[0][1] >= 0.8  # High or critical priority
    
    def test_caching_integration(self):
        """Test caching with quantum-aware features"""
        
        @quantum_cached(cache_name="test_integration", ttl=60, include_quantum_context=True)
        def expensive_quantum_calculation(task_id: str, coherence: float) -> Dict[str, Any]:
            """Simulate expensive calculation"""
            time.sleep(0.1)  # Simulate computation time
            return {
                'task_id': task_id,
                'result': coherence * 1.5,
                'timestamp': time.time()
            }
        
        task = self.create_test_task("Cached Task")
        
        # First call - should compute
        start_time = time.time()
        result1 = expensive_quantum_calculation(task.task_id, task.quantum_coherence)
        duration1 = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        result2 = expensive_quantum_calculation(task.task_id, task.quantum_coherence)
        duration2 = time.time() - start_time
        
        # Verify caching worked
        assert result1['result'] == result2['result']
        assert duration2 < duration1 * 0.5  # Should be much faster
        
        # Check cache statistics
        stats = self.cache.get_stats()
        assert stats['hits'] > 0
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        # Create large task set for performance testing
        num_tasks = 100
        tasks = [
            self.create_test_task(f"Perf Task {i}", "medium", 1.0 + (i % 5) * 0.1)
            for i in range(num_tasks)
        ]
        
        # Benchmark task creation and addition
        start_time = time.time()
        for task in tasks:
            self.scheduler.add_task(task)
        creation_time = time.time() - start_time
        
        assert creation_time < 1.0  # Should complete within 1 second
        assert len(self.scheduler.tasks) == num_tasks
        
        # Benchmark quantum measurements
        start_time = time.time()
        measured_states = [task.measure_state() for task in tasks]
        measurement_time = time.time() - start_time
        
        assert measurement_time < 0.5  # Should complete within 0.5 seconds
        assert len(measured_states) == num_tasks
        
        # Benchmark optimization
        resources = {"cpu": 1000.0, "memory": 64.0}
        start_time = time.time()
        optimization_result = asyncio.run(
            self.optimizer.optimize_task_allocation(tasks, resources)
        )
        optimization_time = time.time() - start_time
        
        assert optimization_time < 2.0  # Should complete within 2 seconds
        assert optimization_result['status'] == 'success'
    
    def test_error_handling_integration(self):
        """Test comprehensive error handling"""
        # Test invalid task creation
        with pytest.raises((ValueError, TypeError)):
            QuantumTask(
                title="",  # Invalid empty title
                description="Test",
                priority=TaskPriority.MEDIUM,
                complexity_factor=-1.0  # Invalid negative complexity
            )
        
        # Test invalid entanglement
        task = self.create_test_task("Test Task")
        
        with pytest.raises(ValueError):
            # Should fail with insufficient tasks
            asyncio.run(
                self.entanglement_manager.create_entanglement([task], SimpleEntanglementType.BELL_STATE, 0.5)
            )
        
        # Test validation error handling
        invalid_task_data = {
            'title': 'x' * 300,  # Too long
            'description': '<script>alert("xss")</script>',  # XSS attempt
            'priority': 'invalid_priority',
            'complexity_factor': 15.0  # Out of range
        }
        
        validation_results = validate_task_data(invalid_task_data)
        errors = []
        for field_results in validation_results.values():
            for result in field_results:
                if not result.valid:
                    errors.append(result)
        
        assert len(errors) > 0  # Should have validation errors
    
    def test_concurrent_operations(self):
        """Test concurrent operations and thread safety"""
        import threading
        
        results = []
        errors = []
        
        def create_and_process_tasks(start_idx: int):
            """Worker function for concurrent testing"""
            try:
                # Create tasks
                task_batch = [
                    self.create_test_task(f"Concurrent Task {start_idx}-{i}", "medium")
                    for i in range(10)
                ]
                
                # Add to scheduler
                for task in task_batch:
                    self.scheduler.add_task(task)
                
                # Perform operations
                for task in task_batch:
                    task.measure_state()
                    task.start_execution()
                
                results.append(len(task_batch))
                
            except Exception as e:
                errors.append(str(e))
        
        # Start concurrent workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_and_process_tasks, args=(i * 10,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"
        assert sum(results) == 50  # 5 threads * 10 tasks each
        assert len(self.scheduler.tasks) >= 50  # May have some from previous tests
    
    def test_real_world_scenario(self):
        """Test realistic development workflow scenario"""
        # Simulate a software development project
        project_tasks = [
            # Planning phase
            ("Define Requirements", "high", 1.0),
            ("System Architecture", "high", 2.0),
            ("Database Design", "high", 2.5),
            
            # Development phase
            ("Setup Development Environment", "medium", 1.5),
            ("Implement Authentication", "high", 3.0),
            ("Build Core APIs", "high", 4.0),
            ("Frontend Development", "medium", 3.5),
            ("Integration Testing", "medium", 2.0),
            
            # Deployment phase
            ("Setup CI/CD Pipeline", "medium", 2.0),
            ("Performance Testing", "medium", 2.5),
            ("Security Audit", "high", 2.0),
            ("Production Deployment", "critical", 1.5),
            
            # Maintenance phase
            ("Documentation", "low", 1.0),
            ("Monitoring Setup", "medium", 1.5),
            ("Bug Fixes", "medium", 1.0)
        ]
        
        tasks = []
        for title, priority_str, complexity in project_tasks:
            priority_map = {
                "critical": TaskPriority.CRITICAL,
                "high": TaskPriority.HIGH,
                "medium": TaskPriority.MEDIUM,
                "low": TaskPriority.LOW
            }
            
            task = QuantumTask(
                title=title,
                description=f"Project task: {title}",
                priority=priority_map[priority_str],
                complexity_factor=complexity,
                estimated_duration=timedelta(hours=complexity * 8)  # 8 hours per complexity unit
            )
            
            tasks.append(task)
            self.scheduler.add_task(task)
        
        # Create logical entanglements (dependencies)
        auth_task = next(t for t in tasks if "Authentication" in t.title)
        api_task = next(t for t in tasks if "Core APIs" in t.title)
        frontend_task = next(t for t in tasks if "Frontend" in t.title)
        
        # Auth -> APIs -> Frontend dependency chain
        asyncio.run(
            self.entanglement_manager.create_entanglement(
                [auth_task, api_task], SimpleEntanglementType.DEPENDENCY, 0.9
            )
        )
        
        asyncio.run(
            self.entanglement_manager.create_entanglement(
                [api_task, frontend_task], SimpleEntanglementType.DEPENDENCY, 0.8
            )
        )
        
        # Optimize project schedule
        project_schedule = asyncio.run(self.scheduler.optimize_schedule())
        
        # Verify schedule makes sense
        assert len(project_schedule) == len(tasks)
        
        # Critical tasks should be prioritized
        critical_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]
        if critical_tasks:
            critical_positions = [
                i for i, (_, task) in enumerate(project_schedule)
                if task.priority == TaskPriority.CRITICAL
            ]
            # Critical tasks should appear in first half of schedule
            assert all(pos < len(project_schedule) / 2 for pos in critical_positions)
        
        # Generate project metrics
        project_stats = {
            'total_tasks': len(tasks),
            'estimated_total_hours': sum(
                t.estimated_duration.total_seconds() / 3600 
                for t in tasks if t.estimated_duration
            ),
            'average_complexity': sum(t.complexity_factor for t in tasks) / len(tasks),
            'priority_distribution': {
                priority.name: sum(1 for t in tasks if t.priority == priority)
                for priority in TaskPriority
            },
            'entanglement_count': len(self.entanglement_manager.entanglement_bonds),
            'optimization_improvement': 0.15  # Assumed improvement from quantum optimization
        }
        
        # Verify project feasibility
        assert project_stats['total_tasks'] > 10
        assert project_stats['estimated_total_hours'] > 50  # Substantial project
        assert project_stats['average_complexity'] > 1.0
        assert project_stats['entanglement_count'] >= 2  # Has dependencies
    
    def teardown_method(self):
        """Cleanup after tests"""
        self.scheduler.tasks.clear()
        self.entanglement_manager.entanglement_bonds.clear()
        self.cache.clear()


class TestSystemBenchmarks:
    """System-wide performance benchmarks"""
    
    def test_scalability_benchmark(self):
        """Test system scalability with increasing load"""
        scheduler = QuantumTaskScheduler()
        optimizer = SimpleQuantumOptimizer()
        
        # Test different scales
        scales = [10, 50, 100, 500]
        results = {}
        
        for scale in scales:
            # Create tasks
            tasks = [
                QuantumTask(
                    title=f"Scale Test Task {i}",
                    description=f"Scalability test with {scale} tasks",
                    priority=TaskPriority.MEDIUM,
                    complexity_factor=1.0 + (i % 3) * 0.5
                )
                for i in range(scale)
            ]
            
            # Benchmark task addition
            start_time = time.time()
            for task in tasks:
                scheduler.add_task(task)
            add_time = time.time() - start_time
            
            # Benchmark optimization
            start_time = time.time()
            resources = {"cpu": scale * 2, "memory": scale * 0.5}
            optimization_result = asyncio.run(
                optimizer.optimize_task_allocation(tasks, resources)
            )
            opt_time = time.time() - start_time
            
            results[scale] = {
                'add_time': add_time,
                'opt_time': opt_time,
                'add_throughput': scale / add_time,
                'opt_throughput': scale / opt_time
            }
            
            # Clean up for next iteration
            scheduler.tasks.clear()
        
        # Verify performance scales reasonably
        for scale in scales[1:]:
            prev_scale = scales[scales.index(scale) - 1]
            scale_factor = scale / prev_scale
            
            # Performance should not degrade linearly with scale
            time_ratio = results[scale]['opt_time'] / results[prev_scale]['opt_time']
            assert time_ratio < scale_factor * 2  # Should not be more than 2x linear scaling
        
        print(f"\\nScalability Benchmark Results:")
        for scale, metrics in results.items():
            print(f"Scale {scale}: Add={metrics['add_time']:.3f}s, Opt={metrics['opt_time']:.3f}s")


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Comprehensive Quantum Task Planner Tests")
    print("=" * 60)
    
    # Run pytest with verbose output
    import subprocess
    
    result = subprocess.run([
        'python', '-m', 'pytest', __file__, '-v', '--tb=short'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run basic smoke tests without pytest
    print("🧪 Quantum Task Planner - Smoke Tests")
    print("=" * 50)
    
    test_integration = TestQuantumTaskIntegration()
    test_benchmarks = TestSystemBenchmarks()
    
    try:
        print("\\n📋 Task Lifecycle Test...")
        test_integration.setup_method()
        test_integration.test_task_lifecycle()
        print("✅ PASSED")
        
        print("\\n🔗 Entanglement Operations Test...")
        test_integration.test_entanglement_operations()
        print("✅ PASSED")
        
        print("\\n🚀 Performance Benchmarks...")
        test_benchmarks.test_scalability_benchmark()
        print("✅ PASSED")
        
        print("\\n🌍 Real-world Scenario Test...")
        test_integration.test_real_world_scenario()
        print("✅ PASSED")
        
        test_integration.teardown_method()
        
        print("\\n✨ All Smoke Tests Passed!")
        print("\\nFor comprehensive testing, run:")
        print("python -m pytest tests/test_comprehensive_integration.py -v")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()