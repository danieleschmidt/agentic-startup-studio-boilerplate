"""
Performance benchmarks and load testing for Quantum Task Planner
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
from quantum_task_planner.core.quantum_scheduler import QuantumTaskScheduler
from quantum_task_planner.core.quantum_optimizer import QuantumProbabilityOptimizer
from quantum_task_planner.core.entanglement_manager import TaskEntanglementManager

from quantum_task_planner.performance.cache import QuantumCache, get_cache
from quantum_task_planner.performance.concurrent import QuantumWorkerPool
from quantum_task_planner.performance.scaling import AutoScaler, QuantumLoadBalancer


class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    def __init__(self):
        self.results = {}
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return result, (end_time - start_time) * 1000  # Return result and time in ms
    
    async def time_async_function(self, func, *args, **kwargs):
        """Time an async function execution"""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return result, (end_time - start_time) * 1000  # Return result and time in ms


class TestQuantumTaskPerformance(PerformanceBenchmark):
    """Performance tests for quantum task operations"""
    
    def test_task_creation_performance(self):
        """Benchmark task creation performance"""
        iterations = 1000
        times = []
        
        for i in range(iterations):
            _, duration = self.time_function(
                QuantumTask,
                title=f"Benchmark Task {i}",
                description=f"Performance test task {i}",
                priority=TaskPriority.MEDIUM,
                complexity_factor=2.0
            )
            times.append(duration)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        # Performance assertions
        assert avg_time < 1.0  # Average should be under 1ms
        assert p95_time < 5.0  # 95th percentile should be under 5ms
        assert max_time < 50.0  # No single creation should take over 50ms
        
        print(f"Task Creation Performance (n={iterations}):")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  Min: {min_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")
        print(f"  P95: {p95_time:.3f}ms")
    
    def test_quantum_measurement_performance(self):
        """Benchmark quantum measurement performance"""
        tasks = [
            QuantumTask(
                title=f"Measurement Task {i}",
                description="Performance measurement test",
                priority=TaskPriority.MEDIUM
            )
            for i in range(100)
        ]
        
        times = []
        for task in tasks:
            _, duration = self.time_function(task.measure_quantum_state)
            times.append(duration)
        
        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        
        # Performance assertions
        assert avg_time < 0.5  # Average measurement should be under 0.5ms
        assert p95_time < 2.0   # 95th percentile should be under 2ms
        
        print(f"Quantum Measurement Performance (n=100):")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  P95: {p95_time:.3f}ms")
    
    def test_decoherence_batch_performance(self):
        """Benchmark batch decoherence performance"""
        tasks = [
            QuantumTask(
                title=f"Decoherence Task {i}",
                description="Batch decoherence test",
                priority=TaskPriority.MEDIUM
            )
            for i in range(500)
        ]
        
        # Measure batch decoherence
        start_time = time.perf_counter()
        for task in tasks:
            task.apply_decoherence(60.0)  # 1 minute decoherence
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_task = total_time / len(tasks)
        
        # Performance assertions
        assert avg_time_per_task < 0.1  # Average should be under 0.1ms per task
        assert total_time < 100.0       # Total batch should be under 100ms
        
        print(f"Batch Decoherence Performance (n=500):")
        print(f"  Total: {total_time:.3f}ms")
        print(f"  Per Task: {avg_time_per_task:.3f}ms")


class TestSchedulerPerformance(PerformanceBenchmark):
    """Performance tests for quantum scheduler"""
    
    def test_large_scale_scheduling(self):
        """Benchmark scheduling with large number of tasks"""
        scheduler = QuantumTaskScheduler()
        
        # Create large number of tasks
        task_counts = [100, 500, 1000]
        results = {}
        
        for count in task_counts:
            tasks = [
                QuantumTask(
                    title=f"Scale Task {i}",
                    description=f"Large scale test {i}",
                    priority=TaskPriority.MEDIUM if i % 3 != 0 else TaskPriority.HIGH,
                    complexity_factor=float((i % 5) + 1)
                )
                for i in range(count)
            ]
            
            # Add tasks to scheduler
            add_start = time.perf_counter()
            for task in tasks:
                scheduler.add_task(task)
            add_end = time.perf_counter()
            
            add_time = (add_end - add_start) * 1000
            
            # Schedule tasks
            schedule_start = time.perf_counter()
            schedule = scheduler.schedule_tasks()
            schedule_end = time.perf_counter()
            
            schedule_time = (schedule_end - schedule_start) * 1000
            
            results[count] = {
                "add_time": add_time,
                "schedule_time": schedule_time,
                "total_time": add_time + schedule_time,
                "tasks_per_second": count / ((add_time + schedule_time) / 1000)
            }
            
            # Clean up for next iteration
            scheduler.tasks.clear()
        
        # Performance assertions
        for count, result in results.items():
            # Scheduling should be reasonably efficient
            assert result["tasks_per_second"] > 100  # At least 100 tasks/second
            assert result["schedule_time"] < count * 0.1  # Less than 0.1ms per task
        
        print("Large Scale Scheduling Performance:")
        for count, result in results.items():
            print(f"  {count} tasks:")
            print(f"    Add time: {result['add_time']:.2f}ms")
            print(f"    Schedule time: {result['schedule_time']:.2f}ms")
            print(f"    Tasks/sec: {result['tasks_per_second']:.1f}")
    
    @pytest.mark.asyncio
    async def test_async_scheduling_performance(self):
        """Benchmark asynchronous scheduling performance"""
        scheduler = QuantumTaskScheduler()
        
        # Create test tasks
        tasks = [
            QuantumTask(
                title=f"Async Task {i}",
                description=f"Async scheduling test {i}",
                priority=TaskPriority.MEDIUM
            )
            for i in range(200)
        ]
        
        for task in tasks:
            scheduler.add_task(task)
        
        # Measure async scheduling
        _, duration = await self.time_async_function(scheduler.schedule_tasks_async)
        
        # Performance assertion
        assert duration < 500  # Should complete within 500ms for 200 tasks
        
        print(f"Async Scheduling Performance (n=200): {duration:.2f}ms")


class TestOptimizationPerformance(PerformanceBenchmark):
    """Performance tests for quantum optimization"""
    
    def test_genetic_algorithm_performance(self):
        """Benchmark genetic algorithm optimization"""
        optimizer = QuantumProbabilityOptimizer()
        
        # Add objectives
        objectives = optimizer.create_standard_objectives()
        for obj in objectives:
            optimizer.add_objective(obj)
        
        # Test different task set sizes
        task_counts = [10, 50, 100]
        results = {}
        
        for count in task_counts:
            tasks = [
                QuantumTask(
                    title=f"Optimization Task {i}",
                    description=f"GA performance test {i}",
                    priority=TaskPriority.MEDIUM,
                    complexity_factor=float((i % 3) + 1)
                )
                for i in range(count)
            ]
            
            # Benchmark optimization
            start_time = time.perf_counter()
            result = optimizer.optimize_genetic_algorithm(
                tasks,
                max_iterations=50,
                population_size=30
            )
            end_time = time.perf_counter()
            
            duration = (end_time - start_time) * 1000
            
            results[count] = {
                "duration": duration,
                "fitness": result.get("best_fitness", 0),
                "tasks_per_second": count / (duration / 1000)
            }
        
        # Performance assertions
        for count, result in results.items():
            # Optimization should complete in reasonable time
            assert result["duration"] < count * 50  # Less than 50ms per task
            assert result["fitness"] >= 0.0  # Valid fitness value
        
        print("Genetic Algorithm Performance:")
        for count, result in results.items():
            print(f"  {count} tasks: {result['duration']:.2f}ms, fitness: {result['fitness']:.3f}")


class TestCachePerformance(PerformanceBenchmark):
    """Performance tests for quantum cache"""
    
    def test_cache_operations_performance(self):
        """Benchmark cache set/get operations"""
        cache = QuantumCache(max_size=10000)
        
        # Benchmark cache writes
        write_times = []
        for i in range(1000):
            key = f"test_key_{i}"
            value = {"data": f"test_value_{i}", "quantum_state": 0.8}
            
            _, duration = self.time_function(
                cache.set, key, value, quantum_coherence=0.8
            )
            write_times.append(duration)
        
        avg_write_time = statistics.mean(write_times)
        
        # Benchmark cache reads
        read_times = []
        for i in range(1000):
            key = f"test_key_{i}"
            _, duration = self.time_function(cache.get, key)
            read_times.append(duration)
        
        avg_read_time = statistics.mean(read_times)
        
        # Performance assertions
        assert avg_write_time < 0.1  # Average write under 0.1ms
        assert avg_read_time < 0.05  # Average read under 0.05ms
        
        print(f"Cache Performance (n=1000):")
        print(f"  Average write: {avg_write_time:.4f}ms")
        print(f"  Average read: {avg_read_time:.4f}ms")
    
    def test_cache_eviction_performance(self):
        """Benchmark cache eviction performance"""
        cache = QuantumCache(max_size=100)
        
        # Fill cache beyond capacity to trigger evictions
        start_time = time.perf_counter()
        
        for i in range(500):  # 5x capacity
            key = f"eviction_key_{i}"
            value = {"data": f"eviction_value_{i}"}
            cache.set(key, value, quantum_coherence=0.5)
        
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        
        # Performance assertion
        assert total_time < 1000  # Should complete within 1 second
        assert len(cache._cache) <= 100  # Should maintain size limit
        
        print(f"Cache Eviction Performance: {total_time:.2f}ms for 500 ops")


class TestConcurrencyPerformance(PerformanceBenchmark):
    """Performance tests for concurrent operations"""
    
    @pytest.mark.asyncio
    async def test_worker_pool_performance(self):
        """Benchmark quantum worker pool performance"""
        worker_pool = QuantumWorkerPool(max_workers=8, worker_type="thread")
        
        def cpu_intensive_task(n):
            """CPU-intensive task for benchmarking"""
            total = 0
            for i in range(n):
                total += i ** 2
            return total
        
        # Submit many tasks
        task_count = 100
        task_size = 1000
        
        start_time = time.perf_counter()
        
        # Submit all tasks
        task_ids = []
        for i in range(task_count):
            task_id = worker_pool.submit_quantum_task(
                cpu_intensive_task, 
                task_size,
                quantum_coherence=0.8,
                priority=1.0
            )
            task_ids.append(task_id)
        
        # Collect all results
        results = []
        for task_id in task_ids:
            try:
                result = worker_pool.get_task_result(task_id, timeout=30.0)
                results.append(result)
            except Exception as e:
                print(f"Task {task_id} failed: {e}")
        
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        tasks_per_second = len(results) / (total_time / 1000)
        
        # Performance assertions
        assert len(results) >= task_count * 0.9  # At least 90% success rate
        assert tasks_per_second > 10  # At least 10 tasks per second
        
        print(f"Worker Pool Performance:")
        print(f"  Completed: {len(results)}/{task_count} tasks")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Tasks/sec: {tasks_per_second:.2f}")
        
        worker_pool.shutdown()
    
    def test_concurrent_task_creation(self):
        """Benchmark concurrent task creation"""
        def create_tasks(start_idx, count):
            tasks = []
            for i in range(start_idx, start_idx + count):
                task = QuantumTask(
                    title=f"Concurrent Task {i}",
                    description=f"Concurrent creation test {i}",
                    priority=TaskPriority.MEDIUM
                )
                tasks.append(task)
            return tasks
        
        # Create tasks concurrently
        thread_count = 4
        tasks_per_thread = 250
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            for i in range(thread_count):
                start_idx = i * tasks_per_thread
                future = executor.submit(create_tasks, start_idx, tasks_per_thread)
                futures.append(future)
            
            all_tasks = []
            for future in as_completed(futures):
                tasks = future.result()
                all_tasks.extend(tasks)
        
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        total_tasks = len(all_tasks)
        tasks_per_second = total_tasks / (total_time / 1000)
        
        # Performance assertions
        assert total_tasks == thread_count * tasks_per_thread
        assert tasks_per_second > 1000  # Should create over 1000 tasks/second
        
        print(f"Concurrent Task Creation:")
        print(f"  Created: {total_tasks} tasks")
        print(f"  Time: {total_time:.2f}ms")
        print(f"  Tasks/sec: {tasks_per_second:.1f}")


class TestMemoryPerformance(PerformanceBenchmark):
    """Memory usage and performance tests"""
    
    def test_memory_usage_scaling(self):
        """Test memory usage with increasing task counts"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        scheduler = QuantumTaskScheduler()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test memory growth with different task counts
        task_counts = [1000, 5000, 10000]
        memory_results = {}
        
        for count in task_counts:
            # Create tasks
            tasks = [
                QuantumTask(
                    title=f"Memory Test Task {i}",
                    description=f"Memory usage test {i}",
                    priority=TaskPriority.MEDIUM,
                    complexity_factor=2.0
                )
                for i in range(count)
            ]
            
            # Add to scheduler
            for task in tasks:
                scheduler.add_task(task)
            
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - initial_memory
            memory_per_task = memory_used / count * 1024  # KB per task
            
            memory_results[count] = {
                "total_memory_mb": current_memory,
                "memory_used_mb": memory_used,
                "memory_per_task_kb": memory_per_task
            }
            
            # Clean up for next test
            scheduler.tasks.clear()
        
        # Memory efficiency assertions
        for count, result in memory_results.items():
            # Memory per task should be reasonable (less than 10KB per task)
            assert result["memory_per_task_kb"] < 10.0
        
        print("Memory Usage Scaling:")
        for count, result in memory_results.items():
            print(f"  {count} tasks:")
            print(f"    Total memory: {result['total_memory_mb']:.1f}MB")
            print(f"    Memory per task: {result['memory_per_task_kb']:.2f}KB")


class TestLoadTesting:
    """Load testing for the entire system"""
    
    @pytest.mark.asyncio
    async def test_system_load_test(self):
        """Comprehensive system load test"""
        # Initialize all components
        scheduler = QuantumTaskScheduler()
        optimizer = QuantumProbabilityOptimizer()
        entanglement_manager = TaskEntanglementManager()
        cache = get_cache("load_test")
        
        # Add optimization objectives
        objectives = optimizer.create_standard_objectives()
        for obj in objectives:
            optimizer.add_objective(obj)
        
        # Simulation parameters
        num_tasks = 1000
        num_entanglements = 100
        optimization_rounds = 5
        
        print(f"Starting system load test with {num_tasks} tasks...")
        
        start_time = time.perf_counter()
        
        # Phase 1: Create and schedule tasks
        tasks = []
        for i in range(num_tasks):
            task = QuantumTask(
                title=f"Load Test Task {i}",
                description=f"System load test task {i}",
                priority=TaskPriority.HIGH if i % 10 == 0 else TaskPriority.MEDIUM,
                complexity_factor=float((i % 5) + 1)
            )
            tasks.append(task)
            scheduler.add_task(task)
            
            # Cache some task data
            if i % 50 == 0:
                cache.set(f"task_{i}", task.to_dict(), quantum_coherence=task.quantum_coherence)
        
        phase1_time = time.perf_counter()
        print(f"Phase 1 (Task Creation): {(phase1_time - start_time) * 1000:.2f}ms")
        
        # Phase 2: Create entanglements
        for i in range(0, min(num_entanglements * 2, num_tasks), 2):
            if i + 1 < len(tasks):
                entanglement_manager.create_entanglement_bond(
                    [tasks[i].task_id, tasks[i+1].task_id],
                    entanglement_type="bell_state",
                    strength=0.8
                )
        
        phase2_time = time.perf_counter()
        print(f"Phase 2 (Entanglement): {(phase2_time - phase1_time) * 1000:.2f}ms")
        
        # Phase 3: Schedule tasks
        schedule = scheduler.schedule_tasks()
        
        phase3_time = time.perf_counter()
        print(f"Phase 3 (Scheduling): {(phase3_time - phase2_time) * 1000:.2f}ms")
        
        # Phase 4: Run optimizations
        for round_num in range(optimization_rounds):
            # Select random subset of tasks for optimization
            import random
            subset_size = min(50, num_tasks)
            task_subset = random.sample(tasks, subset_size)
            
            optimizer.optimize_genetic_algorithm(
                task_subset,
                max_iterations=20,
                population_size=30
            )
        
        phase4_time = time.perf_counter()
        print(f"Phase 4 (Optimization): {(phase4_time - phase3_time) * 1000:.2f}ms")
        
        # Phase 5: Apply decoherence and measurements
        for i, task in enumerate(tasks):
            if i % 10 == 0:  # Measure every 10th task
                task.measure_quantum_state()
            
            if i % 100 == 0:  # Apply decoherence every 100th task
                task.apply_decoherence(30.0)
        
        # Apply system-wide decoherence
        await entanglement_manager.apply_decoherence(60.0)
        
        phase5_time = time.perf_counter()
        print(f"Phase 5 (Quantum Ops): {(phase5_time - phase4_time) * 1000:.2f}ms")
        
        total_time = (phase5_time - start_time) * 1000
        
        # Performance assertions
        assert total_time < 30000  # Should complete within 30 seconds
        assert len(scheduler.tasks) == num_tasks
        assert len(schedule) <= num_tasks
        
        # System health checks
        avg_coherence = sum(task.quantum_coherence for task in tasks) / len(tasks)
        assert avg_coherence > 0.1  # System should maintain some coherence
        
        print(f"Load Test Results:")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Tasks processed: {num_tasks}")
        print(f"  Tasks/second: {num_tasks / (total_time / 1000):.1f}")
        print(f"  Average coherence: {avg_coherence:.3f}")
        print(f"  Entanglements created: {len(entanglement_manager.entanglement_bonds)}")
        print(f"  Cache entries: {len(cache._cache)}")
    
    def test_stress_test_concurrent_operations(self):
        """Stress test with many concurrent operations"""
        import threading
        
        scheduler = QuantumTaskScheduler()
        results = {"tasks_created": 0, "measurements_taken": 0, "errors": 0}
        results_lock = threading.Lock()
        
        def create_and_measure_tasks(thread_id, num_tasks):
            """Worker function for concurrent operations"""
            try:
                for i in range(num_tasks):
                    # Create task
                    task = QuantumTask(
                        title=f"Stress Task {thread_id}-{i}",
                        description=f"Stress test task from thread {thread_id}",
                        priority=TaskPriority.MEDIUM
                    )
                    
                    with results_lock:
                        scheduler.add_task(task)
                        results["tasks_created"] += 1
                    
                    # Perform quantum measurement
                    task.measure_quantum_state()
                    
                    with results_lock:
                        results["measurements_taken"] += 1
                    
                    # Small delay to simulate real work
                    time.sleep(0.001)
                    
            except Exception as e:
                with results_lock:
                    results["errors"] += 1
                print(f"Thread {thread_id} error: {e}")
        
        # Run stress test
        num_threads = 10
        tasks_per_thread = 100
        
        threads = []
        start_time = time.perf_counter()
        
        for i in range(num_threads):
            thread = threading.Thread(
                target=create_and_measure_tasks,
                args=(i, tasks_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000
        
        # Performance assertions
        expected_tasks = num_threads * tasks_per_thread
        assert results["tasks_created"] == expected_tasks
        assert results["measurements_taken"] == expected_tasks
        assert results["errors"] == 0  # No errors should occur
        
        print(f"Stress Test Results:")
        print(f"  Threads: {num_threads}")
        print(f"  Tasks per thread: {tasks_per_thread}")
        print(f"  Total tasks: {results['tasks_created']}")
        print(f"  Total measurements: {results['measurements_taken']}")
        print(f"  Errors: {results['errors']}")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Operations/sec: {(results['tasks_created'] + results['measurements_taken']) / (total_time / 1000):.1f}")


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "-s",  # Show print statements
        "--tb=short"
    ])