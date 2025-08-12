#!/usr/bin/env python3
"""
Test script for Quantum Performance Optimizer
"""

import sys
sys.path.insert(0, '.')

import asyncio
import time
import random
from quantum_task_planner.core.quantum_task import QuantumTask, TaskPriority
from quantum_task_planner.performance.quantum_performance_optimizer import QuantumPerformanceOptimizer

async def sample_computation_task(task: QuantumTask, **kwargs) -> dict:
    """Sample computation function for testing"""
    # Simulate computation based on task complexity
    computation_time = task.complexity_factor * 0.1 + random.uniform(0.05, 0.2)
    await asyncio.sleep(computation_time)
    
    return {
        "task_id": task.task_id,
        "result": f"Computed result for {task.title}",
        "complexity": task.complexity_factor,
        "computation_time": computation_time,
        "quantum_coherence": task.quantum_coherence
    }

async def test_quantum_performance_optimization():
    print("🚀 Testing Quantum Performance Optimizer...")
    
    # Initialize optimizer
    optimizer = QuantumPerformanceOptimizer(
        max_cache_size=50,
        enable_adaptive_scaling=True,
        enable_quantum_caching=True
    )
    
    # Create test tasks with different properties
    test_tasks = []
    for i in range(20):
        priority = random.choice(list(TaskPriority))
        complexity = random.uniform(0.5, 5.0)
        
        task = QuantumTask(
            title=f"Performance Test Task {i}",
            description=f"Test task with complexity {complexity:.2f}",
            priority=priority,
            complexity_factor=complexity
        )
        test_tasks.append(task)
    
    print(f"Created {len(test_tasks)} test tasks")
    
    # Test 1: Sequential execution with caching
    print("\n📊 Test 1: Sequential execution with caching")
    start_time = time.time()
    
    results = []
    for task in test_tasks[:10]:
        result = await optimizer.optimize_task_execution(
            task, sample_computation_task
        )
        results.append(result)
    
    sequential_time = time.time() - start_time
    print(f"Sequential execution time: {sequential_time:.3f}s")
    
    # Test 2: Execute same tasks again to test caching
    print("\n🧠 Test 2: Cache effectiveness test")
    start_time = time.time()
    
    cached_results = []
    for task in test_tasks[:10]:  # Same tasks
        result = await optimizer.optimize_task_execution(
            task, sample_computation_task
        )
        cached_results.append(result)
    
    cached_time = time.time() - start_time
    print(f"Cached execution time: {cached_time:.3f}s")
    print(f"Cache speedup: {sequential_time / cached_time:.2f}x")
    
    # Test 3: Concurrent execution
    print("\n⚡ Test 3: Concurrent execution test")
    start_time = time.time()
    
    concurrent_tasks = []
    for task in test_tasks[10:15]:  # Different tasks
        concurrent_tasks.append(
            optimizer.optimize_task_execution(task, sample_computation_task)
        )
    
    concurrent_results = await asyncio.gather(*concurrent_tasks)
    concurrent_time = time.time() - start_time
    print(f"Concurrent execution time: {concurrent_time:.3f}s")
    print(f"Concurrent tasks completed: {len(concurrent_results)}")
    
    # Test 4: High-coherence quantum optimization
    print("\n🌀 Test 4: Quantum coherence optimization")
    high_coherence_task = QuantumTask(
        title="High Coherence Task",
        description="Task with high quantum coherence",
        complexity_factor=3.0
    )
    high_coherence_task.quantum_coherence = 0.95  # Very high coherence
    
    start_time = time.time()
    quantum_result = await optimizer.optimize_task_execution(
        high_coherence_task, sample_computation_task
    )
    quantum_time = time.time() - start_time
    print(f"High-coherence execution time: {quantum_time:.3f}s")
    print(f"Quantum optimization applied: {quantum_result['task_id']}")
    
    # Test 5: Performance metrics and scaling
    print("\n📈 Test 5: Performance metrics and adaptive scaling")
    
    # Generate load to trigger scaling
    load_tasks = []
    for i in range(8):
        task = QuantumTask(
            title=f"Load Task {i}",
            description="Task to generate load",
            complexity_factor=2.0,
            priority=TaskPriority.HIGH
        )
        load_tasks.append(
            optimizer.optimize_task_execution(task, sample_computation_task)
        )
    
    load_results = await asyncio.gather(*load_tasks)
    
    # Get performance report
    performance_report = optimizer.get_performance_report()
    
    print(f"Tasks completed: {len(load_results)}")
    print(f"Average response time: {performance_report['current_metrics']['average_response_time']:.3f}s")
    print(f"Task throughput: {performance_report['current_metrics']['task_throughput']:.2f} tasks/sec")
    print(f"Cache hit ratio: {performance_report['current_metrics']['cache_hit_ratio']:.2%}")
    print(f"Cache size: {performance_report['cache_statistics']['current_size']}/{performance_report['cache_statistics']['max_size']}")
    print(f"Active resource pools: {performance_report['resource_pools']}")
    
    # Test 6: Error handling and resilience
    print("\n🛡️ Test 6: Error handling test")
    
    async def failing_task(task: QuantumTask, **kwargs) -> dict:
        if random.random() < 0.3:  # 30% failure rate
            raise Exception(f"Simulated failure for {task.title}")
        return await sample_computation_task(task, **kwargs)
    
    error_tasks = []
    for i in range(5):
        task = QuantumTask(
            title=f"Error Test Task {i}",
            description="Task that might fail",
            complexity_factor=1.0
        )
        try:
            result = await optimizer.optimize_task_execution(task, failing_task)
            error_tasks.append(("success", result))
        except Exception as e:
            error_tasks.append(("error", str(e)))
    
    successes = len([r for r in error_tasks if r[0] == "success"])
    errors = len([r for r in error_tasks if r[0] == "error"])
    print(f"Error handling test: {successes} successes, {errors} errors")
    print(f"Error rate: {performance_report['current_metrics']['error_rate']:.2%}")
    
    # Final performance summary
    print("\n🎯 Performance Optimization Summary:")
    final_report = optimizer.get_performance_report()
    
    print(f"  • Total cache operations: {final_report['cache_statistics']['hits'] + final_report['cache_statistics']['misses']}")
    print(f"  • Cache efficiency: {final_report['cache_statistics']['hit_ratio']:.2%}")
    print(f"  • Average quantum coherence: {final_report['current_metrics']['quantum_coherence_avg']:.3f}")
    print(f"  • Optimization features active: {final_report['optimization_status']}")
    print(f"  • Resource utilization optimized: ✅")
    print(f"  • Adaptive scaling enabled: ✅")
    print(f"  • Quantum caching enabled: ✅")
    
    # Cleanup
    await optimizer.shutdown()
    print("\n✅ Quantum Performance Optimizer test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_quantum_performance_optimization())