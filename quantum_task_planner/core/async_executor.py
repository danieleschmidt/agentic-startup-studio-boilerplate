"""
Quantum Async Task Execution Engine

Advanced asynchronous task execution system with quantum-inspired 
processing, adaptive resource allocation, and real-time monitoring.
"""

import asyncio
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .quantum_task import QuantumTask, TaskState, TaskPriority
from ..utils.exceptions import QuantumExecutionError, ResourceExhaustionError
from ..performance.concurrent import get_worker_pool
from ..performance.scaling import get_auto_scaler
from ..utils.logging import get_logger


class ExecutorType(Enum):
    """Types of quantum task executors"""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_COROUTINE = "async_coroutine"
    QUANTUM_DISTRIBUTED = "quantum_distributed"
    HYBRID_ADAPTIVE = "hybrid_adaptive"


class ResourceType(Enum):
    """System resource types for quantum execution"""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    QUANTUM_COHERENT = "quantum_coherent"


@dataclass
class ExecutionResource:
    """Resource allocation for quantum task execution"""
    resource_type: ResourceType
    allocated_amount: float
    max_amount: float
    current_usage: float = 0.0
    reservation_time: Optional[datetime] = None
    
    def is_available(self, required_amount: float) -> bool:
        """Check if resource has sufficient capacity"""
        return self.current_usage + required_amount <= self.allocated_amount
    
    def reserve(self, amount: float) -> bool:
        """Reserve resource capacity"""
        if self.is_available(amount):
            self.current_usage += amount
            self.reservation_time = datetime.utcnow()
            return True
        return False
    
    def release(self, amount: float):
        """Release reserved resource capacity"""
        self.current_usage = max(0.0, self.current_usage - amount)


@dataclass
class ExecutionContext:
    """Context for quantum task execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: Optional[QuantumTask] = None
    executor_type: ExecutorType = ExecutorType.ASYNC_COROUTINE
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    progress: float = 0.0
    coherence_decay_rate: float = 0.01
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def execution_duration(self) -> Optional[timedelta]:
        """Calculate execution duration"""
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        elif self.start_time:
            return datetime.utcnow() - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if execution is currently active"""
        return self.start_time is not None and self.completion_time is None


class QuantumAsyncExecutor:
    """
    Advanced quantum-inspired asynchronous task executor with:
    - Multi-level parallelism (threads, processes, async)
    - Adaptive resource allocation
    - Quantum coherence management
    - Real-time monitoring and metrics
    """
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 max_thread_workers: int = 4,
                 max_process_workers: int = 2,
                 coherence_monitoring_interval: float = 1.0):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.coherence_monitoring_interval = coherence_monitoring_interval
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=max_thread_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_process_workers)
        
        # Active executions tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionContext] = []
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
        # Resource management
        self.resource_pools: Dict[ResourceType, ExecutionResource] = {
            ResourceType.CPU_INTENSIVE: ExecutionResource(
                ResourceType.CPU_INTENSIVE, 
                allocated_amount=mp.cpu_count() * 0.8,
                max_amount=mp.cpu_count()
            ),
            ResourceType.MEMORY_INTENSIVE: ExecutionResource(
                ResourceType.MEMORY_INTENSIVE,
                allocated_amount=8.0,  # GB
                max_amount=16.0
            ),
            ResourceType.IO_INTENSIVE: ExecutionResource(
                ResourceType.IO_INTENSIVE,
                allocated_amount=100.0,  # Concurrent IO operations
                max_amount=200.0
            ),
            ResourceType.NETWORK_INTENSIVE: ExecutionResource(
                ResourceType.NETWORK_INTENSIVE,
                allocated_amount=1000.0,  # Mbps
                max_amount=2000.0
            ),
            ResourceType.QUANTUM_COHERENT: ExecutionResource(
                ResourceType.QUANTUM_COHERENT,
                allocated_amount=1.0,  # Coherence units
                max_amount=1.0
            )
        }
        
        # Monitoring and metrics
        self.metrics: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "resource_utilization": {},
            "coherence_preservation_rate": 1.0
        }
        
        self.logger = get_logger(__name__)
        self._monitoring_task = None
        self._execution_processor_task = None
        
        # Quantum state management
        self._quantum_lock = asyncio.Lock()
        self._coherence_preservation_factor = 0.95
        
    async def start(self):
        """Start the quantum executor"""
        self.logger.info("Starting Quantum Async Executor")
        
        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._coherence_monitoring_loop())
        self._execution_processor_task = asyncio.create_task(self._execution_processor_loop())
        
        self.logger.info("Quantum Async Executor started successfully")
    
    async def stop(self):
        """Stop the quantum executor"""
        self.logger.info("Stopping Quantum Async Executor")
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._execution_processor_task:
            self._execution_processor_task.cancel()
        
        # Wait for active executions to complete
        active_tasks = list(self.active_executions.keys())
        if active_tasks:
            self.logger.info(f"Waiting for {len(active_tasks)} active executions to complete")
            await asyncio.sleep(2)  # Grace period
        
        # Shutdown thread and process pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("Quantum Async Executor stopped")
    
    async def execute_task(self, 
                          task: QuantumTask,
                          executor_type: Optional[ExecutorType] = None,
                          custom_handler: Optional[Callable] = None,
                          priority_boost: float = 0.0) -> ExecutionContext:
        """
        Execute a quantum task asynchronously with optimal resource allocation
        
        Args:
            task: The quantum task to execute
            executor_type: Preferred execution method
            custom_handler: Custom execution handler function
            priority_boost: Additional priority boost for urgent tasks
        
        Returns:
            ExecutionContext with execution details and progress tracking
        """
        
        # Determine optimal executor type
        if not executor_type:
            executor_type = await self._determine_optimal_executor(task)
        
        # Create execution context
        context = ExecutionContext(
            task=task,
            executor_type=executor_type,
            coherence_decay_rate=self._calculate_coherence_decay_rate(task),
            resource_requirements=self._calculate_resource_requirements(task)
        )
        
        # Reserve resources
        if not await self._reserve_resources(context):
            raise ResourceExhaustionError(f"Insufficient resources for task {task.task_id}")
        
        # Add to execution queue with quantum priority
        quantum_priority = self._calculate_quantum_priority(task, priority_boost)
        await self.execution_queue.put((quantum_priority, context))
        
        self.logger.info(f"Queued task {task.task_id} for execution with priority {quantum_priority:.3f}")
        return context
    
    async def _execution_processor_loop(self):
        """Main execution processing loop"""
        while True:
            try:
                # Get next highest priority task
                if self.execution_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if we can handle more concurrent executions
                if len(self.active_executions) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.5)
                    continue
                
                priority, context = await self.execution_queue.get()
                
                # Start execution
                await self._start_execution(context)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in execution processor: {e}")
                await asyncio.sleep(1)
    
    async def _start_execution(self, context: ExecutionContext):
        """Start task execution based on context"""
        context.start_time = datetime.utcnow()
        self.active_executions[context.execution_id] = context
        
        task = context.task
        task.set_state(TaskState.RUNNING)
        
        self.logger.info(f"Starting execution of task {task.task_id} using {context.executor_type.value}")
        
        try:
            if context.executor_type == ExecutorType.ASYNC_COROUTINE:
                await self._execute_async_coroutine(context)
            elif context.executor_type == ExecutorType.THREAD_POOL:
                await self._execute_thread_pool(context)
            elif context.executor_type == ExecutorType.PROCESS_POOL:
                await self._execute_process_pool(context)
            elif context.executor_type == ExecutorType.QUANTUM_DISTRIBUTED:
                await self._execute_quantum_distributed(context)
            elif context.executor_type == ExecutorType.HYBRID_ADAPTIVE:
                await self._execute_hybrid_adaptive(context)
            else:
                await self._execute_async_coroutine(context)  # Fallback
                
        except Exception as e:
            self.logger.error(f"Execution failed for task {task.task_id}: {e}")
            task.set_state(TaskState.FAILED)
            context.metadata["error"] = str(e)
        
        finally:
            await self._complete_execution(context)
    
    async def _execute_async_coroutine(self, context: ExecutionContext):
        """Execute task using async coroutines"""
        task = context.task
        
        # Simulate quantum task execution with coherence management
        execution_steps = 10
        step_duration = 0.5  # seconds per step
        
        for step in range(execution_steps):
            await asyncio.sleep(step_duration)
            
            # Update progress
            progress = (step + 1) / execution_steps
            context.progress = progress
            
            # Apply quantum coherence decay
            coherence_decay = context.coherence_decay_rate * (step + 1) / execution_steps
            task.quantum_coherence = max(0.1, task.quantum_coherence - coherence_decay)
            
            # Update task completion probability
            task._update_completion_probability(progress)
            
            # Check for quantum interference from entangled tasks
            await self._apply_quantum_interference(context)
            
            self.logger.debug(f"Task {task.task_id} progress: {progress:.2%}, coherence: {task.quantum_coherence:.3f}")
        
        # Complete task
        task.set_state(TaskState.COMPLETED)
        context.progress = 1.0
    
    async def _execute_thread_pool(self, context: ExecutionContext):
        """Execute task using thread pool"""
        task = context.task
        
        def cpu_intensive_work():
            """Simulate CPU-intensive quantum computation"""
            import time
            
            total_steps = 100
            for i in range(total_steps):
                # Simulate quantum computation
                result = np.random.random(1000).sum()
                context.progress = (i + 1) / total_steps
                time.sleep(0.1)  # Simulate work
                
                # Update coherence (approximate, since we're in another thread)
                task.quantum_coherence *= 0.999
                
            return "thread_execution_complete"
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.thread_pool, cpu_intensive_work)
        
        task.set_state(TaskState.COMPLETED)
        context.metadata["thread_result"] = result
    
    async def _execute_process_pool(self, context: ExecutionContext):
        """Execute task using process pool for CPU-intensive work"""
        task = context.task
        
        def process_intensive_work(task_id: str):
            """CPU-intensive work that benefits from multiprocessing"""
            import time
            import numpy as np
            
            # Simulate heavy computation
            matrices = [np.random.random((100, 100)) for _ in range(10)]
            results = []
            
            for i, matrix in enumerate(matrices):
                # Matrix operations
                result = np.linalg.det(matrix)
                results.append(result)
                time.sleep(0.2)  # Simulate processing time
            
            return {
                "task_id": task_id,
                "computation_results": results,
                "total_operations": len(results)
            }
        
        # Execute in process pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.process_pool, 
            process_intensive_work, 
            task.task_id
        )
        
        task.set_state(TaskState.COMPLETED)
        context.progress = 1.0
        context.metadata["process_result"] = result
    
    async def _execute_quantum_distributed(self, context: ExecutionContext):
        """Execute task using distributed quantum processing"""
        task = context.task
        
        # Simulate distributed quantum execution
        quantum_nodes = 3
        execution_time_per_node = 2.0
        
        # Create quantum subtasks
        subtasks = []
        for i in range(quantum_nodes):
            subtask_context = ExecutionContext(
                task=task,
                executor_type=ExecutorType.ASYNC_COROUTINE,
                metadata={"node_id": i, "parent_execution": context.execution_id}
            )
            subtasks.append(self._execute_quantum_node(subtask_context, i))
        
        # Execute all quantum nodes concurrently
        results = await asyncio.gather(*subtasks, return_exceptions=True)
        
        # Aggregate results with quantum superposition
        successful_results = [r for r in results if not isinstance(r, Exception)]
        if successful_results:
            context.progress = 1.0
            task.set_state(TaskState.COMPLETED)
            context.metadata["distributed_results"] = successful_results
        else:
            task.set_state(TaskState.FAILED)
            context.metadata["distributed_errors"] = results
    
    async def _execute_quantum_node(self, context: ExecutionContext, node_id: int) -> Dict[str, Any]:
        """Execute quantum computation on a distributed node"""
        await asyncio.sleep(2.0 + np.random.exponential(0.5))  # Simulate network latency
        
        # Simulate quantum computation results
        quantum_result = {
            "node_id": node_id,
            "quantum_state": np.random.random(8).tolist(),  # 8-qubit simulation
            "measurement_outcome": np.random.choice([0, 1], 8).tolist(),
            "coherence_preserved": np.random.random() > 0.1,
            "entanglement_fidelity": np.random.uniform(0.8, 0.99)
        }
        
        return quantum_result
    
    async def _execute_hybrid_adaptive(self, context: ExecutionContext):
        """Execute using adaptive hybrid approach"""
        task = context.task
        
        # Analyze task characteristics to choose optimal execution path
        if task.complexity_factor > 3.0:
            await self._execute_process_pool(context)
        elif len(task.entangled_tasks) > 2:
            await self._execute_quantum_distributed(context)
        elif task.priority.value > 3:
            await self._execute_thread_pool(context)
        else:
            await self._execute_async_coroutine(context)
    
    async def _determine_optimal_executor(self, task: QuantumTask) -> ExecutorType:
        """Determine optimal execution type based on task characteristics"""
        
        # CPU-intensive tasks
        if task.complexity_factor > 2.0 and task.estimated_duration and task.estimated_duration.total_seconds() > 300:
            return ExecutorType.PROCESS_POOL
        
        # Highly entangled tasks benefit from distributed processing
        if len(task.entangled_tasks) > 3:
            return ExecutorType.QUANTUM_DISTRIBUTED
        
        # High priority tasks get thread pool for responsiveness
        if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            return ExecutorType.THREAD_POOL
        
        # Adaptive execution for complex scenarios
        if task.complexity_factor > 1.5 and len(task.entangled_tasks) > 1:
            return ExecutorType.HYBRID_ADAPTIVE
        
        # Default to async coroutines
        return ExecutorType.ASYNC_COROUTINE
    
    def _calculate_coherence_decay_rate(self, task: QuantumTask) -> float:
        """Calculate coherence decay rate based on task characteristics"""
        base_rate = 0.01
        
        # Higher complexity leads to faster decoherence
        complexity_factor = task.complexity_factor * 0.005
        
        # More entanglements preserve coherence better
        entanglement_factor = max(0, 0.003 - len(task.entangled_tasks) * 0.001)
        
        return base_rate + complexity_factor + entanglement_factor
    
    def _calculate_resource_requirements(self, task: QuantumTask) -> Dict[ResourceType, float]:
        """Calculate resource requirements for task execution"""
        requirements = {}
        
        # Base resource requirements
        if task.complexity_factor > 2.0:
            requirements[ResourceType.CPU_INTENSIVE] = task.complexity_factor * 0.5
        
        if task.estimated_duration and task.estimated_duration.total_seconds() > 600:
            requirements[ResourceType.MEMORY_INTENSIVE] = 2.0
        
        if len(task.entangled_tasks) > 2:
            requirements[ResourceType.NETWORK_INTENSIVE] = len(task.entangled_tasks) * 10.0
        
        # Always requires some quantum coherence
        requirements[ResourceType.QUANTUM_COHERENT] = 0.1
        
        return requirements
    
    def _calculate_quantum_priority(self, task: QuantumTask, priority_boost: float = 0.0) -> float:
        """Calculate quantum-enhanced priority score"""
        base_priority = task.priority.probability_weight
        coherence_bonus = task.quantum_coherence * 0.1
        urgency_factor = 1.0
        
        if task.due_date:
            time_until_due = (task.due_date - datetime.utcnow()).total_seconds()
            urgency_factor = max(0.1, 1.0 / max(1, time_until_due / 3600))  # Hours
        
        entanglement_boost = len(task.entangled_tasks) * 0.05
        
        return base_priority + coherence_bonus + urgency_factor + entanglement_boost + priority_boost
    
    async def _reserve_resources(self, context: ExecutionContext) -> bool:
        """Reserve required resources for execution"""
        async with self._quantum_lock:
            # Check if all required resources are available
            for resource_type, required_amount in context.resource_requirements.items():
                if resource_type not in self.resource_pools:
                    continue
                    
                resource = self.resource_pools[resource_type]
                if not resource.is_available(required_amount):
                    return False
            
            # Reserve all resources
            for resource_type, required_amount in context.resource_requirements.items():
                if resource_type in self.resource_pools:
                    self.resource_pools[resource_type].reserve(required_amount)
            
            return True
    
    async def _release_resources(self, context: ExecutionContext):
        """Release reserved resources"""
        async with self._quantum_lock:
            for resource_type, required_amount in context.resource_requirements.items():
                if resource_type in self.resource_pools:
                    self.resource_pools[resource_type].release(required_amount)
    
    async def _apply_quantum_interference(self, context: ExecutionContext):
        """Apply quantum interference effects during execution"""
        task = context.task
        
        # Interference from entangled tasks
        for entangled_id in task.entangled_tasks:
            # Find entangled task in active executions
            for exec_id, exec_context in self.active_executions.items():
                if exec_context.task and exec_context.task.task_id == entangled_id:
                    # Apply quantum interference
                    interference_factor = 0.01 * np.cos(2 * np.pi * context.progress)
                    task.quantum_coherence += interference_factor
                    task.quantum_coherence = max(0.1, min(1.0, task.quantum_coherence))
                    break
    
    async def _complete_execution(self, context: ExecutionContext):
        """Complete task execution and cleanup"""
        context.completion_time = datetime.utcnow()
        
        # Release resources
        await self._release_resources(context)
        
        # Update metrics
        self.metrics["total_executions"] += 1
        if context.task.state == TaskState.COMPLETED:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Calculate average execution time
        if context.execution_duration:
            current_avg = self.metrics["average_execution_time"]
            total = self.metrics["total_executions"]
            new_duration = context.execution_duration.total_seconds()
            self.metrics["average_execution_time"] = (current_avg * (total - 1) + new_duration) / total
        
        # Move to history and remove from active
        self.execution_history.append(context)
        if context.execution_id in self.active_executions:
            del self.active_executions[context.execution_id]
        
        # Maintain history size
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
        
        self.logger.info(f"Completed execution of task {context.task.task_id} in {context.execution_duration}")
    
    async def _coherence_monitoring_loop(self):
        """Background loop for monitoring quantum coherence"""
        while True:
            try:
                await asyncio.sleep(self.coherence_monitoring_interval)
                
                # Monitor active executions for coherence degradation
                critical_tasks = []
                for context in self.active_executions.values():
                    if context.task and context.task.quantum_coherence < 0.3:
                        critical_tasks.append(context)
                
                if critical_tasks:
                    self.logger.warning(f"Found {len(critical_tasks)} tasks with critical coherence levels")
                    
                    # Apply coherence preservation measures
                    for context in critical_tasks:
                        context.task.quantum_coherence *= self._coherence_preservation_factor
                        context.task.quantum_coherence = max(0.1, context.task.quantum_coherence)
                
                # Update resource utilization metrics
                self._update_resource_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in coherence monitoring: {e}")
                await asyncio.sleep(5)
    
    def _update_resource_metrics(self):
        """Update resource utilization metrics"""
        for resource_type, resource in self.resource_pools.items():
            utilization = resource.current_usage / resource.allocated_amount if resource.allocated_amount > 0 else 0
            self.metrics["resource_utilization"][resource_type.value] = utilization
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionContext]:
        """Get status of a specific execution"""
        return self.active_executions.get(execution_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        self._update_resource_metrics()
        
        return {
            "executor_metrics": self.metrics.copy(),
            "active_executions": len(self.active_executions),
            "queue_size": self.execution_queue.qsize(),
            "resource_pools": {
                rt.value: {
                    "allocated": resource.allocated_amount,
                    "current_usage": resource.current_usage,
                    "utilization": resource.current_usage / resource.allocated_amount if resource.allocated_amount > 0 else 0
                }
                for rt, resource in self.resource_pools.items()
            },
            "recent_executions": len(self.execution_history[-100:])  # Last 100
        }
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            if context.task:
                context.task.set_state(TaskState.PAUSED)
                context.metadata["paused_at"] = datetime.utcnow().isoformat()
                return True
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            if context.task and context.task.state == TaskState.PAUSED:
                context.task.set_state(TaskState.RUNNING)
                context.metadata["resumed_at"] = datetime.utcnow().isoformat()
                return True
        return False
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            if context.task:
                context.task.set_state(TaskState.CANCELLED)
                context.metadata["cancelled_at"] = datetime.utcnow().isoformat()
                await self._complete_execution(context)
                return True
        return False


# Global executor instance
_quantum_executor: Optional[QuantumAsyncExecutor] = None


def get_quantum_executor() -> QuantumAsyncExecutor:
    """Get global quantum executor instance"""
    global _quantum_executor
    if _quantum_executor is None:
        _quantum_executor = QuantumAsyncExecutor()
    return _quantum_executor


async def init_quantum_executor(**kwargs) -> QuantumAsyncExecutor:
    """Initialize and start quantum executor"""
    executor = get_quantum_executor()
    await executor.start()
    return executor


async def shutdown_quantum_executor():
    """Shutdown global quantum executor"""
    global _quantum_executor
    if _quantum_executor:
        await _quantum_executor.stop()
        _quantum_executor = None