"""
Quantum Task Planner Concurrent Processing

Advanced concurrent processing with quantum-aware parallelization,
resource pooling, and distributed quantum state management.
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import queue
import weakref

import numpy as np

from ..utils.logging import get_logger
from ..utils.exceptions import QuantumTaskPlannerError


@dataclass
class WorkerStats:
    """Statistics for worker processes/threads"""
    worker_id: str
    created_at: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    current_load: float = 0.0
    quantum_coherence: float = 1.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per task"""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_completed
    
    @property
    def success_rate(self) -> float:
        """Calculate task success rate"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 1.0
        return self.tasks_completed / total_tasks


@dataclass
class QuantumTask:
    """Quantum-aware task for concurrent processing"""
    task_id: str
    task_func: Callable
    args: Tuple
    kwargs: Dict[str, Any]
    priority: float = 1.0
    quantum_coherence: float = 1.0
    entangled_tasks: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


class QuantumWorkerPool:
    """
    Quantum-aware worker pool with adaptive scaling and coherence management
    """
    
    def __init__(self,
                 max_workers: int = None,
                 min_workers: int = 1,
                 worker_type: str = "thread",  # "thread", "process", "async"
                 scaling_factor: float = 0.7,
                 coherence_threshold: float = 0.5):
        
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.min_workers = min_workers
        self.worker_type = worker_type
        self.scaling_factor = scaling_factor
        self.coherence_threshold = coherence_threshold
        
        # Worker management
        self.workers: Dict[str, Any] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.result_futures: Dict[str, concurrent.futures.Future] = {}
        
        # Quantum state management
        self.quantum_states: Dict[str, float] = {}  # task_id -> coherence
        self.entanglement_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Pool state
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.total_tasks_processed = 0
        self.total_processing_time = 0.0
        self.adaptive_scaling_enabled = True
        
        self.logger = get_logger()
        
        # Initialize worker pool
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the worker pool"""
        if self.worker_type == "thread":
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="quantum_worker"
            )
        elif self.worker_type == "process":
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            )
        elif self.worker_type == "async":
            self.executor = None  # Will handle async tasks differently
        
        # Start management threads
        self._start_management_threads()
        self.is_running = True
    
    def _start_management_threads(self):
        """Start background management threads"""
        # Worker monitoring thread
        monitor_thread = threading.Thread(
            target=self._worker_monitor_loop,
            daemon=True,
            name="quantum_monitor"
        )
        monitor_thread.start()
        
        # Adaptive scaling thread
        scaling_thread = threading.Thread(
            target=self._adaptive_scaling_loop,
            daemon=True,
            name="quantum_scaler"
        )
        scaling_thread.start()
        
        # Quantum coherence management thread
        coherence_thread = threading.Thread(
            target=self._coherence_management_loop,
            daemon=True,
            name="quantum_coherence"
        )
        coherence_thread.start()
    
    def submit_quantum_task(self,
                           task_func: Callable,
                           *args,
                           priority: float = 1.0,
                           quantum_coherence: float = 1.0,
                           entangled_tasks: List[str] = None,
                           timeout: Optional[float] = None,
                           **kwargs) -> str:
        """
        Submit a quantum-aware task for processing
        
        Returns:
            Task ID for tracking
        """
        task = QuantumTask(
            task_id=str(uuid.uuid4()),
            task_func=task_func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            quantum_coherence=quantum_coherence,
            entangled_tasks=entangled_tasks or [],
            timeout=timeout
        )
        
        # Register quantum state
        self.quantum_states[task.task_id] = quantum_coherence
        
        # Register entanglements
        for entangled_id in task.entangled_tasks:
            self.entanglement_graph[task.task_id].append(entangled_id)
            self.entanglement_graph[entangled_id].append(task.task_id)
        
        # Submit to appropriate executor
        if self.worker_type in ["thread", "process"]:
            future = self.executor.submit(self._execute_quantum_task, task)
            self.result_futures[task.task_id] = future
        elif self.worker_type == "async":
            # Handle async tasks differently
            future = asyncio.create_task(self._execute_quantum_task_async(task))
            self.result_futures[task.task_id] = future
        
        self.logger.info(f"Submitted quantum task {task.task_id} with coherence {quantum_coherence}")
        return task.task_id
    
    def _execute_quantum_task(self, task: QuantumTask) -> Any:
        """Execute quantum task with coherence tracking"""
        start_time = time.time()
        worker_id = threading.current_thread().name
        
        try:
            # Update worker stats
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = WorkerStats(
                    worker_id=worker_id,
                    created_at=datetime.utcnow()
                )
            
            stats = self.worker_stats[worker_id]
            stats.current_load += 1.0
            stats.last_heartbeat = datetime.utcnow()
            
            # Apply quantum decoherence during processing
            coherence = self.quantum_states.get(task.task_id, 1.0)
            processing_time = time.time() - start_time
            
            # Decoherence based on processing complexity
            decoherence_rate = 0.1 * processing_time  # 10% per second
            new_coherence = coherence * np.exp(-decoherence_rate)
            self.quantum_states[task.task_id] = new_coherence
            
            # Execute the actual task
            if asyncio.iscoroutinefunction(task.task_func):
                # Handle async function in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        task.task_func(*task.args, **task.kwargs)
                    )
                finally:
                    loop.close()
            else:
                result = task.task_func(*task.args, **task.kwargs)
            
            # Update statistics
            processing_time = time.time() - start_time
            stats.tasks_completed += 1
            stats.total_processing_time += processing_time
            stats.current_load -= 1.0
            stats.quantum_coherence = new_coherence
            
            self.total_tasks_processed += 1
            self.total_processing_time += processing_time
            
            self.logger.debug(f"Completed task {task.task_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            # Handle task failure
            processing_time = time.time() - start_time
            stats = self.worker_stats.get(worker_id)
            if stats:
                stats.tasks_failed += 1
                stats.current_load -= 1.0
                stats.total_processing_time += processing_time
            
            # Apply decoherence penalty for failures
            if task.task_id in self.quantum_states:
                self.quantum_states[task.task_id] *= 0.5  # 50% coherence loss
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                return self._execute_quantum_task(task)
            
            raise
    
    async def _execute_quantum_task_async(self, task: QuantumTask) -> Any:
        """Async version of quantum task execution"""
        start_time = time.time()
        
        try:
            # Apply quantum decoherence
            coherence = self.quantum_states.get(task.task_id, 1.0)
            
            # Execute task
            if asyncio.iscoroutinefunction(task.task_func):
                result = await task.task_func(*task.args, **task.kwargs)
            else:
                result = task.task_func(*task.args, **task.kwargs)
            
            # Update quantum state
            processing_time = time.time() - start_time
            decoherence_rate = 0.1 * processing_time
            new_coherence = coherence * np.exp(-decoherence_rate)
            self.quantum_states[task.task_id] = new_coherence
            
            return result
            
        except Exception as e:
            # Handle failure with coherence penalty
            if task.task_id in self.quantum_states:
                self.quantum_states[task.task_id] *= 0.5
            
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await asyncio.sleep(0.1 * task.retry_count)  # Exponential backoff
                return await self._execute_quantum_task_async(task)
            
            raise
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get result of submitted task"""
        if task_id not in self.result_futures:
            raise QuantumTaskPlannerError(f"Task {task_id} not found")
        
        future = self.result_futures[task_id]
        
        try:
            if self.worker_type == "async":
                # Handle async future
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(asyncio.wait_for(future, timeout))
            else:
                return future.result(timeout=timeout)
        finally:
            # Cleanup
            del self.result_futures[task_id]
            self.quantum_states.pop(task_id, None)
    
    async def get_task_result_async(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Async version of get_task_result"""
        if task_id not in self.result_futures:
            raise QuantumTaskPlannerError(f"Task {task_id} not found")
        
        future = self.result_futures[task_id]
        
        try:
            if timeout:
                return await asyncio.wait_for(future, timeout)
            else:
                return await future
        finally:
            del self.result_futures[task_id]
            self.quantum_states.pop(task_id, None)
    
    def _worker_monitor_loop(self):
        """Monitor worker health and performance"""
        while not self.shutdown_event.wait(30):  # Check every 30 seconds
            try:
                current_time = datetime.utcnow()
                
                # Check worker health
                unhealthy_workers = []
                for worker_id, stats in self.worker_stats.items():
                    time_since_heartbeat = (current_time - stats.last_heartbeat).total_seconds()
                    
                    # Mark workers as unhealthy if no heartbeat in 2 minutes
                    if time_since_heartbeat > 120:
                        unhealthy_workers.append(worker_id)
                    
                    # Update quantum coherence decay
                    decay_factor = np.exp(-time_since_heartbeat / 3600)  # 1 hour half-life
                    stats.quantum_coherence *= decay_factor
                
                # Remove unhealthy workers
                for worker_id in unhealthy_workers:
                    del self.worker_stats[worker_id]
                    self.logger.warning(f"Removed unhealthy worker {worker_id}")
                
                # Log pool statistics
                if len(self.worker_stats) > 0:
                    avg_load = np.mean([s.current_load for s in self.worker_stats.values()])
                    avg_coherence = np.mean([s.quantum_coherence for s in self.worker_stats.values()])
                    
                    self.logger.debug(
                        f"Pool stats: {len(self.worker_stats)} workers, "
                        f"avg load: {avg_load:.2f}, avg coherence: {avg_coherence:.3f}"
                    )
                
            except Exception as e:
                self.logger.error(f"Worker monitor error: {e}")
    
    def _adaptive_scaling_loop(self):
        """Adaptive scaling based on load and quantum states"""
        while not self.shutdown_event.wait(60):  # Check every minute
            if not self.adaptive_scaling_enabled:
                continue
            
            try:
                # Calculate current load metrics
                if not self.worker_stats:
                    continue
                
                current_load = np.mean([s.current_load for s in self.worker_stats.values()])
                avg_coherence = np.mean([s.quantum_coherence for s in self.worker_stats.values()])
                queue_size = len(self.result_futures)
                
                # Scaling decision logic
                scale_up = (
                    current_load > self.scaling_factor and
                    len(self.worker_stats) < self.max_workers and
                    avg_coherence > self.coherence_threshold
                )
                
                scale_down = (
                    current_load < self.scaling_factor * 0.3 and
                    len(self.worker_stats) > self.min_workers and
                    queue_size < len(self.worker_stats)
                )
                
                if scale_up:
                    self._scale_up()
                elif scale_down:
                    self._scale_down()
                
            except Exception as e:
                self.logger.error(f"Adaptive scaling error: {e}")
    
    def _coherence_management_loop(self):
        """Manage quantum coherence across the system"""
        while not self.shutdown_event.wait(10):  # Check every 10 seconds
            try:
                current_time = time.time()
                
                # Update quantum state coherence
                for task_id, coherence in list(self.quantum_states.items()):
                    # Natural decoherence over time
                    age = current_time - time.time()  # Simplified age calculation
                    decay_factor = np.exp(-abs(age) / 3600)  # 1 hour half-life
                    new_coherence = coherence * decay_factor
                    
                    if new_coherence < 0.01:  # Remove very low coherence states
                        del self.quantum_states[task_id]
                    else:
                        self.quantum_states[task_id] = new_coherence
                
                # Handle entanglement effects
                self._apply_entanglement_effects()
                
            except Exception as e:
                self.logger.error(f"Coherence management error: {e}")
    
    def _apply_entanglement_effects(self):
        """Apply quantum entanglement effects between tasks"""
        for task_id, entangled_ids in self.entanglement_graph.items():
            if task_id not in self.quantum_states:
                continue
            
            # Calculate entanglement correlation
            task_coherence = self.quantum_states[task_id]
            entangled_coherences = [
                self.quantum_states.get(eid, 0) for eid in entangled_ids
            ]
            
            if entangled_coherences:
                # Entangled tasks influence each other's coherence
                avg_entangled_coherence = np.mean(entangled_coherences)
                correlation_strength = 0.1  # 10% correlation
                
                new_coherence = (
                    task_coherence * (1 - correlation_strength) +
                    avg_entangled_coherence * correlation_strength
                )
                
                self.quantum_states[task_id] = new_coherence
    
    def _scale_up(self):
        """Scale up the worker pool"""
        if hasattr(self.executor, '_max_workers'):
            # For ThreadPoolExecutor and ProcessPoolExecutor
            current_workers = getattr(self.executor, '_threads', set())
            if len(current_workers) < self.max_workers:
                self.logger.info("Scaling up worker pool")
                # Note: Actual scaling implementation depends on executor type
    
    def _scale_down(self):
        """Scale down the worker pool"""
        self.logger.info("Scaling down worker pool")
        # Implementation depends on executor type and current load
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        if not self.worker_stats:
            return {
                "active_workers": 0,
                "total_tasks_processed": self.total_tasks_processed,
                "average_processing_time": 0,
                "pool_utilization": 0,
                "quantum_coherence_avg": 0
            }
        
        active_workers = len(self.worker_stats)
        total_completed = sum(s.tasks_completed for s in self.worker_stats.values())
        total_failed = sum(s.tasks_failed for s in self.worker_stats.values())
        avg_load = np.mean([s.current_load for s in self.worker_stats.values()])
        avg_coherence = np.mean([s.quantum_coherence for s in self.worker_stats.values()])
        
        return {
            "active_workers": active_workers,
            "max_workers": self.max_workers,
            "total_tasks_processed": self.total_tasks_processed,
            "tasks_completed": total_completed,
            "tasks_failed": total_failed,
            "success_rate": total_completed / (total_completed + total_failed) if (total_completed + total_failed) > 0 else 1.0,
            "average_processing_time": self.total_processing_time / self.total_tasks_processed if self.total_tasks_processed > 0 else 0,
            "pool_utilization": avg_load,
            "quantum_coherence_avg": avg_coherence,
            "pending_tasks": len(self.result_futures),
            "quantum_states_tracked": len(self.quantum_states),
            "entanglement_pairs": sum(len(v) for v in self.entanglement_graph.values()) // 2
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool"""
        self.is_running = False
        self.shutdown_event.set()
        
        if self.executor:
            self.executor.shutdown(wait=wait)
        
        self.logger.info("Quantum worker pool shutdown complete")


# Global worker pool instances
_worker_pools: Dict[str, QuantumWorkerPool] = {}
_pool_lock = threading.RLock()


def get_worker_pool(name: str = "default", **kwargs) -> QuantumWorkerPool:
    """Get or create named worker pool"""
    with _pool_lock:
        if name not in _worker_pools:
            _worker_pools[name] = QuantumWorkerPool(**kwargs)
        return _worker_pools[name]


def shutdown_all_pools():
    """Shutdown all worker pools"""
    with _pool_lock:
        for pool in _worker_pools.values():
            pool.shutdown()
        _worker_pools.clear()


# Concurrent processing decorators
def quantum_parallel(pool_name: str = "default",
                     priority: float = 1.0,
                     quantum_coherence: float = 1.0,
                     timeout: Optional[float] = None):
    """
    Decorator to execute function in quantum worker pool
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            pool = get_worker_pool(pool_name)
            task_id = pool.submit_quantum_task(
                func, *args,
                priority=priority,
                quantum_coherence=quantum_coherence,
                timeout=timeout,
                **kwargs
            )
            return pool.get_task_result(task_id, timeout)
        
        return wrapper
    return decorator


def quantum_parallel_async(pool_name: str = "default",
                          priority: float = 1.0,
                          quantum_coherence: float = 1.0,
                          timeout: Optional[float] = None):
    """
    Async decorator for quantum parallel execution
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            pool = get_worker_pool(pool_name, worker_type="async")
            task_id = pool.submit_quantum_task(
                func, *args,
                priority=priority,
                quantum_coherence=quantum_coherence,
                timeout=timeout,
                **kwargs
            )
            return await pool.get_task_result_async(task_id, timeout)
        
        return wrapper
    return decorator


# Utility functions for batch processing
async def process_quantum_batch(tasks: List[Tuple[Callable, Tuple, Dict]],
                               pool_name: str = "default",
                               max_concurrent: int = 10,
                               preserve_order: bool = True) -> List[Any]:
    """
    Process a batch of tasks concurrently with quantum awareness
    """
    pool = get_worker_pool(pool_name, worker_type="async")
    
    # Submit all tasks
    task_ids = []
    for task_func, args, kwargs in tasks:
        task_id = pool.submit_quantum_task(task_func, *args, **kwargs)
        task_ids.append(task_id)
    
    # Collect results
    if preserve_order:
        results = []
        for task_id in task_ids:
            result = await pool.get_task_result_async(task_id)
            results.append(result)
        return results
    else:
        # Return results as they complete
        results = await asyncio.gather(*[
            pool.get_task_result_async(task_id) for task_id in task_ids
        ], return_exceptions=True)
        return results