"""
Quantum Performance Optimizer

Advanced performance optimization system with quantum-inspired algorithms,
adaptive resource allocation, and intelligent caching strategies.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging
from datetime import datetime, timedelta

from ..core.quantum_task import QuantumTask, TaskState
# from ..utils.logging import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking for quantum optimization"""
    task_throughput: float = 0.0
    average_response_time: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    cache_hit_ratio: float = 0.0
    quantum_coherence_avg: float = 0.0
    optimization_efficiency: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_throughput": self.task_throughput,
            "average_response_time": self.average_response_time,
            "resource_utilization": self.resource_utilization,
            "cache_hit_ratio": self.cache_hit_ratio,
            "quantum_coherence_avg": self.quantum_coherence_avg,
            "optimization_efficiency": self.optimization_efficiency,
            "error_rate": self.error_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent
        }


@dataclass
class CacheEntry:
    """Enhanced cache entry with quantum properties"""
    value: Any
    timestamp: datetime
    access_count: int = 0
    quantum_coherence: float = 1.0
    ttl: Optional[timedelta] = None
    
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return datetime.utcnow() - self.timestamp > self.ttl
    
    def get_relevance_score(self) -> float:
        """Calculate relevance score based on access patterns and quantum coherence"""
        age_factor = max(0.1, 1.0 - (datetime.utcnow() - self.timestamp).total_seconds() / 3600)
        access_factor = min(1.0, self.access_count / 10.0)
        return (age_factor * 0.4 + access_factor * 0.3 + self.quantum_coherence * 0.3)


class QuantumPerformanceOptimizer:
    """
    Advanced performance optimizer with quantum-inspired algorithms
    for adaptive resource allocation and intelligent caching.
    """
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 enable_adaptive_scaling: bool = True,
                 enable_quantum_caching: bool = True):
        """
        Initialize quantum performance optimizer
        
        Args:
            max_cache_size: Maximum number of cached entries
            enable_adaptive_scaling: Enable automatic resource scaling
            enable_quantum_caching: Enable quantum-enhanced caching
        """
        self.max_cache_size = max_cache_size
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.enable_quantum_caching = enable_quantum_caching
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Quantum-enhanced cache
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        # Resource pool management
        self.resource_pools: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
        self.active_tasks: Dict[str, QuantumTask] = {}
        self.performance_sampling_queue = deque(maxlen=100)
        
        # Auto-scaling configuration
        self.scaling_thresholds = {
            "cpu_high": 80.0,
            "cpu_low": 20.0,
            "memory_high": 85.0,
            "response_time_high": 2.0,
            "throughput_low": 10.0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Start background optimization tasks
        self._optimization_task = None
        if self.enable_adaptive_scaling:
            self._start_background_optimization()
    
    async def optimize_task_execution(self, 
                                    task: QuantumTask,
                                    execution_function: Callable,
                                    **kwargs) -> Any:
        """
        Execute task with quantum performance optimization
        
        Args:
            task: Quantum task to execute
            execution_function: Function to execute
            **kwargs: Additional arguments
            
        Returns:
            Execution result with performance metrics
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(task, kwargs)
        
        # Check quantum cache
        if self.enable_quantum_caching:
            cached_result = await self._check_quantum_cache(cache_key, task)
            if cached_result is not None:
                self.cache_stats["hits"] += 1
                execution_time = time.time() - start_time
                await self._update_performance_metrics(execution_time, True)
                return cached_result
            
            self.cache_stats["misses"] += 1
        
        # Get optimized resource allocation
        resource_allocation = await self._get_optimal_resource_allocation(task)
        
        try:
            # Execute with resource management
            self.active_tasks[task.task_id] = task
            
            # Apply quantum-enhanced execution optimization
            result = await self._execute_with_quantum_optimization(
                task, execution_function, resource_allocation, **kwargs
            )
            
            # Cache result with quantum properties
            if self.enable_quantum_caching:
                await self._cache_result_with_quantum_properties(cache_key, result, task)
            
            execution_time = time.time() - start_time
            await self._update_performance_metrics(execution_time, False)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            execution_time = time.time() - start_time
            await self._update_performance_metrics(execution_time, False, error=True)
            raise
        finally:
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _check_quantum_cache(self, cache_key: str, task: QuantumTask) -> Optional[Any]:
        """Check quantum-enhanced cache for stored results"""
        if cache_key not in self.cache:
            return None
        
        entry = self.cache[cache_key]
        
        # Check expiration
        if entry.is_expired():
            del self.cache[cache_key]
            self.cache_stats["evictions"] += 1
            return None
        
        # Quantum coherence check - results with low coherence may be stale
        coherence_threshold = 0.3
        if entry.quantum_coherence < coherence_threshold:
            # Probabilistic cache invalidation based on quantum decoherence
            invalidation_probability = 1.0 - entry.quantum_coherence
            if np.random.random() < invalidation_probability:
                del self.cache[cache_key]
                self.cache_stats["evictions"] += 1
                return None
        
        # Update access statistics
        entry.access_count += 1
        entry.quantum_coherence *= 0.98  # Slight decoherence on access
        
        return entry.value
    
    async def _cache_result_with_quantum_properties(self, 
                                                   cache_key: str, 
                                                   result: Any, 
                                                   task: QuantumTask):
        """Cache result with quantum-enhanced properties"""
        # Calculate TTL based on task properties
        base_ttl = timedelta(hours=1)
        if task.complexity_factor > 5.0:
            base_ttl = timedelta(hours=6)  # Complex computations cached longer
        elif task.priority.probability_weight > 0.8:
            base_ttl = timedelta(minutes=30)  # High priority results cached shorter
        
        # Create cache entry with quantum coherence
        entry = CacheEntry(
            value=result,
            timestamp=datetime.utcnow(),
            quantum_coherence=task.quantum_coherence,
            ttl=base_ttl
        )
        
        # Manage cache size with quantum-inspired eviction
        if len(self.cache) >= self.max_cache_size:
            await self._quantum_cache_eviction()
        
        self.cache[cache_key] = entry
    
    async def _quantum_cache_eviction(self):
        """Quantum-inspired cache eviction based on relevance scores"""
        # Calculate relevance scores for all entries
        scored_entries = [
            (key, entry.get_relevance_score()) 
            for key, entry in self.cache.items()
        ]
        
        # Sort by relevance (lowest first for eviction)
        scored_entries.sort(key=lambda x: x[1])
        
        # Evict lowest relevance entries (25% of cache)
        eviction_count = max(1, len(self.cache) // 4)
        for i in range(eviction_count):
            if i < len(scored_entries):
                key_to_evict = scored_entries[i][0]
                del self.cache[key_to_evict]
                self.cache_stats["evictions"] += 1
    
    async def _get_optimal_resource_allocation(self, task: QuantumTask) -> Dict[str, float]:
        """Calculate optimal resource allocation using quantum algorithms"""
        # Base allocation
        allocation = {
            "cpu": 1.0,
            "memory": 100.0,  # MB
            "io": 1.0,
            "network": 1.0
        }
        
        # Quantum-inspired resource optimization
        task_complexity = task.complexity_factor
        priority_weight = task.priority.probability_weight
        quantum_coherence = task.quantum_coherence
        
        # Adaptive scaling based on quantum properties
        scaling_factor = (priority_weight * 0.4 + 
                         quantum_coherence * 0.3 + 
                         min(2.0, task_complexity) / 2.0 * 0.3)
        
        # Apply quantum uncertainty to resource allocation
        uncertainty_factor = 1.0 + np.random.normal(0, 0.1 * (1.0 - quantum_coherence))
        
        for resource_type in allocation:
            allocation[resource_type] *= scaling_factor * uncertainty_factor
            allocation[resource_type] = max(0.1, allocation[resource_type])  # Minimum allocation
        
        return allocation
    
    async def _execute_with_quantum_optimization(self,
                                                task: QuantumTask,
                                                execution_function: Callable,
                                                resource_allocation: Dict[str, float],
                                                **kwargs) -> Any:
        """Execute function with quantum optimization techniques"""
        # Set up resource context
        async with await self._create_resource_context(resource_allocation):
            # Apply quantum superposition to execution path
            if task.quantum_coherence > 0.8:
                # High coherence: try parallel execution paths
                result = await self._parallel_quantum_execution(
                    task, execution_function, **kwargs
                )
            else:
                # Low coherence: standard execution
                if asyncio.iscoroutinefunction(execution_function):
                    result = await execution_function(task, **kwargs)
                else:
                    result = execution_function(task, **kwargs)
        
        return result
    
    async def _parallel_quantum_execution(self,
                                        task: QuantumTask,
                                        execution_function: Callable,
                                        **kwargs) -> Any:
        """Execute function using quantum superposition principles"""
        # Create multiple execution contexts (quantum superposition)
        execution_contexts = []
        
        for i in range(3):  # 3 parallel execution paths
            context_kwargs = kwargs.copy()
            # Add slight variations to explore different execution paths
            context_kwargs["_quantum_context_id"] = i
            context_kwargs["_quantum_variation"] = np.random.normal(0, 0.1)
            
            execution_contexts.append(
                self._execute_single_context(execution_function, task, context_kwargs)
            )
        
        # Execute all contexts concurrently
        try:
            results = await asyncio.gather(*execution_contexts, return_exceptions=True)
            
            # Quantum measurement: select best result
            best_result = None
            best_score = -float('inf')
            
            for result in results:
                if not isinstance(result, Exception):
                    score = self._evaluate_result_quality(result, task)
                    if score > best_score:
                        best_score = score
                        best_result = result
            
            if best_result is not None:
                return best_result
            else:
                # Fallback to first non-exception result
                for result in results:
                    if not isinstance(result, Exception):
                        return result
                
                # All failed, raise first exception
                for result in results:
                    if isinstance(result, Exception):
                        raise result
                        
        except Exception as e:
            # Fallback to single execution
            self.logger.warning(f"Parallel execution failed, falling back: {e}")
            if asyncio.iscoroutinefunction(execution_function):
                return await execution_function(task, **kwargs)
            else:
                return execution_function(task, **kwargs)
    
    async def _execute_single_context(self, 
                                     execution_function: Callable,
                                     task: QuantumTask,
                                     context_kwargs: Dict[str, Any]) -> Any:
        """Execute function in a single quantum context"""
        try:
            if asyncio.iscoroutinefunction(execution_function):
                return await execution_function(task, **context_kwargs)
            else:
                return execution_function(task, **context_kwargs)
        except Exception as e:
            self.logger.debug(f"Context execution failed: {e}")
            raise
    
    def _evaluate_result_quality(self, result: Any, task: QuantumTask) -> float:
        """Evaluate quality of execution result"""
        # Basic quality scoring - can be extended
        base_score = 1.0
        
        # Consider result type and properties
        if result is None:
            return 0.0
        
        if hasattr(result, '__len__'):
            # Favor results with content
            base_score += min(1.0, len(result) / 10.0)
        
        if isinstance(result, dict):
            # Favor dictionaries with more information
            base_score += min(1.0, len(result) / 5.0)
        
        return base_score
    
    async def _create_resource_context(self, allocation: Dict[str, float]):
        """Create resource allocation context manager"""
        class ResourceContext:
            def __init__(self, optimizer, allocation):
                self.optimizer = optimizer
                self.allocation = allocation
                
            async def __aenter__(self):
                # Reserve resources
                for resource_type, amount in self.allocation.items():
                    await self.optimizer._reserve_resource(resource_type, amount)
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # Release resources
                for resource_type, amount in self.allocation.items():
                    await self.optimizer._release_resource(resource_type, amount)
        
        return ResourceContext(self, allocation)
    
    async def _reserve_resource(self, resource_type: str, amount: float):
        """Reserve resource from pool"""
        # Simple resource reservation - can be enhanced with actual limits
        pass
    
    async def _release_resource(self, resource_type: str, amount: float):
        """Release resource back to pool"""
        # Simple resource release
        pass
    
    async def _update_performance_metrics(self, 
                                        execution_time: float, 
                                        cache_hit: bool, 
                                        error: bool = False):
        """Update performance metrics"""
        self.performance_sampling_queue.append({
            "execution_time": execution_time,
            "cache_hit": cache_hit,
            "error": error,
            "timestamp": time.time()
        })
        
        # Calculate current metrics
        recent_samples = list(self.performance_sampling_queue)
        if recent_samples:
            self.metrics.average_response_time = np.mean([s["execution_time"] for s in recent_samples])
            self.metrics.error_rate = np.mean([s["error"] for s in recent_samples])
            
            # Calculate throughput (tasks per second)
            time_window = 60  # 1 minute window
            current_time = time.time()
            recent_tasks = [s for s in recent_samples if current_time - s["timestamp"] <= time_window]
            self.metrics.task_throughput = len(recent_tasks) / time_window
        
        # Calculate cache hit ratio
        total_cache_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_cache_ops > 0:
            self.metrics.cache_hit_ratio = self.cache_stats["hits"] / total_cache_ops
        
        # Calculate quantum coherence average
        if self.active_tasks:
            self.metrics.quantum_coherence_avg = np.mean([
                task.quantum_coherence for task in self.active_tasks.values()
            ])
    
    def _generate_cache_key(self, task: QuantumTask, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for task and parameters"""
        # Create deterministic cache key
        key_components = [
            task.title,
            task.description,
            str(task.priority.name),
            str(task.complexity_factor),
            str(sorted(kwargs.items()))
        ]
        
        # Hash the components for compact key
        import hashlib
        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode(), usedforsecurity=False).hexdigest()[:32]
    
    def _start_background_optimization(self):
        """Start background optimization tasks"""
        async def optimization_loop():
            while True:
                try:
                    await self._perform_adaptive_scaling()
                    await self._optimize_cache_coherence()
                    await asyncio.sleep(30)  # Run every 30 seconds
                except Exception as e:
                    self.logger.error(f"Background optimization error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        self._optimization_task = asyncio.create_task(optimization_loop())
    
    async def _perform_adaptive_scaling(self):
        """Perform adaptive resource scaling based on metrics"""
        # Check if scaling is needed
        needs_scale_up = (
            self.metrics.cpu_usage_percent > self.scaling_thresholds["cpu_high"] or
            self.metrics.average_response_time > self.scaling_thresholds["response_time_high"] or
            self.metrics.memory_usage_mb > self.scaling_thresholds["memory_high"]
        )
        
        needs_scale_down = (
            self.metrics.cpu_usage_percent < self.scaling_thresholds["cpu_low"] and
            self.metrics.task_throughput < self.scaling_thresholds["throughput_low"] and
            len(self.active_tasks) == 0
        )
        
        if needs_scale_up:
            await self._scale_up_resources()
        elif needs_scale_down:
            await self._scale_down_resources()
    
    async def _scale_up_resources(self):
        """Scale up resources when needed"""
        self.logger.info("Scaling up resources based on performance metrics")
        
        # Increase resource pool sizes
        for resource_type in ["cpu", "memory", "io"]:
            current_size = self.resource_pools[resource_type].qsize()
            for _ in range(min(5, current_size + 1)):  # Add up to 5 more resources
                await self.resource_pools[resource_type].put(f"{resource_type}_resource")
        
        # Increase cache size if needed
        if self.metrics.cache_hit_ratio < 0.7:
            self.max_cache_size = min(2000, int(self.max_cache_size * 1.2))
    
    async def _scale_down_resources(self):
        """Scale down resources when underutilized"""
        self.logger.info("Scaling down resources to optimize efficiency")
        
        # Reduce resource pool sizes
        for resource_type in ["cpu", "memory", "io"]:
            pool = self.resource_pools[resource_type]
            # Remove up to half the resources
            remove_count = min(pool.qsize() // 2, 3)
            for _ in range(remove_count):
                try:
                    pool.get_nowait()
                except asyncio.QueueEmpty:
                    break
    
    async def _optimize_cache_coherence(self):
        """Optimize cache coherence and perform maintenance"""
        # Decay quantum coherence over time
        current_time = datetime.utcnow()
        
        for entry in self.cache.values():
            age_hours = (current_time - entry.timestamp).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - age_hours * 0.01)  # 1% decay per hour
            entry.quantum_coherence *= decay_factor
        
        # Remove entries with very low coherence
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.quantum_coherence < 0.1
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
            self.cache_stats["evictions"] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "current_metrics": self.metrics.to_dict(),
            "cache_statistics": {
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "evictions": self.cache_stats["evictions"],
                "current_size": len(self.cache),
                "max_size": self.max_cache_size,
                "hit_ratio": self.metrics.cache_hit_ratio
            },
            "resource_pools": {
                resource_type: pool.qsize()
                for resource_type, pool in self.resource_pools.items()
            },
            "active_tasks": len(self.active_tasks),
            "optimization_status": {
                "adaptive_scaling_enabled": self.enable_adaptive_scaling,
                "quantum_caching_enabled": self.enable_quantum_caching,
                "background_optimization_active": self._optimization_task is not None and not self._optimization_task.done()
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown optimizer"""
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Quantum Performance Optimizer shutdown complete")