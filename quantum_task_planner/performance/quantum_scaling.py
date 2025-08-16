"""
Quantum-Enhanced Auto-Scaling and Performance Optimization for Generation 3
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import psutil
import queue
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """System resource metrics for scaling decisions"""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_resource_pressure(self) -> float:
        """Calculate overall resource pressure (0.0 to 1.0)"""
        weights = {"cpu": 0.4, "memory": 0.3, "disk": 0.2, "network": 0.1}
        pressure = (
            self.cpu_percent * weights["cpu"] +
            self.memory_percent * weights["memory"] +
            self.disk_io_percent * weights["disk"] +
            self.network_io_percent * weights["network"]
        ) / 100.0
        return min(1.0, pressure)


@dataclass
class ScalingEvent:
    """Scaling event record"""
    timestamp: datetime
    event_type: str  # "scale_up", "scale_down", "optimize"
    resource_pressure: float
    worker_count_before: int
    worker_count_after: int
    reason: str


class QuantumResourceMonitor:
    """Monitor system resources with quantum-inspired adaptive thresholds"""
    
    def __init__(self):
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history = 1000
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Adaptive thresholds
        self.cpu_threshold = 70.0
        self.memory_threshold = 80.0
        self.adaptation_rate = 0.1
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history size manageable
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                # Adapt thresholds based on historical data
                self._adapt_thresholds()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()
            
            # Calculate IO percentages (simplified)
            disk_io_percent = min(100.0, (disk_io.read_bytes + disk_io.write_bytes) / (1024**3) * 10)
            network_io_percent = min(100.0, (network_io.bytes_sent + network_io.bytes_recv) / (1024**3) * 5)
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_io_percent=disk_io_percent,
                network_io_percent=network_io_percent
            )
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return ResourceMetrics(0, 0, 0, 0)
    
    def _adapt_thresholds(self):
        """Adapt scaling thresholds based on historical performance"""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = self.metrics_history[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        # Quantum-inspired adaptation: adjust thresholds based on stability
        cpu_variance = sum((m.cpu_percent - avg_cpu) ** 2 for m in recent_metrics) / len(recent_metrics)
        memory_variance = sum((m.memory_percent - avg_memory) ** 2 for m in recent_metrics) / len(recent_metrics)
        
        # Lower variance = more stable = can handle higher thresholds
        if cpu_variance < 50:  # Low variance
            self.cpu_threshold = min(85.0, self.cpu_threshold + self.adaptation_rate)
        else:  # High variance
            self.cpu_threshold = max(60.0, self.cpu_threshold - self.adaptation_rate)
        
        if memory_variance < 50:
            self.memory_threshold = min(90.0, self.memory_threshold + self.adaptation_rate)
        else:
            self.memory_threshold = max(70.0, self.memory_threshold - self.adaptation_rate)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def should_scale_up(self) -> bool:
        """Check if system should scale up"""
        metrics = self.get_current_metrics()
        if not metrics:
            return False
        
        return (metrics.cpu_percent > self.cpu_threshold or 
                metrics.memory_percent > self.memory_threshold)
    
    def should_scale_down(self) -> bool:
        """Check if system can scale down"""
        if len(self.metrics_history) < 5:
            return False
        
        recent_metrics = self.metrics_history[-5:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        
        return (avg_cpu < self.cpu_threshold * 0.5 and 
                avg_memory < self.memory_threshold * 0.5)


class QuantumWorkerPool:
    """Quantum-enhanced worker pool with auto-scaling"""
    
    def __init__(self, 
                 min_workers: int = 2,
                 max_workers: int = None,
                 scale_factor: float = 1.5):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.scale_factor = scale_factor
        
        self.current_workers = min_workers
        self.thread_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ProcessPoolExecutor] = None
        
        self.resource_monitor = QuantumResourceMonitor()
        self.scaling_history: List[ScalingEvent] = []
        self.task_queue: queue.Queue = queue.Queue()
        
        self._initialize_executors()
    
    def _initialize_executors(self):
        """Initialize thread and process executors"""
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="quantum_worker"
        )
        
        self.process_executor = ProcessPoolExecutor(
            max_workers=min(self.current_workers, multiprocessing.cpu_count())
        )
    
    def start_auto_scaling(self):
        """Start auto-scaling monitoring"""
        self.resource_monitor.start_monitoring()
        
        # Start scaling decision loop
        scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        scaling_thread.start()
        
        logger.info(f"Auto-scaling started with {self.current_workers} workers")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling"""
        self.resource_monitor.stop_monitoring()
        logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main scaling decision loop"""
        while self.resource_monitor.monitoring:
            try:
                time.sleep(5.0)  # Check every 5 seconds
                
                if self.resource_monitor.should_scale_up():
                    self._scale_up()
                elif self.resource_monitor.should_scale_down():
                    self._scale_down()
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
    
    def _scale_up(self):
        """Scale up worker pool"""
        if self.current_workers >= self.max_workers:
            return
        
        old_workers = self.current_workers
        new_workers = min(
            self.max_workers,
            int(self.current_workers * self.scale_factor)
        )
        
        if new_workers == old_workers:
            return
        
        try:
            # Create new executors with more workers
            self.thread_executor.shutdown(wait=False)
            self.process_executor.shutdown(wait=False)
            
            self.current_workers = new_workers
            self._initialize_executors()
            
            # Record scaling event
            metrics = self.resource_monitor.get_current_metrics()
            event = ScalingEvent(
                timestamp=datetime.now(),
                event_type="scale_up",
                resource_pressure=metrics.get_resource_pressure() if metrics else 0.0,
                worker_count_before=old_workers,
                worker_count_after=new_workers,
                reason="High resource utilization"
            )
            self.scaling_history.append(event)
            
            logger.info(f"Scaled up from {old_workers} to {new_workers} workers")
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
    
    def _scale_down(self):
        """Scale down worker pool"""
        if self.current_workers <= self.min_workers:
            return
        
        old_workers = self.current_workers
        new_workers = max(
            self.min_workers,
            int(self.current_workers / self.scale_factor)
        )
        
        if new_workers == old_workers:
            return
        
        try:
            # Create new executors with fewer workers
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            self.current_workers = new_workers
            self._initialize_executors()
            
            # Record scaling event
            metrics = self.resource_monitor.get_current_metrics()
            event = ScalingEvent(
                timestamp=datetime.now(),
                event_type="scale_down",
                resource_pressure=metrics.get_resource_pressure() if metrics else 0.0,
                worker_count_before=old_workers,
                worker_count_after=new_workers,
                reason="Low resource utilization"
            )
            self.scaling_history.append(event)
            
            logger.info(f"Scaled down from {old_workers} to {new_workers} workers")
            
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
    
    def submit_task(self, func: Callable, *args, use_process: bool = False, **kwargs):
        """Submit task to appropriate executor"""
        if use_process:
            return self.process_executor.submit(func, *args, **kwargs)
        else:
            return self.thread_executor.submit(func, *args, **kwargs)
    
    async def submit_async_task(self, coro):
        """Submit async task"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_executor, lambda: asyncio.run(coro))
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics"""
        metrics = self.resource_monitor.get_current_metrics()
        
        return {
            "current_workers": self.current_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scaling_events": len(self.scaling_history),
            "recent_events": self.scaling_history[-5:] if self.scaling_history else [],
            "current_resource_pressure": metrics.get_resource_pressure() if metrics else 0.0,
            "cpu_threshold": self.resource_monitor.cpu_threshold,
            "memory_threshold": self.resource_monitor.memory_threshold,
            "metrics_history_size": len(self.resource_monitor.metrics_history)
        }
    
    def shutdown(self):
        """Shutdown worker pool"""
        self.stop_auto_scaling()
        
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
        
        logger.info("Worker pool shutdown complete")


class QuantumLoadBalancer:
    """Quantum-inspired load balancer for distributed task processing"""
    
    def __init__(self, worker_pools: List[QuantumWorkerPool]):
        self.worker_pools = worker_pools
        self.pool_weights = [1.0] * len(worker_pools)  # Initial equal weights
        self.task_history: Dict[int, List[float]] = {}  # pool_id -> [response_times]
        self.quantum_coherence = 1.0
    
    def select_pool(self) -> QuantumWorkerPool:
        """Select optimal worker pool using quantum-inspired algorithm"""
        if not self.worker_pools:
            raise ValueError("No worker pools available")
        
        # Calculate pool scores based on multiple factors
        pool_scores = []
        
        for i, pool in enumerate(self.worker_pools):
            stats = pool.get_scaling_stats()
            
            # Factors: current load, resource pressure, historical performance
            load_factor = 1.0 - (stats["current_workers"] / stats["max_workers"])
            pressure_factor = 1.0 - stats["current_resource_pressure"]
            
            # Historical performance factor
            if i in self.task_history and self.task_history[i]:
                avg_response_time = sum(self.task_history[i]) / len(self.task_history[i])
                performance_factor = 1.0 / (1.0 + avg_response_time)  # Lower time = higher score
            else:
                performance_factor = 0.5  # Default for new pools
            
            # Quantum weight factor
            weight_factor = self.pool_weights[i] / max(self.pool_weights)
            
            # Combined score with quantum coherence influence
            score = (
                load_factor * 0.3 +
                pressure_factor * 0.3 +
                performance_factor * 0.2 +
                weight_factor * 0.2
            ) * self.quantum_coherence
            
            pool_scores.append((score, i, pool))
        
        # Select pool with highest score
        pool_scores.sort(reverse=True)
        return pool_scores[0][2]
    
    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to optimal pool"""
        start_time = time.time()
        pool = self.select_pool()
        pool_index = self.worker_pools.index(pool)
        
        # Submit task
        future = pool.submit_task(func, *args, **kwargs)
        
        # Record performance for learning
        def record_performance(fut):
            try:
                response_time = time.time() - start_time
                if pool_index not in self.task_history:
                    self.task_history[pool_index] = []
                
                self.task_history[pool_index].append(response_time)
                
                # Keep only recent history
                if len(self.task_history[pool_index]) > 100:
                    self.task_history[pool_index] = self.task_history[pool_index][-100:]
                
                # Update pool weights based on performance
                self._update_pool_weights()
                
            except Exception as e:
                logger.error(f"Performance recording failed: {e}")
        
        future.add_done_callback(record_performance)
        return future
    
    def _update_pool_weights(self):
        """Update pool weights based on performance history"""
        for i, pool in enumerate(self.worker_pools):
            if i not in self.task_history or not self.task_history[i]:
                continue
            
            # Calculate average response time
            avg_response_time = sum(self.task_history[i]) / len(self.task_history[i])
            
            # Update weight: lower response time = higher weight
            self.pool_weights[i] = 1.0 / (1.0 + avg_response_time)
        
        # Normalize weights
        total_weight = sum(self.pool_weights)
        if total_weight > 0:
            self.pool_weights = [w / total_weight for w in self.pool_weights]
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "pool_count": len(self.worker_pools),
            "pool_weights": self.pool_weights,
            "quantum_coherence": self.quantum_coherence,
            "task_history_sizes": {i: len(history) for i, history in self.task_history.items()},
            "average_response_times": {
                i: sum(history) / len(history) if history else 0
                for i, history in self.task_history.items()
            }
        }


# Global scaling infrastructure
quantum_worker_pool = QuantumWorkerPool()


def enable_quantum_scaling():
    """Enable quantum auto-scaling globally"""
    quantum_worker_pool.start_auto_scaling()
    logger.info("Quantum scaling enabled globally")


def disable_quantum_scaling():
    """Disable quantum auto-scaling globally"""
    quantum_worker_pool.stop_auto_scaling()
    logger.info("Quantum scaling disabled globally")


def quantum_parallel(use_process: bool = False):
    """Decorator for parallel execution with quantum scaling"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return quantum_worker_pool.submit_task(func, *args, use_process=use_process, **kwargs)
        return wrapper
    return decorator