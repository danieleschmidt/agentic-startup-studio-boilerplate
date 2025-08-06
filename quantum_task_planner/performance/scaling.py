"""
Quantum Task Planner Auto-Scaling and Load Balancing

Advanced scaling system with quantum-aware load balancing,
predictive scaling, and distributed quantum state management.
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import psutil

from ..utils.logging import get_logger, QuantumMetric, PerformanceMetric
from ..utils.exceptions import QuantumTaskPlannerError


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    queue_length: int
    response_time_p95: float
    quantum_coherence_avg: float
    throughput: float
    error_rate: float


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""
    name: str
    metric: str  # 'cpu', 'memory', 'queue_length', 'response_time', etc.
    threshold_up: float
    threshold_down: float
    scale_up_amount: int = 1
    scale_down_amount: int = 1
    cooldown_seconds: int = 300
    min_instances: int = 1
    max_instances: int = 10
    enabled: bool = True
    last_scaling_action: Optional[datetime] = None


class QuantumLoadBalancer:
    """Quantum-aware load balancer with coherence-based routing"""
    
    def __init__(self, coherence_weight: float = 0.3):
        self.coherence_weight = coherence_weight
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self.routing_history: deque = deque(maxlen=1000)
        self.logger = get_logger()
    
    def register_worker(self, worker_id: str, endpoint: str, quantum_coherence: float = 1.0):
        """Register a worker instance"""
        self.worker_stats[worker_id] = {
            "endpoint": endpoint,
            "quantum_coherence": quantum_coherence,
            "active_tasks": 0,
            "total_requests": 0,
            "success_rate": 1.0,
            "avg_response_time": 0.0,
            "last_health_check": datetime.utcnow(),
            "healthy": True,
            "load_factor": 0.0
        }
        self.logger.info(f"Registered worker {worker_id} at {endpoint}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker instance"""
        if worker_id in self.worker_stats:
            del self.worker_stats[worker_id]
            self.logger.info(f"Unregistered worker {worker_id}")
    
    def update_worker_stats(self, worker_id: str, stats: Dict[str, Any]):
        """Update worker statistics"""
        if worker_id in self.worker_stats:
            self.worker_stats[worker_id].update(stats)
            self.worker_stats[worker_id]["last_health_check"] = datetime.utcnow()
    
    def select_worker(self, task_quantum_coherence: float = 1.0) -> Optional[str]:
        """Select optimal worker based on quantum-aware load balancing"""
        available_workers = [
            (worker_id, stats) for worker_id, stats in self.worker_stats.items()
            if stats["healthy"] and self._is_worker_available(worker_id)
        ]
        
        if not available_workers:
            return None
        
        # Calculate selection probabilities
        worker_scores = []
        for worker_id, stats in available_workers:
            score = self._calculate_worker_score(stats, task_quantum_coherence)
            worker_scores.append((worker_id, score))
        
        # Quantum-weighted selection
        if worker_scores:
            total_score = sum(score for _, score in worker_scores)
            if total_score > 0:
                # Normalize scores to probabilities
                probabilities = [score / total_score for _, score in worker_scores]
                worker_ids = [worker_id for worker_id, _ in worker_scores]
                
                # Select using quantum probability
                selected_worker = np.random.choice(worker_ids, p=probabilities)
                
                # Record routing decision
                self.routing_history.append({
                    "timestamp": datetime.utcnow(),
                    "selected_worker": selected_worker,
                    "task_coherence": task_quantum_coherence,
                    "worker_coherence": self.worker_stats[selected_worker]["quantum_coherence"],
                    "selection_score": dict(worker_scores)[selected_worker]
                })
                
                return selected_worker
        
        # Fallback to first available worker
        return available_workers[0][0]
    
    def _is_worker_available(self, worker_id: str) -> bool:
        """Check if worker is available for new tasks"""
        stats = self.worker_stats[worker_id]
        
        # Check health check recency
        time_since_health = (datetime.utcnow() - stats["last_health_check"]).total_seconds()
        if time_since_health > 60:  # 1 minute timeout
            stats["healthy"] = False
            return False
        
        # Check if worker is overloaded
        if stats["active_tasks"] > 50:  # Max concurrent tasks
            return False
        
        return True
    
    def _calculate_worker_score(self, stats: Dict[str, Any], task_coherence: float) -> float:
        """Calculate worker selection score"""
        # Base score components
        load_score = 1.0 / (1.0 + stats["load_factor"])
        performance_score = stats["success_rate"] * (1.0 / max(0.1, stats["avg_response_time"]))
        
        # Quantum coherence matching bonus
        coherence_diff = abs(stats["quantum_coherence"] - task_coherence)
        coherence_score = np.exp(-coherence_diff)  # Higher score for similar coherence
        
        # Combined score
        total_score = (
            load_score * 0.4 +
            performance_score * 0.4 +
            coherence_score * self.coherence_weight
        )
        
        return max(0.0, total_score)
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across workers"""
        if not self.worker_stats:
            return {"workers": 0, "distribution": {}}
        
        total_tasks = sum(stats["active_tasks"] for stats in self.worker_stats.values())
        
        distribution = {}
        for worker_id, stats in self.worker_stats.items():
            load_percentage = (stats["active_tasks"] / max(1, total_tasks)) * 100
            distribution[worker_id] = {
                "active_tasks": stats["active_tasks"],
                "load_percentage": load_percentage,
                "quantum_coherence": stats["quantum_coherence"],
                "healthy": stats["healthy"]
            }
        
        return {
            "workers": len(self.worker_stats),
            "total_active_tasks": total_tasks,
            "distribution": distribution
        }


class AutoScaler:
    """Intelligent auto-scaling system with predictive capabilities"""
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 metrics_window: int = 300,  # 5 minutes
                 prediction_horizon: int = 60):  # 1 minute ahead
        
        self.check_interval = check_interval
        self.metrics_window = metrics_window
        self.prediction_horizon = prediction_horizon
        
        # Scaling configuration
        self.scaling_rules: List[ScalingRule] = []
        self.metrics_history: deque = deque(maxlen=metrics_window)
        self.scaling_actions: List[Dict[str, Any]] = []
        
        # State management
        self.current_instances = 1
        self.target_instances = 1
        self.scaling_in_progress = False
        
        # Prediction model (simple linear regression for now)
        self.prediction_weights: Dict[str, float] = {
            "cpu_trend": 0.3,
            "memory_trend": 0.2,
            "queue_trend": 0.3,
            "time_of_day": 0.1,
            "day_of_week": 0.1
        }
        
        self.logger = get_logger()
        self._setup_default_rules()
        
        # Start scaling monitor
        self.monitor_thread = None
        self.shutdown_event = threading.Event()
        self.start_monitoring()
    
    def _setup_default_rules(self):
        """Setup default auto-scaling rules"""
        self.scaling_rules = [
            ScalingRule(
                name="cpu_scaling",
                metric="cpu_usage",
                threshold_up=70.0,
                threshold_down=30.0,
                scale_up_amount=2,
                scale_down_amount=1,
                cooldown_seconds=300
            ),
            ScalingRule(
                name="memory_scaling",
                metric="memory_usage", 
                threshold_up=80.0,
                threshold_down=40.0,
                scale_up_amount=1,
                scale_down_amount=1,
                cooldown_seconds=240
            ),
            ScalingRule(
                name="queue_scaling",
                metric="queue_length",
                threshold_up=50.0,
                threshold_down=10.0,
                scale_up_amount=3,
                scale_down_amount=1,
                cooldown_seconds=180
            ),
            ScalingRule(
                name="response_time_scaling",
                metric="response_time_p95",
                threshold_up=1000.0,  # 1 second
                threshold_down=200.0,  # 200ms
                scale_up_amount=2,
                scale_down_amount=1,
                cooldown_seconds=240
            )
        ]
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule"""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="autoscaler"
        )
        self.monitor_thread.start()
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.shutdown_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_event.wait(self.check_interval):
            try:
                # Collect current metrics
                current_metrics = self._collect_metrics()
                self.metrics_history.append(current_metrics)
                
                # Make scaling decision
                scaling_decision = self._evaluate_scaling_need(current_metrics)
                
                if scaling_decision:
                    asyncio.run(self._execute_scaling_action(scaling_decision))
                
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Application metrics (mock values - would be real in production)
        active_tasks = self._get_active_task_count()
        queue_length = self._get_queue_length()
        response_time_p95 = self._get_response_time_p95()
        quantum_coherence_avg = self._get_average_quantum_coherence()
        throughput = self._get_current_throughput()
        error_rate = self._get_error_rate()
        
        return ScalingMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_tasks=active_tasks,
            queue_length=queue_length,
            response_time_p95=response_time_p95,
            quantum_coherence_avg=quantum_coherence_avg,
            throughput=throughput,
            error_rate=error_rate
        )
    
    def _get_active_task_count(self) -> int:
        """Get current active task count"""
        # Mock implementation - would interface with actual task manager
        return np.random.randint(10, 100)
    
    def _get_queue_length(self) -> int:
        """Get current queue length"""
        # Mock implementation
        return np.random.randint(0, 50)
    
    def _get_response_time_p95(self) -> float:
        """Get 95th percentile response time in ms"""
        # Mock implementation
        return np.random.uniform(100, 2000)
    
    def _get_average_quantum_coherence(self) -> float:
        """Get average quantum coherence"""
        # Mock implementation
        return np.random.uniform(0.3, 1.0)
    
    def _get_current_throughput(self) -> float:
        """Get current throughput (requests/second)"""
        # Mock implementation
        return np.random.uniform(10, 200)
    
    def _get_error_rate(self) -> float:
        """Get current error rate"""
        # Mock implementation
        return np.random.uniform(0.0, 0.1)
    
    def _evaluate_scaling_need(self, current_metrics: ScalingMetrics) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling is needed based on rules and predictions"""
        if self.scaling_in_progress:
            return None
        
        # Check each scaling rule
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue
            
            # Check cooldown period
            if (rule.last_scaling_action and 
                (datetime.utcnow() - rule.last_scaling_action).total_seconds() < rule.cooldown_seconds):
                continue
            
            metric_value = getattr(current_metrics, rule.metric, 0)
            
            # Scale up decision
            if (metric_value > rule.threshold_up and 
                self.current_instances < rule.max_instances):
                
                # Use predictive scaling to determine scale amount
                predicted_load = self._predict_future_load(rule.metric)
                scale_amount = self._calculate_scale_amount(
                    rule, metric_value, predicted_load, "up"
                )
                
                return {
                    "action": "scale_up",
                    "rule": rule.name,
                    "current_value": metric_value,
                    "threshold": rule.threshold_up,
                    "scale_amount": scale_amount,
                    "predicted_load": predicted_load
                }
            
            # Scale down decision
            elif (metric_value < rule.threshold_down and 
                  self.current_instances > rule.min_instances):
                
                predicted_load = self._predict_future_load(rule.metric)
                scale_amount = self._calculate_scale_amount(
                    rule, metric_value, predicted_load, "down"
                )
                
                return {
                    "action": "scale_down",
                    "rule": rule.name,
                    "current_value": metric_value,
                    "threshold": rule.threshold_down,
                    "scale_amount": scale_amount,
                    "predicted_load": predicted_load
                }
        
        return None
    
    def _predict_future_load(self, metric: str) -> float:
        """Predict future load for a specific metric"""
        if len(self.metrics_history) < 10:
            return 0.0  # Not enough data for prediction
        
        # Simple trend analysis
        recent_values = [
            getattr(m, metric, 0) for m in list(self.metrics_history)[-10:]
        ]
        
        if len(recent_values) < 2:
            return recent_values[-1] if recent_values else 0.0
        
        # Calculate trend (simple linear regression)
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            # Predict future value
            future_value = recent_values[-1] + (slope * (self.prediction_horizon / self.check_interval))
            return max(0, future_value)
        
        return recent_values[-1]
    
    def _calculate_scale_amount(self, rule: ScalingRule, current_value: float, 
                              predicted_value: float, direction: str) -> int:
        """Calculate intelligent scale amount based on load and prediction"""
        base_amount = rule.scale_up_amount if direction == "up" else rule.scale_down_amount
        
        # Adjust based on severity
        threshold = rule.threshold_up if direction == "up" else rule.threshold_down
        severity = abs(current_value - threshold) / threshold
        
        # Consider prediction
        if predicted_value > 0:
            prediction_factor = predicted_value / current_value
            if direction == "up" and prediction_factor > 1.2:
                base_amount = int(base_amount * 1.5)  # Scale more aggressively
            elif direction == "down" and prediction_factor < 0.8:
                base_amount = max(1, int(base_amount * 0.5))  # Scale down more cautiously
        
        # Apply severity multiplier
        if severity > 0.5:  # High severity
            base_amount = int(base_amount * 1.5)
        
        return max(1, base_amount)
    
    async def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute the scaling action"""
        self.scaling_in_progress = True
        action = decision["action"]
        scale_amount = decision["scale_amount"]
        
        try:
            if action == "scale_up":
                new_target = min(
                    self.current_instances + scale_amount,
                    max(rule.max_instances for rule in self.scaling_rules)
                )
                await self._scale_up(new_target - self.current_instances)
                
            elif action == "scale_down":
                new_target = max(
                    self.current_instances - scale_amount,
                    max(rule.min_instances for rule in self.scaling_rules)
                )
                await self._scale_down(self.current_instances - new_target)
            
            # Record scaling action
            self.scaling_actions.append({
                "timestamp": datetime.utcnow(),
                "decision": decision,
                "previous_instances": self.current_instances,
                "new_instances": self.target_instances
            })
            
            # Update rule last action time
            for rule in self.scaling_rules:
                if rule.name == decision["rule"]:
                    rule.last_scaling_action = datetime.utcnow()
                    break
            
            self.logger.info(f"Executed {action} from {self.current_instances} to {self.target_instances} instances")
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")
        
        finally:
            self.scaling_in_progress = False
    
    async def _scale_up(self, instances_to_add: int):
        """Scale up by adding instances"""
        # Mock implementation - would integrate with orchestration system
        self.target_instances = self.current_instances + instances_to_add
        
        # Simulate gradual scaling
        for i in range(instances_to_add):
            await asyncio.sleep(1)  # Simulate startup time
            self.current_instances += 1
            self.logger.info(f"Started new instance {self.current_instances}")
    
    async def _scale_down(self, instances_to_remove: int):
        """Scale down by removing instances"""
        # Mock implementation - would integrate with orchestration system
        self.target_instances = self.current_instances - instances_to_remove
        
        # Simulate gradual scaling
        for i in range(instances_to_remove):
            await asyncio.sleep(0.5)  # Simulate shutdown time
            self.current_instances -= 1
            self.logger.info(f"Stopped instance, now {self.current_instances}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        recent_actions = self.scaling_actions[-10:] if self.scaling_actions else []
        
        return {
            "current_instances": self.current_instances,
            "target_instances": self.target_instances,
            "scaling_in_progress": self.scaling_in_progress,
            "rules_enabled": len([r for r in self.scaling_rules if r.enabled]),
            "recent_actions": [
                {
                    "timestamp": action["timestamp"].isoformat(),
                    "action": action["decision"]["action"],
                    "rule": action["decision"]["rule"],
                    "instances_change": action["new_instances"] - action["previous_instances"]
                }
                for action in recent_actions
            ],
            "metrics_window_size": len(self.metrics_history)
        }


class ResourcePool:
    """Dynamic resource pool with quantum-aware allocation"""
    
    def __init__(self, total_cpu: float, total_memory: float, total_storage: float):
        self.total_cpu = total_cpu
        self.total_memory = total_memory
        self.total_storage = total_storage
        
        self.allocated_cpu = 0.0
        self.allocated_memory = 0.0
        self.allocated_storage = 0.0
        
        self.reservations: Dict[str, Dict[str, float]] = {}
        self.quantum_priority_multiplier = 1.5
        
        self.logger = get_logger()
    
    def reserve_resources(self, 
                         reservation_id: str,
                         cpu: float,
                         memory: float,
                         storage: float,
                         quantum_coherence: float = 1.0) -> bool:
        """Reserve resources with quantum priority"""
        
        # Apply quantum priority
        if quantum_coherence > 0.8:
            cpu *= self.quantum_priority_multiplier
            memory *= self.quantum_priority_multiplier
        
        # Check availability
        if (self.allocated_cpu + cpu <= self.total_cpu and
            self.allocated_memory + memory <= self.total_memory and
            self.allocated_storage + storage <= self.total_storage):
            
            # Make reservation
            self.reservations[reservation_id] = {
                "cpu": cpu,
                "memory": memory,
                "storage": storage,
                "quantum_coherence": quantum_coherence,
                "timestamp": datetime.utcnow()
            }
            
            self.allocated_cpu += cpu
            self.allocated_memory += memory
            self.allocated_storage += storage
            
            self.logger.info(f"Reserved resources for {reservation_id}: CPU {cpu}, Memory {memory}, Storage {storage}")
            return True
        
        return False
    
    def release_resources(self, reservation_id: str) -> bool:
        """Release reserved resources"""
        if reservation_id in self.reservations:
            reservation = self.reservations[reservation_id]
            
            self.allocated_cpu -= reservation["cpu"]
            self.allocated_memory -= reservation["memory"] 
            self.allocated_storage -= reservation["storage"]
            
            del self.reservations[reservation_id]
            
            self.logger.info(f"Released resources for {reservation_id}")
            return True
        
        return False
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        return {
            "cpu_utilization": self.allocated_cpu / self.total_cpu,
            "memory_utilization": self.allocated_memory / self.total_memory,
            "storage_utilization": self.allocated_storage / self.total_storage,
            "total_reservations": len(self.reservations)
        }


# Global instances
_load_balancer = None
_auto_scaler = None
_resource_pool = None


def get_load_balancer(**kwargs) -> QuantumLoadBalancer:
    """Get global load balancer instance"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = QuantumLoadBalancer(**kwargs)
    return _load_balancer


def get_auto_scaler(**kwargs) -> AutoScaler:
    """Get global auto-scaler instance"""
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler(**kwargs)
    return _auto_scaler


def get_resource_pool(cpu: float = 100.0, memory: float = 32.0, storage: float = 1000.0) -> ResourcePool:
    """Get global resource pool instance"""
    global _resource_pool
    if _resource_pool is None:
        _resource_pool = ResourcePool(cpu, memory, storage)
    return _resource_pool


def shutdown_scaling_systems():
    """Shutdown all scaling systems"""
    global _auto_scaler
    if _auto_scaler:
        _auto_scaler.stop_monitoring()