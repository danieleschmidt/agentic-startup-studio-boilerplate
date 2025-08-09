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


class QuantumClusterOrchestrator:
    """Advanced quantum cluster orchestration with multi-region support"""
    
    def __init__(self):
        self.regions: Dict[str, Dict[str, Any]] = {}
        self.global_load_balancer = get_load_balancer()
        self.cluster_topology = {}
        self.quantum_entanglement_links: Dict[str, List[str]] = {}
        
        # Orchestration state
        self.deployment_queue = asyncio.Queue()
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.cluster_health: Dict[str, float] = {}
        
        # Performance tracking
        self.regional_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.cross_region_latency: Dict[Tuple[str, str], float] = {}
        
        self.logger = get_logger()
    
    async def register_region(self, region_id: str, region_config: Dict[str, Any]):
        """Register a new quantum computation region"""
        self.regions[region_id] = {
            "config": region_config,
            "instances": {},
            "auto_scaler": AutoScaler(),
            "resource_pool": ResourcePool(
                region_config.get("cpu", 100.0),
                region_config.get("memory", 32.0),
                region_config.get("storage", 1000.0)
            ),
            "quantum_coherence_baseline": region_config.get("coherence_baseline", 0.8),
            "max_instances": region_config.get("max_instances", 20),
            "min_instances": region_config.get("min_instances", 1),
            "status": "active"
        }
        
        # Initialize cluster health
        self.cluster_health[region_id] = 1.0
        
        self.logger.info(f"Registered quantum region {region_id}")
    
    async def create_quantum_entanglement_link(self, region_a: str, region_b: str, 
                                             strength: float = 0.8):
        """Create quantum entanglement link between regions"""
        if region_a not in self.regions or region_b not in self.regions:
            raise QuantumTaskPlannerError(f"Invalid regions for entanglement: {region_a}, {region_b}")
        
        # Create bidirectional entanglement
        if region_a not in self.quantum_entanglement_links:
            self.quantum_entanglement_links[region_a] = []
        if region_b not in self.quantum_entanglement_links:
            self.quantum_entanglement_links[region_b] = []
        
        # Add entanglement link
        if region_b not in self.quantum_entanglement_links[region_a]:
            self.quantum_entanglement_links[region_a].append(region_b)
        if region_a not in self.quantum_entanglement_links[region_b]:
            self.quantum_entanglement_links[region_b].append(region_a)
        
        # Measure cross-region latency (simulation)
        latency = np.random.uniform(10, 100)  # 10-100ms
        self.cross_region_latency[(region_a, region_b)] = latency
        self.cross_region_latency[(region_b, region_a)] = latency
        
        self.logger.info(f"Created quantum entanglement link between {region_a} and {region_b} (strength: {strength})")
    
    async def deploy_quantum_cluster(self, region_id: str, cluster_spec: Dict[str, Any]) -> str:
        """Deploy quantum computing cluster in specified region"""
        if region_id not in self.regions:
            raise QuantumTaskPlannerError(f"Region {region_id} not registered")
        
        deployment_id = f"deploy_{region_id}_{int(time.time())}"
        
        # Queue deployment
        deployment_task = {
            "deployment_id": deployment_id,
            "region_id": region_id,
            "cluster_spec": cluster_spec,
            "timestamp": datetime.utcnow(),
            "status": "queued"
        }
        
        await self.deployment_queue.put(deployment_task)
        self.active_deployments[deployment_id] = deployment_task
        
        # Start deployment processor if not running
        asyncio.create_task(self._process_deployments())
        
        return deployment_id
    
    async def _process_deployments(self):
        """Process pending deployments"""
        while True:
            try:
                if self.deployment_queue.empty():
                    await asyncio.sleep(1)
                    continue
                
                deployment = await self.deployment_queue.get()
                await self._execute_deployment(deployment)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Deployment processing error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_deployment(self, deployment: Dict[str, Any]):
        """Execute cluster deployment"""
        deployment_id = deployment["deployment_id"]
        region_id = deployment["region_id"]
        cluster_spec = deployment["cluster_spec"]
        
        try:
            self.active_deployments[deployment_id]["status"] = "deploying"
            
            # Get region configuration
            region = self.regions[region_id]
            instances_requested = cluster_spec.get("instances", 1)
            
            # Check resource availability
            resource_pool = region["resource_pool"]
            cpu_per_instance = cluster_spec.get("cpu_per_instance", 2.0)
            memory_per_instance = cluster_spec.get("memory_per_instance", 4.0)
            storage_per_instance = cluster_spec.get("storage_per_instance", 100.0)
            
            # Reserve resources
            reservation_id = f"cluster_{deployment_id}"
            total_cpu = instances_requested * cpu_per_instance
            total_memory = instances_requested * memory_per_instance
            total_storage = instances_requested * storage_per_instance
            
            if not resource_pool.reserve_resources(
                reservation_id, total_cpu, total_memory, total_storage,
                cluster_spec.get("quantum_coherence", 0.8)
            ):
                raise QuantumTaskPlannerError(f"Insufficient resources in region {region_id}")
            
            # Deploy instances
            deployed_instances = []
            for i in range(instances_requested):
                instance_id = f"{deployment_id}_instance_{i}"
                
                # Simulate deployment time
                await asyncio.sleep(np.random.uniform(2, 5))
                
                # Create instance configuration
                instance_config = {
                    "instance_id": instance_id,
                    "region_id": region_id,
                    "cpu": cpu_per_instance,
                    "memory": memory_per_instance,
                    "storage": storage_per_instance,
                    "quantum_coherence": cluster_spec.get("quantum_coherence", 0.8),
                    "deployed_at": datetime.utcnow(),
                    "status": "running",
                    "endpoint": f"qnode-{instance_id}.{region_id}.quantum.local"
                }
                
                deployed_instances.append(instance_config)
                region["instances"][instance_id] = instance_config
                
                # Register with load balancer
                self.global_load_balancer.register_worker(
                    instance_id,
                    instance_config["endpoint"],
                    instance_config["quantum_coherence"]
                )
                
                self.logger.info(f"Deployed quantum instance {instance_id} in region {region_id}")
            
            # Update deployment status
            self.active_deployments[deployment_id].update({
                "status": "completed",
                "deployed_instances": deployed_instances,
                "completed_at": datetime.utcnow()
            })
            
            self.logger.info(f"Successfully deployed cluster {deployment_id} with {len(deployed_instances)} instances")
            
        except Exception as e:
            self.active_deployments[deployment_id]["status"] = "failed"
            self.active_deployments[deployment_id]["error"] = str(e)
            self.logger.error(f"Failed to deploy cluster {deployment_id}: {e}")
    
    async def scale_cluster_globally(self, target_coherence: float = 0.8, 
                                   load_threshold: float = 0.7) -> Dict[str, Any]:
        """Intelligent global cluster scaling based on quantum metrics"""
        scaling_actions = []
        
        for region_id, region in self.regions.items():
            if region["status"] != "active":
                continue
            
            # Analyze regional metrics
            current_load = await self._calculate_regional_load(region_id)
            coherence_level = await self._get_regional_coherence(region_id)
            
            # Determine scaling need
            scale_decision = None
            
            if current_load > load_threshold and coherence_level > target_coherence:
                # High load, good coherence - scale up
                scale_up_amount = max(1, int((current_load - load_threshold) * 10))
                scale_decision = {
                    "region_id": region_id,
                    "action": "scale_up",
                    "amount": scale_up_amount,
                    "reason": f"high_load_{current_load:.2f}_good_coherence_{coherence_level:.2f}"
                }
            
            elif current_load < load_threshold * 0.5 and len(region["instances"]) > region["min_instances"]:
                # Low load - scale down
                scale_down_amount = max(1, min(2, len(region["instances"]) - region["min_instances"]))
                scale_decision = {
                    "region_id": region_id,
                    "action": "scale_down", 
                    "amount": scale_down_amount,
                    "reason": f"low_load_{current_load:.2f}"
                }
            
            elif coherence_level < target_coherence * 0.8:
                # Low coherence - redistribute load
                scale_decision = {
                    "region_id": region_id,
                    "action": "redistribute",
                    "reason": f"low_coherence_{coherence_level:.2f}"
                }
            
            if scale_decision:
                scaling_actions.append(scale_decision)
                await self._execute_scaling_action(scale_decision)
        
        return {
            "total_actions": len(scaling_actions),
            "scaling_actions": scaling_actions,
            "timestamp": datetime.utcnow(),
            "global_coherence": await self._calculate_global_coherence()
        }
    
    async def _calculate_regional_load(self, region_id: str) -> float:
        """Calculate current load for a region"""
        region = self.regions[region_id]
        
        if not region["instances"]:
            return 0.0
        
        # Simulate load calculation
        total_capacity = len(region["instances"]) * 100  # Each instance can handle 100 units
        current_utilization = np.random.uniform(20, 80) * len(region["instances"])
        
        return current_utilization / total_capacity
    
    async def _get_regional_coherence(self, region_id: str) -> float:
        """Get average quantum coherence for region"""
        region = self.regions[region_id]
        
        if not region["instances"]:
            return region["quantum_coherence_baseline"]
        
        # Calculate weighted average coherence
        coherence_values = [inst["quantum_coherence"] for inst in region["instances"].values()]
        return np.mean(coherence_values)
    
    async def _calculate_global_coherence(self) -> float:
        """Calculate global quantum coherence across all regions"""
        all_coherence_values = []
        
        for region_id in self.regions:
            coherence = await self._get_regional_coherence(region_id)
            all_coherence_values.append(coherence)
        
        return np.mean(all_coherence_values) if all_coherence_values else 0.0
    
    async def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute a scaling action"""
        region_id = action["region_id"]
        action_type = action["action"]
        
        if action_type == "scale_up":
            cluster_spec = {
                "instances": action["amount"],
                "cpu_per_instance": 2.0,
                "memory_per_instance": 4.0,
                "storage_per_instance": 100.0,
                "quantum_coherence": 0.8
            }
            await self.deploy_quantum_cluster(region_id, cluster_spec)
        
        elif action_type == "scale_down":
            region = self.regions[region_id]
            instances_to_remove = list(region["instances"].keys())[:action["amount"]]
            
            for instance_id in instances_to_remove:
                await self._terminate_instance(region_id, instance_id)
        
        elif action_type == "redistribute":
            await self._redistribute_regional_load(region_id)
    
    async def _terminate_instance(self, region_id: str, instance_id: str):
        """Terminate a quantum computing instance"""
        region = self.regions[region_id]
        
        if instance_id not in region["instances"]:
            return
        
        # Remove from load balancer
        self.global_load_balancer.unregister_worker(instance_id)
        
        # Release resources
        instance = region["instances"][instance_id]
        reservation_id = f"instance_{instance_id}"
        region["resource_pool"].release_resources(reservation_id)
        
        # Remove from region
        del region["instances"][instance_id]
        
        self.logger.info(f"Terminated quantum instance {instance_id} in region {region_id}")
    
    async def _redistribute_regional_load(self, region_id: str):
        """Redistribute load within a region for better coherence"""
        # Simulate load redistribution by updating worker stats
        region = self.regions[region_id]
        
        for instance_id, instance in region["instances"].items():
            # Adjust quantum coherence through load redistribution
            coherence_boost = np.random.uniform(0.05, 0.15)
            instance["quantum_coherence"] = min(1.0, instance["quantum_coherence"] + coherence_boost)
            
            # Update load balancer
            self.global_load_balancer.update_worker_stats(instance_id, {
                "quantum_coherence": instance["quantum_coherence"],
                "load_factor": np.random.uniform(0.3, 0.7)
            })
        
        self.logger.info(f"Redistributed load for region {region_id}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        total_instances = sum(len(region["instances"]) for region in self.regions.values())
        active_regions = len([r for r in self.regions.values() if r["status"] == "active"])
        
        regional_status = {}
        for region_id, region in self.regions.items():
            utilization = region["resource_pool"].get_utilization()
            regional_status[region_id] = {
                "instances": len(region["instances"]),
                "status": region["status"],
                "resource_utilization": utilization,
                "health": self.cluster_health.get(region_id, 0.0),
                "entangled_regions": self.quantum_entanglement_links.get(region_id, [])
            }
        
        return {
            "total_instances": total_instances,
            "active_regions": active_regions,
            "regional_status": regional_status,
            "active_deployments": len([d for d in self.active_deployments.values() if d["status"] in ["queued", "deploying"]]),
            "entanglement_links": len(self.quantum_entanglement_links),
            "global_load_distribution": self.global_load_balancer.get_load_distribution()
        }


# Enhanced global instances
_load_balancer = None
_auto_scaler = None
_resource_pool = None
_cluster_orchestrator = None


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


def get_cluster_orchestrator() -> QuantumClusterOrchestrator:
    """Get global cluster orchestrator instance"""
    global _cluster_orchestrator
    if _cluster_orchestrator is None:
        _cluster_orchestrator = QuantumClusterOrchestrator()
    return _cluster_orchestrator


def shutdown_scaling_systems():
    """Shutdown all scaling systems"""
    global _auto_scaler
    if _auto_scaler:
        _auto_scaler.stop_monitoring()