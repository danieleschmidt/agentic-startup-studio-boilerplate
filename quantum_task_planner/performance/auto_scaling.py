"""
Advanced Auto-Scaling System

Intelligent auto-scaling with predictive algorithms, quantum load balancing, and resource optimization.
"""

import asyncio
import time
import math
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import statistics
from collections import deque, defaultdict

from ..utils.exceptions import PerformanceError, ConfigurationError
from ..utils.robust_logging import QuantumLoggerAdapter, performance_logger


class ScalingDirection(Enum):
    """Scaling direction"""
    UP = "up"
    DOWN = "down" 
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Scaling trigger types"""
    CPU_UTILIZATION = "cpu"
    MEMORY_UTILIZATION = "memory"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    QUANTUM_COHERENCE = "quantum_coherence"
    CUSTOM_METRIC = "custom"


@dataclass
class ScalingMetric:
    """Scaling metric configuration"""
    name: str
    trigger_type: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    evaluation_periods: int = 3
    datapoints_to_alarm: int = 2
    weight: float = 1.0
    custom_evaluator: Optional[Callable[[List[float]], float]] = None
    
    def evaluate(self, values: List[float]) -> Tuple[bool, bool, float]:
        """Evaluate metric for scaling decisions
        
        Returns:
            (should_scale_up, should_scale_down, current_value)
        """
        if not values:
            return False, False, 0.0
        
        # Use custom evaluator if provided
        if self.custom_evaluator:
            current_value = self.custom_evaluator(values)
        else:
            # Take recent values for evaluation
            recent_values = values[-self.evaluation_periods:]
            
            # Use different aggregation based on trigger type
            if self.trigger_type in [ScalingTrigger.CPU_UTILIZATION, ScalingTrigger.MEMORY_UTILIZATION]:
                current_value = statistics.mean(recent_values)
            elif self.trigger_type == ScalingTrigger.RESPONSE_TIME:
                current_value = max(recent_values)  # Use max for response time
            elif self.trigger_type == ScalingTrigger.QUEUE_LENGTH:
                current_value = max(recent_values)  # Use max for queue length
            elif self.trigger_type == ScalingTrigger.QUANTUM_COHERENCE:
                current_value = min(recent_values)  # Use min for coherence (lower is worse)
            else:
                current_value = statistics.mean(recent_values)
        
        # Determine scaling needs
        alarming_values_up = sum(1 for v in recent_values if v >= self.scale_up_threshold)
        alarming_values_down = sum(1 for v in recent_values if v <= self.scale_down_threshold)
        
        should_scale_up = alarming_values_up >= self.datapoints_to_alarm
        should_scale_down = alarming_values_down >= self.datapoints_to_alarm
        
        return should_scale_up, should_scale_down, current_value


@dataclass
class ScalingAction:
    """Scaling action record"""
    timestamp: float
    direction: ScalingDirection
    instances_before: int
    instances_after: int
    trigger_metrics: Dict[str, float]
    reason: str
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'direction': self.direction.value,
            'instances_before': self.instances_before,
            'instances_after': self.instances_after,
            'trigger_metrics': self.trigger_metrics,
            'reason': self.reason,
            'success': self.success,
            'error_message': self.error_message
        }


class PredictiveScaler:
    """Predictive scaling based on historical patterns"""
    
    def __init__(self, lookback_hours: int = 24, prediction_horizon_minutes: int = 15):
        self.lookback_hours = lookback_hours
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.historical_data: deque = deque(maxlen=int(lookback_hours * 60 / 5))  # 5-minute intervals
        self.patterns: Dict[str, List[float]] = defaultdict(list)
        
    def record_metrics(self, timestamp: float, metrics: Dict[str, float]):
        """Record metrics for pattern learning"""
        hour_of_day = datetime.fromtimestamp(timestamp).hour
        day_of_week = datetime.fromtimestamp(timestamp).weekday()
        
        self.historical_data.append({
            'timestamp': timestamp,
            'hour': hour_of_day,
            'day': day_of_week,
            'metrics': metrics.copy()
        })
        
        # Update patterns
        pattern_key = f"{day_of_week}_{hour_of_day}"
        for metric_name, value in metrics.items():
            self.patterns[f"{pattern_key}_{metric_name}"].append(value)
            # Keep only recent patterns
            if len(self.patterns[f"{pattern_key}_{metric_name}"]) > 100:
                self.patterns[f"{pattern_key}_{metric_name}"] = \
                    self.patterns[f"{pattern_key}_{metric_name}"][-50:]
    
    def predict_demand(self, current_time: float) -> Dict[str, float]:
        """Predict future demand based on patterns"""
        future_time = current_time + (self.prediction_horizon_minutes * 60)
        future_dt = datetime.fromtimestamp(future_time)
        future_hour = future_dt.hour
        future_day = future_dt.weekday()
        
        predictions = {}
        pattern_key = f"{future_day}_{future_hour}"
        
        # Generate predictions for each metric
        for key, values in self.patterns.items():
            if key.startswith(pattern_key):
                metric_name = key.split(f"{pattern_key}_", 1)[1]
                if values:
                    # Use trend analysis
                    recent_avg = statistics.mean(values[-10:]) if len(values) >= 10 else statistics.mean(values)
                    historical_avg = statistics.mean(values)
                    
                    # Apply trend factor
                    trend_factor = recent_avg / historical_avg if historical_avg > 0 else 1.0
                    prediction = historical_avg * trend_factor
                    
                    predictions[metric_name] = max(0, prediction)
        
        return predictions
    
    def should_proactive_scale(self, current_metrics: Dict[str, float], 
                             scaling_metrics: List[ScalingMetric]) -> Optional[ScalingDirection]:
        """Determine if proactive scaling is needed"""
        predictions = self.predict_demand(time.time())
        
        if not predictions:
            return None
        
        # Evaluate predicted metrics against thresholds
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        
        for scaling_metric in scaling_metrics:
            if scaling_metric.name not in predictions:
                continue
            
            predicted_value = predictions[scaling_metric.name]
            
            if predicted_value >= scaling_metric.scale_up_threshold:
                scale_up_votes += scaling_metric.weight
            elif predicted_value <= scaling_metric.scale_down_threshold:
                scale_down_votes += scaling_metric.weight
            
            total_weight += scaling_metric.weight
        
        if total_weight == 0:
            return None
        
        # Require majority vote for proactive scaling
        if scale_up_votes / total_weight > 0.6:
            return ScalingDirection.UP
        elif scale_down_votes / total_weight > 0.6:
            return ScalingDirection.DOWN
        
        return None


class QuantumLoadBalancer:
    """Quantum-aware load balancer for optimal task distribution"""
    
    def __init__(self, coherence_weight: float = 0.3):
        self.coherence_weight = coherence_weight
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.logger = QuantumLoggerAdapter(
            import_logging().getLogger(__name__),
            component="quantum_load_balancer"
        )
    
    def register_instance(self, instance_id: str, capacity: float = 1.0, 
                         quantum_coherence: float = 1.0):
        """Register compute instance"""
        self.instances[instance_id] = {
            'capacity': capacity,
            'current_load': 0.0,
            'quantum_coherence': quantum_coherence,
            'last_updated': time.time(),
            'task_count': 0,
            'health_score': 1.0
        }
        
        self.logger.info(f"Registered instance {instance_id}")
    
    def unregister_instance(self, instance_id: str):
        """Unregister compute instance"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            if instance_id in self.load_history:
                del self.load_history[instance_id]
            
            self.logger.info(f"Unregistered instance {instance_id}")
    
    def update_instance_load(self, instance_id: str, load: float, 
                           quantum_coherence: float = None, 
                           task_count: int = None):
        """Update instance load metrics"""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            instance['current_load'] = load
            instance['last_updated'] = time.time()
            
            if quantum_coherence is not None:
                instance['quantum_coherence'] = quantum_coherence
            
            if task_count is not None:
                instance['task_count'] = task_count
            
            # Record load history
            self.load_history[instance_id].append({
                'timestamp': time.time(),
                'load': load,
                'coherence': instance['quantum_coherence'],
                'tasks': instance['task_count']
            })
    
    def calculate_quantum_score(self, instance: Dict[str, Any]) -> float:
        """Calculate quantum-aware load balancing score"""
        # Base utilization (lower is better)
        utilization_score = 1.0 - (instance['current_load'] / instance['capacity'])
        
        # Quantum coherence (higher is better)
        coherence_score = instance['quantum_coherence']
        
        # Health score (higher is better)
        health_score = instance['health_score']
        
        # Task density consideration (lower density is better for new tasks)
        task_density = instance['task_count'] / instance['capacity']
        density_score = 1.0 / (1.0 + task_density)
        
        # Combine scores with weights
        quantum_score = (
            utilization_score * 0.4 +
            coherence_score * self.coherence_weight +
            health_score * 0.2 +
            density_score * 0.1
        )
        
        return quantum_score
    
    def select_optimal_instance(self, task_requirements: Dict[str, Any] = None) -> Optional[str]:
        """Select optimal instance for task placement"""
        if not self.instances:
            return None
        
        # Filter available instances
        available_instances = {
            instance_id: instance 
            for instance_id, instance in self.instances.items()
            if instance['current_load'] < instance['capacity'] * 0.9  # Keep 10% headroom
        }
        
        if not available_instances:
            # Return least loaded instance as fallback
            return min(self.instances.keys(), 
                      key=lambda x: self.instances[x]['current_load'])
        
        # Calculate quantum scores
        scored_instances = [
            (instance_id, self.calculate_quantum_score(instance))
            for instance_id, instance in available_instances.items()
        ]
        
        # Consider task requirements if provided
        if task_requirements:
            required_coherence = task_requirements.get('min_coherence', 0.0)
            scored_instances = [
                (instance_id, score) 
                for instance_id, score in scored_instances
                if self.instances[instance_id]['quantum_coherence'] >= required_coherence
            ]
        
        if not scored_instances:
            return None
        
        # Select instance with highest quantum score
        best_instance = max(scored_instances, key=lambda x: x[1])
        return best_instance[0]
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution"""
        if not self.instances:
            return {'instances': 0, 'total_load': 0, 'average_load': 0}
        
        total_load = sum(instance['current_load'] for instance in self.instances.values())
        average_load = total_load / len(self.instances)
        
        instance_loads = {
            instance_id: {
                'load': instance['current_load'],
                'capacity': instance['capacity'],
                'utilization': instance['current_load'] / instance['capacity'],
                'quantum_coherence': instance['quantum_coherence'],
                'task_count': instance['task_count'],
                'quantum_score': self.calculate_quantum_score(instance)
            }
            for instance_id, instance in self.instances.items()
        }
        
        return {
            'instances': len(self.instances),
            'total_load': total_load,
            'average_load': average_load,
            'load_std_dev': statistics.stdev([i['current_load'] for i in self.instances.values()]) if len(self.instances) > 1 else 0,
            'instance_loads': instance_loads
        }


class AdvancedAutoScaler:
    """Advanced auto-scaler with quantum awareness and predictive capabilities"""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 20,
        target_utilization: float = 0.7,
        scale_up_cooldown: int = 300,  # 5 minutes
        scale_down_cooldown: int = 600,  # 10 minutes
        enable_predictive: bool = True
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.scale_up_cooldown = scale_up_cooldown
        self.scale_down_cooldown = scale_down_cooldown
        self.enable_predictive = enable_predictive
        
        # Current state
        self.current_instances = min_instances
        self.last_scale_action = 0
        self.last_scale_direction = ScalingDirection.MAINTAIN
        
        # Metrics and monitoring
        self.metrics: List[ScalingMetric] = []
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.scaling_history: List[ScalingAction] = []
        
        # Components
        self.load_balancer = QuantumLoadBalancer()
        self.predictive_scaler = PredictiveScaler() if enable_predictive else None
        
        # Callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        self.logger = QuantumLoggerAdapter(
            import_logging().getLogger(__name__),
            component="auto_scaler"
        )
        
        # Initialize default metrics
        self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Setup default scaling metrics"""
        self.add_metric(ScalingMetric(
            name="cpu_utilization",
            trigger_type=ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            weight=1.0
        ))
        
        self.add_metric(ScalingMetric(
            name="memory_utilization", 
            trigger_type=ScalingTrigger.MEMORY_UTILIZATION,
            scale_up_threshold=85.0,
            scale_down_threshold=40.0,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            weight=0.8
        ))
        
        self.add_metric(ScalingMetric(
            name="quantum_coherence",
            trigger_type=ScalingTrigger.QUANTUM_COHERENCE,
            scale_up_threshold=0.3,  # Scale up if coherence drops below 30%
            scale_down_threshold=0.8,  # Scale down if coherence is above 80%
            evaluation_periods=3,
            datapoints_to_alarm=2,
            weight=0.5
        ))
    
    def add_metric(self, metric: ScalingMetric):
        """Add scaling metric"""
        self.metrics.append(metric)
        self.logger.info(f"Added scaling metric: {metric.name}")
    
    def set_scale_callbacks(self, scale_up_cb: Callable[[int], bool], 
                           scale_down_cb: Callable[[int], bool]):
        """Set scaling callbacks"""
        self.scale_up_callback = scale_up_cb
        self.scale_down_callback = scale_down_cb
    
    def record_metric(self, metric_name: str, value: float):
        """Record metric value"""
        timestamp = time.time()
        self.metric_history[metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
        
        # Update predictive scaler
        if self.predictive_scaler:
            self.predictive_scaler.record_metrics(timestamp, {metric_name: value})
    
    def get_recent_metric_values(self, metric_name: str, count: int = 10) -> List[float]:
        """Get recent metric values"""
        history = self.metric_history[metric_name]
        recent = list(history)[-count:] if history else []
        return [entry['value'] for entry in recent]
    
    @performance_logger("scaling_decision")
    async def evaluate_scaling_decision(self) -> Tuple[ScalingDirection, str, Dict[str, float]]:
        """Evaluate whether scaling is needed"""
        current_time = time.time()
        
        # Check cooldown periods
        time_since_last_scale = current_time - self.last_scale_action
        
        if (self.last_scale_direction == ScalingDirection.UP and 
            time_since_last_scale < self.scale_up_cooldown):
            return ScalingDirection.MAINTAIN, "Scale up cooldown active", {}
        
        if (self.last_scale_direction == ScalingDirection.DOWN and 
            time_since_last_scale < self.scale_down_cooldown):
            return ScalingDirection.MAINTAIN, "Scale down cooldown active", {}
        
        # Evaluate metrics
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        current_metric_values = {}
        
        for metric in self.metrics:
            recent_values = self.get_recent_metric_values(metric.name, metric.evaluation_periods)
            
            if not recent_values:
                continue
            
            should_scale_up, should_scale_down, current_value = metric.evaluate(recent_values)
            current_metric_values[metric.name] = current_value
            
            if should_scale_up:
                scale_up_votes += metric.weight
            elif should_scale_down:
                scale_down_votes += metric.weight
            
            total_weight += metric.weight
        
        if total_weight == 0:
            return ScalingDirection.MAINTAIN, "No metrics available", current_metric_values
        
        # Make scaling decision based on votes
        scale_up_ratio = scale_up_votes / total_weight
        scale_down_ratio = scale_down_votes / total_weight
        
        # Require majority vote (> 50%) for scaling decisions
        if scale_up_ratio > 0.5 and self.current_instances < self.max_instances:
            reason = f"Scale up triggered: {scale_up_ratio:.1%} vote ({scale_up_votes:.1f}/{total_weight:.1f})"
            return ScalingDirection.UP, reason, current_metric_values
        
        elif scale_down_ratio > 0.5 and self.current_instances > self.min_instances:
            reason = f"Scale down triggered: {scale_down_ratio:.1%} vote ({scale_down_votes:.1f}/{total_weight:.1f})"
            return ScalingDirection.DOWN, reason, current_metric_values
        
        # Check predictive scaling
        if self.predictive_scaler and time_since_last_scale > self.scale_up_cooldown:
            predicted_direction = self.predictive_scaler.should_proactive_scale(
                current_metric_values, self.metrics
            )
            
            if predicted_direction == ScalingDirection.UP and self.current_instances < self.max_instances:
                return ScalingDirection.UP, "Predictive scaling up", current_metric_values
            elif predicted_direction == ScalingDirection.DOWN and self.current_instances > self.min_instances:
                return ScalingDirection.DOWN, "Predictive scaling down", current_metric_values
        
        return ScalingDirection.MAINTAIN, "Metrics within acceptable range", current_metric_values
    
    async def execute_scaling(self, direction: ScalingDirection, reason: str, 
                            trigger_metrics: Dict[str, float]) -> bool:
        """Execute scaling action"""
        if direction == ScalingDirection.MAINTAIN:
            return True
        
        instances_before = self.current_instances
        success = False
        error_message = None
        
        try:
            if direction == ScalingDirection.UP:
                # Calculate how many instances to add (1-3 based on urgency)
                scale_factor = min(3, max(1, int(self.current_instances * 0.2)))
                new_instances = min(self.max_instances, self.current_instances + scale_factor)
                
                if self.scale_up_callback:
                    success = await self._call_callback(self.scale_up_callback, new_instances)
                else:
                    success = True  # Default success if no callback
                
                if success:
                    self.current_instances = new_instances
            
            elif direction == ScalingDirection.DOWN:
                # Calculate how many instances to remove (1-2 based on over-provisioning)
                scale_factor = min(2, max(1, int((self.current_instances - self.min_instances) * 0.3)))
                new_instances = max(self.min_instances, self.current_instances - scale_factor)
                
                if self.scale_down_callback:
                    success = await self._call_callback(self.scale_down_callback, new_instances)
                else:
                    success = True  # Default success if no callback
                
                if success:
                    self.current_instances = new_instances
        
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"Scaling execution failed: {e}")
        
        # Record scaling action
        action = ScalingAction(
            timestamp=time.time(),
            direction=direction,
            instances_before=instances_before,
            instances_after=self.current_instances,
            trigger_metrics=trigger_metrics.copy(),
            reason=reason,
            success=success,
            error_message=error_message
        )
        
        self.scaling_history.append(action)
        
        if success and direction != ScalingDirection.MAINTAIN:
            self.last_scale_action = time.time()
            self.last_scale_direction = direction
            
            self.logger.info(f"Scaling {direction.value}: {instances_before} -> {self.current_instances}")
        
        return success
    
    async def _call_callback(self, callback: Callable, target_instances: int) -> bool:
        """Call scaling callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                return await callback(target_instances)
            else:
                return callback(target_instances)
        except Exception as e:
            self.logger.error(f"Scaling callback failed: {e}")
            return False
    
    async def start_monitoring(self, check_interval: float = 60.0):
        """Start auto-scaling monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(check_interval))
        self.logger.info("Started auto-scaling monitoring")
    
    async def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped auto-scaling monitoring")
    
    async def _monitoring_loop(self, check_interval: float):
        """Auto-scaling monitoring loop"""
        while self.running:
            try:
                direction, reason, metrics = await self.evaluate_scaling_decision()
                
                if direction != ScalingDirection.MAINTAIN:
                    await self.execute_scaling(direction, reason, metrics)
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Auto-scaling monitoring error: {e}")
                await asyncio.sleep(check_interval)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        current_time = time.time()
        
        recent_metrics = {}
        for metric in self.metrics:
            recent_values = self.get_recent_metric_values(metric.name, 1)
            if recent_values:
                recent_metrics[metric.name] = recent_values[0]
        
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'target_utilization': self.target_utilization,
            'last_scale_action': {
                'timestamp': self.last_scale_action,
                'direction': self.last_scale_direction.value,
                'seconds_ago': current_time - self.last_scale_action if self.last_scale_action > 0 else None
            },
            'cooldown_status': {
                'scale_up_ready': (current_time - self.last_scale_action) > self.scale_up_cooldown,
                'scale_down_ready': (current_time - self.last_scale_action) > self.scale_down_cooldown
            },
            'recent_metrics': recent_metrics,
            'monitoring': self.running,
            'predictive_enabled': self.enable_predictive
        }
    
    def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get scaling history"""
        return [action.to_dict() for action in self.scaling_history[-limit:]]


def import_logging():
    """Import logging module"""
    import logging
    return logging