"""
Simple Scaling Components

Basic auto-scaling and load balancing for immediate functionality.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging


class SimpleLoadBalancer:
    """Simple load balancer"""
    
    def __init__(self, coherence_weight: float = 0.3):
        self.coherence_weight = coherence_weight
        self.instances: List[str] = []
        self.load_distribution: Dict[str, float] = {}
    
    def add_instance(self, instance_id: str):
        """Add instance to load balancer"""
        self.instances.append(instance_id)
        self.load_distribution[instance_id] = 0.0
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get load distribution statistics"""
        return {
            'instances': len(self.instances),
            'load_distribution': self.load_distribution.copy(),
            'total_load': sum(self.load_distribution.values())
        }


class SimpleAutoScaler:
    """Simple auto-scaler"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.current_instances = 1
        self.min_instances = 1
        self.max_instances = 10
        self.monitoring = False
        self.logger = logging.getLogger(__name__)
    
    async def _scale_up(self, count: int = 1):
        """Scale up instances"""
        new_count = min(self.max_instances, self.current_instances + count)
        self.current_instances = new_count
        self.logger.info(f"Scaled up to {new_count} instances")
    
    async def _scale_down(self, count: int = 1):
        """Scale down instances"""
        new_count = max(self.min_instances, self.current_instances - count)
        self.current_instances = new_count
        self.logger.info(f"Scaled down to {new_count} instances")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status"""
        return {
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'monitoring': self.monitoring,
            'last_check': datetime.utcnow().isoformat()
        }
    
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.monitoring = False


# Factory functions
def get_load_balancer(coherence_weight: float = 0.3) -> SimpleLoadBalancer:
    """Get load balancer instance"""
    return SimpleLoadBalancer(coherence_weight)


def get_auto_scaler(check_interval: float = 30.0) -> SimpleAutoScaler:
    """Get auto-scaler instance"""
    return SimpleAutoScaler(check_interval)