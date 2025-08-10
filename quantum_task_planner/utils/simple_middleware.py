"""
Simple Middleware Implementation

Basic middleware components for immediate functionality.
"""

from typing import Dict, Any, Optional
import time
import logging


class SimpleSecurityConfig:
    """Simple security configuration"""
    def __init__(self):
        self.api_key_required = False
        self.rate_limit = 1000
        self.cors_enabled = True


def create_default_security_config() -> SimpleSecurityConfig:
    """Create default security config"""
    return SimpleSecurityConfig()


def setup_middleware(app, security_config=None, enable_quantum_middleware=True, enable_monitoring=True):
    """Setup simple middleware"""
    
    middleware_instances = {}
    
    if enable_monitoring:
        middleware_instances['monitoring'] = SimpleMonitoringMiddleware()
    
    return middleware_instances


class SimpleMonitoringMiddleware:
    """Simple monitoring middleware"""
    
    def __init__(self):
        self.metrics = {
            'requests': 0,
            'errors': 0,
            'response_times': []
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        return {
            'total_requests': self.metrics['requests'],
            'total_errors': self.metrics['errors'],
            'avg_response_time': sum(self.metrics['response_times'][-100:]) / max(1, len(self.metrics['response_times'][-100:]))
        }