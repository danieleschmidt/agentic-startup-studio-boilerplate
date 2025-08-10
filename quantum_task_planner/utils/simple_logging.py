"""
Simple Logging Implementation

Basic logging setup for immediate functionality.
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime


class SimpleLogger:
    """Simple logger wrapper"""
    
    def __init__(self, level: str = "INFO"):
        self.logger = logging.getLogger("quantum_task_planner")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        if kwargs:
            message = f"{message} - {kwargs}"
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        if kwargs:
            message = f"{message} - {kwargs}"
        self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        if kwargs:
            message = f"{message} - {kwargs}"
        self.logger.warning(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        if kwargs:
            message = f"{message} - {kwargs}"
        self.logger.debug(message)


def setup_logging(level: str = "INFO") -> SimpleLogger:
    """Setup logging system"""
    return SimpleLogger(level)


def get_logger() -> SimpleLogger:
    """Get logger instance"""
    return setup_logging()