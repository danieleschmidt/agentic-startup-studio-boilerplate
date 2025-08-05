"""
Quantum-Inspired Task Planner

A revolutionary task planning system that applies quantum computing principles
to optimize task scheduling, resource allocation, and execution strategies.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__license__ = "Apache-2.0"

from .core.quantum_task import QuantumTask, TaskState
from .core.quantum_scheduler import QuantumTaskScheduler
from .core.quantum_optimizer import QuantumProbabilityOptimizer
from .core.entanglement_manager import TaskEntanglementManager
from .api.quantum_api import QuantumPlannerAPI

__all__ = [
    "QuantumTask",
    "TaskState", 
    "QuantumTaskScheduler",
    "QuantumProbabilityOptimizer",
    "TaskEntanglementManager",
    "QuantumPlannerAPI"
]