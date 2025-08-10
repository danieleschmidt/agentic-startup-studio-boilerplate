"""
Quantum Research Module

Advanced quantum computing research implementations for task optimization,
multi-agent coordination, and novel algorithmic contributions.
"""

from .quantum_annealing_optimizer import QuantumAnnealingOptimizer
from .adiabatic_task_scheduler import AdiabaticTaskScheduler
from .variational_resource_optimizer import VariationalResourceOptimizer
from .quantum_approximate_optimization import QAOATaskPlanner
from .multi_agent_quantum_coordinator import MultiAgentQuantumCoordinator
from .research_benchmarks import QuantumResearchBenchmarks

__all__ = [
    "QuantumAnnealingOptimizer",
    "AdiabaticTaskScheduler", 
    "VariationalResourceOptimizer",
    "QAOATaskPlanner",
    "MultiAgentQuantumCoordinator",
    "QuantumResearchBenchmarks"
]