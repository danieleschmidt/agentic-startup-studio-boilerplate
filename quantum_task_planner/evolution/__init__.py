"""
Quantum Task Planner - Evolution Module

Self-improving and autonomous enhancement capabilities for the quantum task planner.
This module implements evolutionary algorithms that allow the system to autonomously
improve its performance, adapt to new challenges, and evolve new capabilities.
"""

__version__ = "4.0.0"
__author__ = "Terragon Labs - Autonomous Evolution Engine"

# Import available components without networkx dependencies
try:
    from .autonomous_evolution_engine import AutonomousEvolutionEngine
    AUTONOMOUS_EVOLUTION_AVAILABLE = True
except ImportError:
    AUTONOMOUS_EVOLUTION_AVAILABLE = False

try:
    from .self_improving_algorithms import SelfImprovingAlgorithms
    SELF_IMPROVING_AVAILABLE = True
except ImportError:
    SELF_IMPROVING_AVAILABLE = False

try:
    from .adaptive_quantum_framework import AdaptiveQuantumFramework
    ADAPTIVE_FRAMEWORK_AVAILABLE = True
except ImportError:
    ADAPTIVE_FRAMEWORK_AVAILABLE = False

# Import Generation 7 meta-learning without networkx dependencies
try:
    from .generation_7_meta_learning_consciousness import (
        MetaLearningConsciousnessEngine,
        MetaLearningStrategy,
        SelfImprovementMode
    )
    GENERATION_7_AVAILABLE = True
except ImportError:
    GENERATION_7_AVAILABLE = False

# Build __all__ list based on available components
__all__ = []

if AUTONOMOUS_EVOLUTION_AVAILABLE:
    __all__.append("AutonomousEvolutionEngine")

if SELF_IMPROVING_AVAILABLE:
    __all__.append("SelfImprovingAlgorithms")

if ADAPTIVE_FRAMEWORK_AVAILABLE:
    __all__.append("AdaptiveQuantumFramework")

if GENERATION_7_AVAILABLE:
    __all__.extend([
        "MetaLearningConsciousnessEngine",
        "MetaLearningStrategy", 
        "SelfImprovementMode"
    ])