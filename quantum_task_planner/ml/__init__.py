"""
Machine Learning Module for Quantum Task Optimization

Advanced ML system including:
- Deep Reinforcement Learning for task scheduling
- Neural networks for quantum coherence prediction
- Traditional ML for performance modeling
- Transfer learning and optimization insights
"""

from .quantum_ml_optimizer import (
    QuantumMLOptimizer,
    QuantumReinforcementLearner,
    QuantumStateEncoder,
    OptimizationExperience,
    OptimizationMetrics,
    get_ml_optimizer
)

# Only import PyTorch models if available
try:
    from .quantum_ml_optimizer import QuantumDQN, QuantumCoherencePredictor
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False

__all__ = [
    "QuantumMLOptimizer",
    "QuantumReinforcementLearner", 
    "QuantumStateEncoder",
    "OptimizationExperience",
    "OptimizationMetrics",
    "get_ml_optimizer"
]

if ML_MODELS_AVAILABLE:
    __all__.extend(["QuantumDQN", "QuantumCoherencePredictor"])