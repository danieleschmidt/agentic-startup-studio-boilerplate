"""
Quantum Neural Optimization Engine - Generation 1 Enhancement

Implements quantum-neural hybrid optimization combining quantum annealing
with neural network learning for superior task scheduling and resource allocation.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor
import logging

from .quantum_task import QuantumTask, TaskState, TaskPriority


@dataclass
class NeuralWeight:
    """Quantum-enhanced neural network weight with uncertainty"""
    value: float
    uncertainty: float = 0.01
    last_update: datetime = field(default_factory=datetime.utcnow)
    update_count: int = 0
    
    def quantum_update(self, gradient: float, learning_rate: float = 0.01, quantum_noise: float = 0.001):
        """Update weight with quantum noise for enhanced exploration"""
        # Standard gradient descent with quantum enhancement
        quantum_perturbation = np.random.normal(0, quantum_noise)
        self.value -= learning_rate * (gradient + quantum_perturbation)
        
        # Update uncertainty based on gradient magnitude
        self.uncertainty = max(0.001, self.uncertainty * 0.99 + abs(gradient) * 0.01)
        self.last_update = datetime.utcnow()
        self.update_count += 1
    
    def get_quantum_value(self) -> float:
        """Get weight value with quantum uncertainty sampling"""
        return np.random.normal(self.value, self.uncertainty)


class QuantumNeuralLayer:
    """Quantum-enhanced neural network layer"""
    
    def __init__(self, input_size: int, output_size: int, activation: str = "quantum_sigmoid"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize quantum neural weights
        self.weights = [
            [NeuralWeight(np.random.normal(0, 0.1)) for _ in range(output_size)]
            for _ in range(input_size)
        ]
        self.biases = [NeuralWeight(np.random.normal(0, 0.01)) for _ in range(output_size)]
        
        # Quantum coherence tracking
        self.layer_coherence = 1.0
        self.quantum_entanglement_strength = 0.1
        
    def quantum_forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass with quantum enhancement"""
        outputs = np.zeros(self.output_size)
        
        for j in range(self.output_size):
            # Quantum-enhanced weighted sum
            weighted_sum = sum(
                inputs[i] * self.weights[i][j].get_quantum_value()
                for i in range(self.input_size)
            ) + self.biases[j].get_quantum_value()
            
            # Apply quantum activation function
            outputs[j] = self._quantum_activation(weighted_sum)
        
        # Apply quantum entanglement effects
        if self.quantum_entanglement_strength > 0:
            outputs = self._apply_quantum_entanglement(outputs)
        
        # Update layer coherence
        self._update_coherence()
        
        return outputs
    
    def _quantum_activation(self, x: float) -> float:
        """Quantum-enhanced activation functions"""
        if self.activation == "quantum_sigmoid":
            # Sigmoid with quantum uncertainty
            base_sigmoid = 1.0 / (1.0 + np.exp(-x))
            quantum_noise = np.random.normal(0, 0.01 * self.layer_coherence)
            return np.clip(base_sigmoid + quantum_noise, 0, 1)
        
        elif self.activation == "quantum_tanh":
            # Tanh with quantum enhancement
            base_tanh = np.tanh(x)
            quantum_phase = np.cos(x * self.layer_coherence) * 0.05
            return np.clip(base_tanh + quantum_phase, -1, 1)
        
        elif self.activation == "quantum_relu":
            # ReLU with quantum tunneling
            if x > 0:
                return x * (1 + np.random.exponential(0.01))
            else:
                # Quantum tunneling allows small negative values through
                tunnel_prob = np.exp(x * self.layer_coherence)
                return x * tunnel_prob if np.random.random() < tunnel_prob else 0
        
        else:  # Linear activation
            return x
    
    def _apply_quantum_entanglement(self, outputs: np.ndarray) -> np.ndarray:
        """Apply quantum entanglement between neurons"""
        entangled_outputs = outputs.copy()
        
        # Create entanglement correlations
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if np.random.random() < self.quantum_entanglement_strength:
                    # Entangle neurons with quantum correlation
                    correlation = np.cos(outputs[i] + outputs[j]) * 0.1
                    entangled_outputs[i] += correlation
                    entangled_outputs[j] += correlation
        
        return entangled_outputs
    
    def _update_coherence(self, decay_rate: float = 0.001):
        """Update quantum coherence of the layer"""
        self.layer_coherence = max(0.1, self.layer_coherence - decay_rate)
    
    def quantum_backward(self, gradient: np.ndarray, inputs: np.ndarray, learning_rate: float = 0.01):
        """Quantum-enhanced backpropagation"""
        # Calculate weight gradients with quantum enhancement
        for i in range(self.input_size):
            for j in range(self.output_size):
                weight_gradient = inputs[i] * gradient[j]
                self.weights[i][j].quantum_update(weight_gradient, learning_rate)
        
        # Update biases
        for j in range(self.output_size):
            self.biases[j].quantum_update(gradient[j], learning_rate)
        
        # Enhance coherence after learning
        self.layer_coherence = min(1.0, self.layer_coherence + 0.001)


class QuantumNeuralNetwork:
    """Multi-layer quantum neural network for optimization"""
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None):
        self.layer_sizes = layer_sizes
        self.layers: List[QuantumNeuralLayer] = []
        
        if activations is None:
            activations = ["quantum_sigmoid"] * (len(layer_sizes) - 1)
        
        # Build quantum neural layers
        for i in range(len(layer_sizes) - 1):
            layer = QuantumNeuralLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i] if i < len(activations) else "quantum_sigmoid"
            )
            self.layers.append(layer)
        
        self.network_coherence = 1.0
        self.training_iterations = 0
        self.performance_history: List[float] = []
    
    def quantum_predict(self, inputs: np.ndarray) -> np.ndarray:
        """Forward prediction with quantum enhancement"""
        current_outputs = inputs.copy()
        
        # Forward pass through all layers
        for layer in self.layers:
            current_outputs = layer.quantum_forward(current_outputs)
        
        # Apply network-level quantum effects
        current_outputs = self._apply_network_quantum_effects(current_outputs)
        
        return current_outputs
    
    def quantum_train(self, training_data: List[Tuple[np.ndarray, np.ndarray]], 
                     epochs: int = 100, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Quantum-enhanced training with adaptive learning"""
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for inputs, targets in training_data:
                # Forward pass
                predictions = self.quantum_predict(inputs)
                
                # Calculate quantum loss
                loss = self._quantum_loss(predictions, targets)
                epoch_loss += loss
                
                # Quantum backpropagation
                self._quantum_backprop(inputs, targets, predictions, learning_rate)
            
            avg_loss = epoch_loss / len(training_data)
            training_losses.append(avg_loss)
            
            # Adaptive learning rate with quantum annealing
            learning_rate *= 0.999  # Gradual cooldown
            
            # Update network coherence
            self._update_network_coherence(avg_loss)
            
            self.training_iterations += 1
        
        self.performance_history.extend(training_losses)
        
        return {
            "training_completed": True,
            "epochs": epochs,
            "final_loss": training_losses[-1] if training_losses else 0,
            "network_coherence": self.network_coherence,
            "training_losses": training_losses
        }
    
    def _apply_network_quantum_effects(self, outputs: np.ndarray) -> np.ndarray:
        """Apply network-level quantum effects"""
        # Quantum superposition effects
        superposition_factor = self.network_coherence * 0.05
        quantum_superposition = np.random.normal(0, superposition_factor, len(outputs))
        
        # Quantum interference patterns
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                phase_diff = outputs[i] - outputs[j]
                interference = np.cos(phase_diff) * superposition_factor
                outputs[i] += interference
                outputs[j] -= interference
        
        return outputs
    
    def _quantum_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Quantum-enhanced loss function"""
        # Base mean squared error
        mse_loss = np.mean((predictions - targets) ** 2)
        
        # Add quantum uncertainty penalty
        uncertainty_penalty = sum(
            sum(weight.uncertainty for weight in layer.biases)
            for layer in self.layers
        ) * 0.001
        
        # Coherence regularization
        coherence_penalty = (1.0 - self.network_coherence) * 0.01
        
        return mse_loss + uncertainty_penalty + coherence_penalty
    
    def _quantum_backprop(self, inputs: np.ndarray, targets: np.ndarray, 
                         predictions: np.ndarray, learning_rate: float):
        """Quantum-enhanced backpropagation"""
        # Calculate output gradient
        output_gradient = 2 * (predictions - targets) / len(targets)
        
        # Backpropagate through layers in reverse
        current_gradient = output_gradient
        layer_inputs = [inputs] + [None] * (len(self.layers) - 1)
        
        # Store intermediate outputs for backprop
        current_input = inputs
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer_inputs[i] = current_input
            current_input = layer.quantum_forward(current_input)
        
        # Backpropagate gradients
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            layer_input = layer_inputs[i]
            
            # Apply quantum gradient enhancement
            quantum_enhanced_gradient = self._enhance_gradient_quantum(current_gradient)
            
            # Update layer weights
            layer.quantum_backward(quantum_enhanced_gradient, layer_input, learning_rate)
            
            # Calculate gradient for previous layer (if not input layer)
            if i > 0:
                next_gradient = np.zeros(layer.input_size)
                for j in range(layer.input_size):
                    for k in range(layer.output_size):
                        next_gradient[j] += (current_gradient[k] * 
                                           layer.weights[j][k].get_quantum_value())
                current_gradient = next_gradient
    
    def _enhance_gradient_quantum(self, gradient: np.ndarray) -> np.ndarray:
        """Enhance gradient with quantum effects"""
        enhanced_gradient = gradient.copy()
        
        # Add quantum momentum
        quantum_momentum = np.random.normal(0, 0.01 * self.network_coherence, len(gradient))
        enhanced_gradient += quantum_momentum
        
        # Apply quantum tunneling to help escape local minima
        for i in range(len(enhanced_gradient)):
            if abs(enhanced_gradient[i]) < 0.001:  # Near zero gradient
                tunnel_boost = np.random.exponential(0.01)
                enhanced_gradient[i] += tunnel_boost * np.sign(np.random.randn())
        
        return enhanced_gradient
    
    def _update_network_coherence(self, current_loss: float):
        """Update network quantum coherence based on performance"""
        if len(self.performance_history) > 0:
            if current_loss < self.performance_history[-1]:
                # Improving - maintain coherence
                self.network_coherence = min(1.0, self.network_coherence + 0.001)
            else:
                # Not improving - slight decoherence
                self.network_coherence = max(0.1, self.network_coherence - 0.0005)
        
        # Random quantum fluctuations
        quantum_noise = np.random.normal(0, 0.001)
        self.network_coherence = np.clip(self.network_coherence + quantum_noise, 0.1, 1.0)


class QuantumNeuralOptimizer:
    """
    Advanced quantum-neural hybrid optimizer for task scheduling and resource allocation.
    Combines quantum annealing with neural network learning for superior optimization.
    """
    
    def __init__(self, input_features: int = 10, hidden_layers: List[int] = None):
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        # Build quantum neural network architecture
        layer_sizes = [input_features] + hidden_layers + [1]  # Single output for optimization score
        activations = ["quantum_sigmoid"] * (len(layer_sizes) - 2) + ["linear"]
        
        self.neural_network = QuantumNeuralNetwork(layer_sizes, activations)
        
        # Quantum annealing parameters
        self.temperature = 100.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.01
        
        # Optimization history and performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_solution: Optional[Dict[str, Any]] = None
        self.best_score = float('-inf')
        
        # Neural training data collection
        self.training_examples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.max_training_examples = 10000
        
        self.logger = logging.getLogger(__name__)
    
    async def optimize_task_allocation(self, tasks: List[QuantumTask], 
                                     resources: Dict[str, float],
                                     max_iterations: int = 1000) -> Dict[str, Any]:
        """Optimize task allocation using quantum-neural hybrid approach"""
        if not tasks:
            return {"status": "error", "message": "No tasks to optimize"}
        
        self.logger.info(f"Starting quantum-neural optimization for {len(tasks)} tasks")
        
        # Initialize optimization state
        current_solution = await self._generate_initial_solution(tasks, resources)
        current_score = await self._evaluate_solution(current_solution, tasks, resources)
        
        best_solution = current_solution.copy()
        best_score = current_score
        
        # Hybrid optimization loop
        for iteration in range(max_iterations):
            # Neural network prediction for solution improvement
            if len(self.training_examples) > 50:
                neural_enhancement = await self._neural_enhance_solution(current_solution, tasks, resources)
                current_solution = neural_enhancement
            
            # Quantum annealing perturbation
            candidate_solution = await self._quantum_perturb_solution(current_solution, tasks)
            candidate_score = await self._evaluate_solution(candidate_solution, tasks, resources)
            
            # Quantum acceptance criteria
            if await self._quantum_accept(current_score, candidate_score, self.temperature):
                current_solution = candidate_solution
                current_score = candidate_score
                
                # Collect training data for neural network
                await self._collect_training_data(current_solution, current_score, tasks)
                
                if candidate_score > best_score:
                    best_solution = candidate_solution.copy()
                    best_score = candidate_score
                    self.logger.debug(f"New best score: {best_score:.4f} at iteration {iteration}")
            
            # Quantum annealing temperature update
            self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
            
            # Periodically train neural network
            if iteration % 100 == 0 and len(self.training_examples) > 20:
                await self._train_neural_network()
        
        # Final neural network training
        if len(self.training_examples) > 10:
            await self._train_neural_network()
        
        optimization_result = {
            "status": "success",
            "best_solution": best_solution,
            "best_score": best_score,
            "iterations_completed": max_iterations,
            "neural_network_coherence": self.neural_network.network_coherence,
            "training_examples_collected": len(self.training_examples),
            "final_temperature": self.temperature
        }
        
        self.optimization_history.append(optimization_result)
        self.best_solution = best_solution
        self.best_score = best_score
        
        self.logger.info(f"Quantum-neural optimization completed. Best score: {best_score:.4f}")
        
        return optimization_result
    
    async def _generate_initial_solution(self, tasks: List[QuantumTask], 
                                       resources: Dict[str, float]) -> Dict[str, Any]:
        """Generate initial solution using quantum superposition"""
        task_assignments = {}
        resource_allocations = {resource: 0.0 for resource in resources}
        
        for task in tasks:
            # Quantum-weighted random assignment
            priority_weight = task.priority.probability_weight
            complexity_factor = min(1.0, task.complexity_factor / 10.0)
            
            # Quantum superposition of resource allocation
            quantum_weights = np.random.dirichlet([priority_weight, complexity_factor, 
                                                  task.quantum_coherence, 0.5])
            
            task_assignment = {
                "task_id": task.task_id,
                "priority_allocation": float(quantum_weights[0]),
                "complexity_allocation": float(quantum_weights[1]),
                "coherence_allocation": float(quantum_weights[2]),
                "quantum_allocation": float(quantum_weights[3]),
                "total_score": float(np.sum(quantum_weights))
            }
            
            task_assignments[task.task_id] = task_assignment
            
            # Update resource usage
            for resource_type in resources:
                if resource_type in ["cpu", "memory"]:
                    usage = complexity_factor * quantum_weights[1] * resources[resource_type] * 0.1
                    resource_allocations[resource_type] += usage
        
        return {
            "task_assignments": task_assignments,
            "resource_allocations": resource_allocations,
            "solution_coherence": np.random.uniform(0.5, 1.0)
        }
    
    async def _evaluate_solution(self, solution: Dict[str, Any], 
                               tasks: List[QuantumTask], resources: Dict[str, float]) -> float:
        """Evaluate solution quality using multiple criteria"""
        if not solution or "task_assignments" not in solution:
            return 0.0
        
        task_assignments = solution["task_assignments"]
        resource_allocations = solution.get("resource_allocations", {})
        
        # Task completion score
        completion_score = sum(
            assignment.get("total_score", 0) * 
            next((t.priority.probability_weight for t in tasks if t.task_id == task_id), 0.5)
            for task_id, assignment in task_assignments.items()
        ) / len(tasks)
        
        # Resource efficiency score
        resource_efficiency = 1.0
        for resource_type, allocated in resource_allocations.items():
            if resource_type in resources and resources[resource_type] > 0:
                utilization = allocated / resources[resource_type]
                # Optimal utilization around 0.8 (80%)
                efficiency = 1.0 - abs(utilization - 0.8)
                resource_efficiency *= max(0.1, efficiency)
        
        # Quantum coherence bonus
        coherence_bonus = solution.get("solution_coherence", 0.5) * 0.1
        
        # Overall score with quantum enhancement
        base_score = (completion_score * 0.6 + resource_efficiency * 0.3 + coherence_bonus)
        quantum_enhancement = np.random.normal(1.0, 0.05)  # Small quantum fluctuation
        
        return base_score * quantum_enhancement
    
    async def _neural_enhance_solution(self, solution: Dict[str, Any], 
                                     tasks: List[QuantumTask], 
                                     resources: Dict[str, float]) -> Dict[str, Any]:
        """Use neural network to enhance solution"""
        # Convert solution to neural network input
        solution_features = await self._extract_solution_features(solution, tasks, resources)
        
        # Get neural network prediction
        enhancement_factors = self.neural_network.quantum_predict(solution_features)
        
        # Apply neural enhancement to solution
        enhanced_solution = solution.copy()
        
        if "task_assignments" in enhanced_solution:
            for task_id, assignment in enhanced_solution["task_assignments"].items():
                # Apply neural enhancement
                enhancement_factor = float(enhancement_factors[0])  # Single output network
                for key in ["priority_allocation", "complexity_allocation", "coherence_allocation"]:
                    if key in assignment:
                        assignment[key] *= (1 + enhancement_factor * 0.1)  # 10% max enhancement
                
                # Recalculate total score
                assignment["total_score"] = sum(
                    assignment.get(key, 0) for key in 
                    ["priority_allocation", "complexity_allocation", "coherence_allocation", "quantum_allocation"]
                )
        
        # Update solution coherence
        coherence_enhancement = abs(enhancement_factors[0]) * 0.05
        enhanced_solution["solution_coherence"] = min(1.0, 
            enhanced_solution.get("solution_coherence", 0.5) + coherence_enhancement)
        
        return enhanced_solution
    
    async def _quantum_perturb_solution(self, solution: Dict[str, Any], 
                                      tasks: List[QuantumTask]) -> Dict[str, Any]:
        """Apply quantum perturbation to solution"""
        perturbed_solution = solution.copy()
        
        if "task_assignments" not in perturbed_solution:
            return perturbed_solution
        
        # Quantum perturbation operations
        perturbation_type = np.random.choice(["quantum_swap", "quantum_shift", "quantum_tunnel"], 
                                           p=[0.4, 0.4, 0.2])
        
        task_ids = list(perturbed_solution["task_assignments"].keys())
        
        if perturbation_type == "quantum_swap" and len(task_ids) >= 2:
            # Swap quantum allocations between two tasks
            id1, id2 = np.random.choice(task_ids, 2, replace=False)
            assignment1 = perturbed_solution["task_assignments"][id1]
            assignment2 = perturbed_solution["task_assignments"][id2]
            
            # Quantum entangled swap
            temp = assignment1["quantum_allocation"]
            assignment1["quantum_allocation"] = assignment2["quantum_allocation"]
            assignment2["quantum_allocation"] = temp
        
        elif perturbation_type == "quantum_shift":
            # Apply quantum phase shift to random task
            task_id = np.random.choice(task_ids)
            assignment = perturbed_solution["task_assignments"][task_id]
            
            # Quantum shift in allocation
            shift_amount = np.random.normal(0, 0.05)
            for key in ["priority_allocation", "complexity_allocation", "coherence_allocation"]:
                if key in assignment:
                    assignment[key] = max(0, assignment[key] + shift_amount)
        
        elif perturbation_type == "quantum_tunnel":
            # Quantum tunneling - dramatic allocation change
            task_id = np.random.choice(task_ids)
            assignment = perturbed_solution["task_assignments"][task_id]
            
            # Quantum tunnel to new allocation state
            tunnel_factor = np.random.exponential(0.2)
            assignment["quantum_allocation"] *= (1 + tunnel_factor)
        
        # Renormalize and recalculate scores
        for assignment in perturbed_solution["task_assignments"].values():
            assignment["total_score"] = sum(
                assignment.get(key, 0) for key in 
                ["priority_allocation", "complexity_allocation", "coherence_allocation", "quantum_allocation"]
            )
        
        # Update solution coherence
        quantum_noise = np.random.normal(0, 0.02)
        perturbed_solution["solution_coherence"] = np.clip(
            perturbed_solution.get("solution_coherence", 0.5) + quantum_noise, 0.1, 1.0
        )
        
        return perturbed_solution
    
    async def _quantum_accept(self, current_score: float, candidate_score: float, temperature: float) -> bool:
        """Quantum acceptance criteria with thermal fluctuations"""
        if candidate_score > current_score:
            return True
        
        if temperature <= 0:
            return False
        
        # Quantum Boltzmann acceptance with quantum tunneling
        energy_diff = candidate_score - current_score
        acceptance_prob = np.exp(energy_diff / temperature)
        
        # Add quantum tunneling probability
        tunnel_prob = np.exp(-abs(energy_diff) * 2.0) * 0.1  # 10% max tunneling
        total_prob = acceptance_prob + tunnel_prob
        
        return np.random.random() < total_prob
    
    async def _extract_solution_features(self, solution: Dict[str, Any], 
                                       tasks: List[QuantumTask], 
                                       resources: Dict[str, float]) -> np.ndarray:
        """Extract numerical features from solution for neural network"""
        features = []
        
        # Solution-level features
        features.append(solution.get("solution_coherence", 0.5))
        features.append(len(solution.get("task_assignments", {})) / max(1, len(tasks)))
        
        # Resource allocation features
        resource_allocations = solution.get("resource_allocations", {})
        for resource_type in ["cpu", "memory", "network", "storage"]:
            allocated = resource_allocations.get(resource_type, 0)
            available = resources.get(resource_type, 1)
            features.append(allocated / max(1, available))
        
        # Task assignment statistics
        task_assignments = solution.get("task_assignments", {})
        if task_assignments:
            total_scores = [a.get("total_score", 0) for a in task_assignments.values()]
            features.extend([
                np.mean(total_scores),
                np.std(total_scores) if len(total_scores) > 1 else 0,
                np.max(total_scores) if total_scores else 0,
                np.min(total_scores) if total_scores else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Pad or truncate to fixed size
        target_size = 10
        while len(features) < target_size:
            features.append(0.0)
        
        return np.array(features[:target_size])
    
    async def _collect_training_data(self, solution: Dict[str, Any], score: float, tasks: List[QuantumTask]):
        """Collect training data for neural network"""
        # Extract features as input
        features = await self._extract_solution_features(solution, tasks, {})
        
        # Normalize score as target (0-1 range)
        normalized_score = np.clip(score, 0, 1)
        target = np.array([normalized_score])
        
        # Add to training examples
        self.training_examples.append((features, target))
        
        # Limit training data size
        if len(self.training_examples) > self.max_training_examples:
            # Remove oldest examples
            self.training_examples = self.training_examples[-self.max_training_examples:]
    
    async def _train_neural_network(self, epochs: int = 50):
        """Train neural network on collected data"""
        if len(self.training_examples) < 10:
            return
        
        # Prepare training data
        training_data = self.training_examples.copy()
        
        # Train network asynchronously
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            training_result = await loop.run_in_executor(
                executor, 
                self.neural_network.quantum_train, 
                training_data, epochs, 0.01
            )
        
        self.logger.info(f"Neural network training completed: {training_result['final_loss']:.4f} loss")
    
    def get_optimizer_status(self) -> Dict[str, Any]:
        """Get comprehensive optimizer status"""
        return {
            "neural_network_coherence": self.neural_network.network_coherence,
            "current_temperature": self.temperature,
            "training_examples_count": len(self.training_examples),
            "optimization_runs": len(self.optimization_history),
            "best_score_achieved": self.best_score,
            "neural_training_iterations": self.neural_network.training_iterations,
            "network_layer_count": len(self.neural_network.layers),
            "recent_performance": self.neural_network.performance_history[-10:] if self.neural_network.performance_history else []
        }
