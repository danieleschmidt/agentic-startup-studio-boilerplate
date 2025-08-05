"""
Quantum Probability Optimizer

Advanced optimization engine using quantum-inspired algorithms including
quantum genetic algorithms, variational quantum eigensolvers, and
quantum approximate optimization algorithms.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import logging

from .quantum_task import QuantumTask, TaskState, TaskPriority


@dataclass
class OptimizationObjective:
    """Defines optimization objectives with quantum weightings"""
    name: str
    weight: float
    evaluation_function: Callable[[List[QuantumTask]], float]
    quantum_interference: float = 0.0


@dataclass
class QuantumGene:
    """Quantum genetic algorithm gene representation"""
    task_id: str
    schedule_position: float  # 0-1 normalized position
    priority_boost: float    # -1 to 1 priority adjustment
    resource_allocation: Dict[str, float]  # Resource allocation percentages
    quantum_phase: complex   # Quantum phase information


class QuantumProbabilityOptimizer:
    """
    Quantum-inspired optimization engine for task planning that uses
    quantum algorithms to find optimal resource allocation and scheduling.
    """
    
    def __init__(self, population_size: int = 50, generations: int = 100):
        self.population_size = population_size
        self.generations = generations
        self.objectives: List[OptimizationObjective] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Quantum optimization parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.quantum_entanglement_rate = 0.3
        self.decoherence_rate = 0.05
        
        # Variational quantum parameters
        self.vqe_layers = 3
        self.ansatz_parameters: np.ndarray = None
        
        self.logger = logging.getLogger(__name__)
    
    def add_objective(self, objective: OptimizationObjective):
        """Add optimization objective"""
        self.objectives.append(objective)
        self.logger.info(f"Added optimization objective: {objective.name}")
    
    async def optimize_task_allocation(self, tasks: List[QuantumTask], 
                                     resources: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize task allocation using quantum-inspired genetic algorithm
        
        Args:
            tasks: List of tasks to optimize
            resources: Available resource pools
        
        Returns:
            Optimization results with resource allocation and scheduling
        """
        self.logger.info(f"Starting quantum optimization for {len(tasks)} tasks")
        
        # Initialize quantum population
        population = await self._initialize_quantum_population(tasks, resources)
        
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness using quantum superposition
            fitness_scores = await self._evaluate_population_fitness(population, tasks)
            
            # Track best solution
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()
            
            # Apply quantum evolution operators
            population = await self._quantum_evolution_step(
                population, fitness_scores, tasks, resources
            )
            
            # Apply decoherence
            await self._apply_decoherence(population)
            
            if generation % 10 == 0:
                self.logger.debug(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
        
        # Extract optimization results
        results = await self._extract_optimization_results(best_solution, tasks, resources)
        
        self._record_optimization_history(results, best_fitness)
        
        self.logger.info(f"Optimization complete. Best fitness: {best_fitness:.4f}")
        return results
    
    async def _initialize_quantum_population(self, tasks: List[QuantumTask], 
                                           resources: Dict[str, float]) -> List[List[QuantumGene]]:
        """Initialize population with quantum superposition"""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for task in tasks:
                # Create quantum gene with superposition of possibilities
                gene = QuantumGene(
                    task_id=task.task_id,
                    schedule_position=np.random.beta(2, 2),  # Beta distribution for smooth scheduling
                    priority_boost=np.random.normal(0, 0.3),  # Gaussian priority adjustment
                    resource_allocation={
                        res_type: np.random.dirichlet([1] * len(task.resources))[i] 
                        if i < len(task.resources) else 0.0
                        for i, res_type in enumerate(resources.keys())
                    },
                    quantum_phase=np.exp(1j * np.random.uniform(0, 2*np.pi))
                )
                individual.append(gene)
            population.append(individual)
        
        return population
    
    async def _evaluate_population_fitness(self, population: List[List[QuantumGene]], 
                                         tasks: List[QuantumTask]) -> np.ndarray:
        """Evaluate fitness using quantum superposition of objectives"""
        fitness_scores = np.zeros(len(population))
        
        for i, individual in enumerate(population):
            total_fitness = 0.0
            
            # Evaluate each objective
            for objective in self.objectives:
                # Create temporary task configuration
                temp_tasks = await self._apply_gene_configuration(individual, tasks)
                
                # Calculate objective fitness
                objective_score = objective.evaluation_function(temp_tasks)
                
                # Apply quantum interference
                if objective.quantum_interference != 0:
                    interference_phase = sum(gene.quantum_phase for gene in individual)
                    interference_factor = 1.0 + objective.quantum_interference * np.real(interference_phase)
                    objective_score *= interference_factor
                
                total_fitness += objective.weight * objective_score
            
            # Add quantum coherence bonus
            coherence_bonus = self._calculate_quantum_coherence(individual)
            total_fitness *= (1.0 + coherence_bonus * 0.1)
            
            fitness_scores[i] = total_fitness
        
        return fitness_scores
    
    async def _apply_gene_configuration(self, individual: List[QuantumGene], 
                                      tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Apply genetic configuration to tasks"""
        task_dict = {task.task_id: task for task in tasks}
        configured_tasks = []
        
        for gene in individual:
            if gene.task_id in task_dict:
                task = task_dict[gene.task_id]
                
                # Apply priority boost
                if gene.priority_boost > 0.5:
                    new_priority = min(TaskPriority.CRITICAL, 
                                     TaskPriority(task.priority.value[0], 
                                                task.priority.value[1] + gene.priority_boost))
                elif gene.priority_boost < -0.5:
                    new_priority = max(TaskPriority.MINIMAL,
                                     TaskPriority(task.priority.value[0],
                                                task.priority.value[1] + gene.priority_boost))
                else:
                    new_priority = task.priority
                
                # Create modified task copy
                modified_task = QuantumTask(
                    title=task.title,
                    description=task.description,
                    priority=new_priority,
                    estimated_duration=task.estimated_duration,
                    due_date=task.due_date
                )
                
                configured_tasks.append(modified_task)
        
        return configured_tasks
    
    def _calculate_quantum_coherence(self, individual: List[QuantumGene]) -> float:
        """Calculate quantum coherence of gene configuration"""
        phases = [gene.quantum_phase for gene in individual]
        
        # Calculate phase coherence
        avg_phase = np.mean(phases)
        phase_variance = np.var([np.angle(phase) for phase in phases])
        
        # Coherence is higher when phases are aligned
        coherence = 1.0 / (1.0 + phase_variance)
        
        return coherence
    
    async def _quantum_evolution_step(self, population: List[List[QuantumGene]], 
                                    fitness_scores: np.ndarray,
                                    tasks: List[QuantumTask],
                                    resources: Dict[str, float]) -> List[List[QuantumGene]]:
        """Apply quantum evolution operators"""
        new_population = []
        
        # Selection using quantum tournament
        selected_parents = await self._quantum_tournament_selection(population, fitness_scores)
        
        # Generate offspring
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]
            
            # Quantum crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = await self._quantum_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Quantum mutation
            if np.random.random() < self.mutation_rate:
                child1 = await self._quantum_mutation(child1, resources)
            if np.random.random() < self.mutation_rate:
                child2 = await self._quantum_mutation(child2, resources)
            
            new_population.extend([child1, child2])
        
        # Maintain population size
        return new_population[:self.population_size]
    
    async def _quantum_tournament_selection(self, population: List[List[QuantumGene]], 
                                          fitness_scores: np.ndarray) -> List[List[QuantumGene]]:
        """Quantum tournament selection with superposition"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Select tournament participants with quantum probability
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            
            # Apply quantum superposition to selection
            tournament_fitness = fitness_scores[tournament_indices]
            
            # Quantum selection probability (not just max)
            selection_probs = np.exp(tournament_fitness / np.max(tournament_fitness))
            selection_probs /= np.sum(selection_probs)
            
            winner_idx = np.random.choice(tournament_indices, p=selection_probs)
            selected.append(population[winner_idx])
        
        return selected
    
    async def _quantum_crossover(self, parent1: List[QuantumGene], 
                               parent2: List[QuantumGene]) -> Tuple[List[QuantumGene], List[QuantumGene]]:
        """Quantum crossover with entanglement"""
        child1, child2 = [], []
        
        for gene1, gene2 in zip(parent1, parent2):
            # Quantum entanglement probability
            if np.random.random() < self.quantum_entanglement_rate:
                # Entangled crossover - create superposition
                alpha = np.random.uniform(0.3, 0.7)
                
                new_gene1 = QuantumGene(
                    task_id=gene1.task_id,
                    schedule_position=alpha * gene1.schedule_position + (1-alpha) * gene2.schedule_position,
                    priority_boost=alpha * gene1.priority_boost + (1-alpha) * gene2.priority_boost,
                    resource_allocation={
                        res: alpha * gene1.resource_allocation.get(res, 0) + 
                             (1-alpha) * gene2.resource_allocation.get(res, 0)
                        for res in set(gene1.resource_allocation.keys()) | set(gene2.resource_allocation.keys())
                    },
                    quantum_phase=alpha * gene1.quantum_phase + (1-alpha) * gene2.quantum_phase
                )
                
                new_gene2 = QuantumGene(
                    task_id=gene2.task_id,
                    schedule_position=(1-alpha) * gene1.schedule_position + alpha * gene2.schedule_position,
                    priority_boost=(1-alpha) * gene1.priority_boost + alpha * gene2.priority_boost,
                    resource_allocation={
                        res: (1-alpha) * gene1.resource_allocation.get(res, 0) + 
                             alpha * gene2.resource_allocation.get(res, 0)
                        for res in set(gene1.resource_allocation.keys()) | set(gene2.resource_allocation.keys())
                    },
                    quantum_phase=(1-alpha) * gene1.quantum_phase + alpha * gene2.quantum_phase
                )
            else:
                # Classical crossover
                if np.random.random() < 0.5:
                    new_gene1, new_gene2 = gene1, gene2
                else:
                    new_gene1, new_gene2 = gene2, gene1
            
            child1.append(new_gene1)
            child2.append(new_gene2)
        
        return child1, child2
    
    async def _quantum_mutation(self, individual: List[QuantumGene], 
                              resources: Dict[str, float]) -> List[QuantumGene]:
        """Apply quantum mutation operators"""
        mutated = []
        
        for gene in individual:
            mutated_gene = QuantumGene(
                task_id=gene.task_id,
                schedule_position=gene.schedule_position,
                priority_boost=gene.priority_boost,
                resource_allocation=gene.resource_allocation.copy(),
                quantum_phase=gene.quantum_phase
            )
            
            # Quantum position mutation
            if np.random.random() < 0.3:
                mutation_strength = np.random.normal(0, 0.1)
                mutated_gene.schedule_position = np.clip(
                    mutated_gene.schedule_position + mutation_strength, 0, 1
                )
            
            # Priority boost mutation
            if np.random.random() < 0.3:
                mutation_strength = np.random.normal(0, 0.2)
                mutated_gene.priority_boost = np.clip(
                    mutated_gene.priority_boost + mutation_strength, -1, 1
                )
            
            # Resource allocation quantum tunneling
            if np.random.random() < 0.2:
                for res_type in mutated_gene.resource_allocation:
                    tunneling_effect = np.random.normal(0, 0.05)
                    mutated_gene.resource_allocation[res_type] = max(0, 
                        mutated_gene.resource_allocation[res_type] + tunneling_effect
                    )
            
            # Quantum phase mutation
            phase_mutation = np.random.normal(0, np.pi/6)
            current_angle = np.angle(mutated_gene.quantum_phase)
            new_angle = current_angle + phase_mutation
            mutated_gene.quantum_phase = np.exp(1j * new_angle)
            
            mutated.append(mutated_gene)
        
        return mutated
    
    async def _apply_decoherence(self, population: List[List[QuantumGene]]):
        """Apply quantum decoherence to population"""
        for individual in population:
            for gene in individual:
                # Decoherence reduces quantum phase information
                decoherence_factor = 1.0 - self.decoherence_rate
                current_magnitude = abs(gene.quantum_phase)
                current_phase = np.angle(gene.quantum_phase)
                
                # Add noise to phase
                phase_noise = np.random.normal(0, self.decoherence_rate * np.pi)
                new_phase = current_phase + phase_noise
                
                gene.quantum_phase = current_magnitude * decoherence_factor * np.exp(1j * new_phase)
    
    async def _extract_optimization_results(self, best_solution: List[QuantumGene], 
                                          tasks: List[QuantumTask],
                                          resources: Dict[str, float]) -> Dict[str, Any]:
        """Extract actionable results from optimization"""
        task_dict = {task.task_id: task for task in tasks}
        
        optimized_schedule = []
        resource_allocation = {res_type: {} for res_type in resources.keys()}
        
        # Sort genes by schedule position
        sorted_genes = sorted(best_solution, key=lambda g: g.schedule_position)
        
        for gene in sorted_genes:
            if gene.task_id in task_dict:
                task = task_dict[gene.task_id]
                
                optimized_schedule.append({
                    "task_id": gene.task_id,
                    "title": task.title,
                    "schedule_position": gene.schedule_position,
                    "priority_boost": gene.priority_boost,
                    "completion_probability": task.get_completion_probability(),
                    "quantum_coherence": task.quantum_coherence
                })
                
                # Extract resource allocation
                for res_type, allocation in gene.resource_allocation.items():
                    resource_allocation[res_type][gene.task_id] = allocation
        
        return {
            "optimized_schedule": optimized_schedule,
            "resource_allocation": resource_allocation,
            "total_tasks": len(optimized_schedule),
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "quantum_metrics": {
                "average_coherence": np.mean([
                    self._calculate_quantum_coherence([gene]) for gene in best_solution
                ]),
                "phase_distribution": [
                    np.angle(gene.quantum_phase) for gene in best_solution
                ],
                "entanglement_strength": self.quantum_entanglement_rate
            }
        }
    
    def _record_optimization_history(self, results: Dict[str, Any], fitness: float):
        """Record optimization run in history"""
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "fitness_score": fitness,
            "tasks_optimized": results["total_tasks"],
            "generations": self.generations,
            "population_size": self.population_size,
            "quantum_metrics": results["quantum_metrics"]
        }
        
        self.optimization_history.append(history_entry)
    
    def create_standard_objectives(self) -> List[OptimizationObjective]:
        """Create standard optimization objectives"""
        objectives = [
            OptimizationObjective(
                name="completion_probability",
                weight=0.3,
                evaluation_function=lambda tasks: np.mean([t.get_completion_probability() for t in tasks]),
                quantum_interference=0.1
            ),
            OptimizationObjective(
                name="priority_alignment",
                weight=0.25,
                evaluation_function=lambda tasks: np.mean([t.priority.probability_weight for t in tasks]),
                quantum_interference=0.05
            ),
            OptimizationObjective(
                name="quantum_coherence",
                weight=0.2,
                evaluation_function=lambda tasks: np.mean([t.quantum_coherence for t in tasks]),
                quantum_interference=0.15
            ),
            OptimizationObjective(
                name="complexity_efficiency",
                weight=0.15,
                evaluation_function=lambda tasks: 1.0 / np.mean([t.complexity_factor for t in tasks]),
                quantum_interference=0.0
            ),
            OptimizationObjective(
                name="entanglement_optimization",
                weight=0.1,
                evaluation_function=lambda tasks: np.mean([len(t.entangled_tasks) for t in tasks]) / 10.0,
                quantum_interference=0.2
            )
        ]
        
        return objectives