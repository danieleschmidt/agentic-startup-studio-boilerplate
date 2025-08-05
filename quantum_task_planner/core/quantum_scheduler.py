"""
Quantum Task Scheduler

Implements quantum-inspired scheduling algorithms using superposition,
interference patterns, and quantum annealing optimization techniques.
"""

import asyncio
import heapq
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import logging

from .quantum_task import QuantumTask, TaskState, TaskPriority


@dataclass
class SchedulingConstraint:
    """Represents scheduling constraints with quantum uncertainty"""
    constraint_type: str
    weight: float
    flexibility: float = 0.1  # Quantum uncertainty in constraint
    
    def evaluate(self, task: QuantumTask, context: Dict[str, Any]) -> float:
        """Evaluate constraint satisfaction (0-1)"""
        raise NotImplementedError


@dataclass
class ResourceConstraint(SchedulingConstraint):
    """Resource availability constraint"""
    resource_type: str
    available_amount: float
    
    def evaluate(self, task: QuantumTask, context: Dict[str, Any]) -> float:
        required = sum(
            res.get_expected_requirement() 
            for res in task.resources 
            if res.resource_type == self.resource_type
        )
        if required == 0:
            return 1.0
        satisfaction = min(1.0, self.available_amount / required)
        # Apply quantum uncertainty
        uncertainty = np.random.normal(0, self.flexibility)
        return max(0.0, min(1.0, satisfaction + uncertainty))


@dataclass
class TimeConstraint(SchedulingConstraint):
    """Time-based scheduling constraint"""
    deadline: datetime
    criticality: float = 1.0
    
    def evaluate(self, task: QuantumTask, context: Dict[str, Any]) -> float:
        if not task.due_date:
            return 0.8  # Neutral satisfaction for tasks without deadlines
        
        time_until_deadline = (self.deadline - datetime.utcnow()).total_seconds()
        if time_until_deadline <= 0:
            return 0.0  # Past deadline
        
        # Calculate satisfaction based on time pressure
        if task.estimated_duration:
            time_ratio = time_until_deadline / task.estimated_duration.total_seconds()
            satisfaction = min(1.0, time_ratio / 2.0)  # Prefer 2x buffer time
        else:
            satisfaction = 0.5  # Unknown duration
        
        # Apply criticality and quantum uncertainty
        weighted_satisfaction = satisfaction * self.criticality
        uncertainty = np.random.normal(0, self.flexibility)
        return max(0.0, min(1.0, weighted_satisfaction + uncertainty))


class QuantumTaskScheduler:
    """
    Advanced quantum-inspired task scheduler using superposition-based
    optimization and quantum annealing algorithms.
    """
    
    def __init__(self, max_iterations: int = 1000, temperature_schedule: str = "exponential"):
        self.tasks: Dict[str, QuantumTask] = {}
        self.scheduled_tasks: List[Tuple[datetime, QuantumTask]] = []
        self.constraints: List[SchedulingConstraint] = []
        self.resource_pools: Dict[str, float] = defaultdict(float)
        
        # Quantum annealing parameters
        self.max_iterations = max_iterations
        self.temperature_schedule = temperature_schedule
        self.initial_temperature = 100.0
        self.final_temperature = 0.01
        
        # Performance tracking
        self.scheduling_history: List[Dict[str, Any]] = []
        self.optimization_metrics: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task: QuantumTask):
        """Add a task to the scheduler"""
        self.tasks[task.task_id] = task
        self.logger.info(f"Added task {task.task_id}: {task.title}")
    
    def remove_task(self, task_id: str) -> Optional[QuantumTask]:
        """Remove a task from the scheduler"""
        return self.tasks.pop(task_id, None)
    
    def add_constraint(self, constraint: SchedulingConstraint):
        """Add a scheduling constraint"""
        self.constraints.append(constraint)
    
    def set_resource_pool(self, resource_type: str, amount: float):
        """Set available resource pool"""
        self.resource_pools[resource_type] = amount
    
    async def optimize_schedule(self) -> List[Tuple[datetime, QuantumTask]]:
        """
        Optimize task schedule using quantum-inspired annealing algorithm
        
        Returns:
            Optimized schedule as list of (start_time, task) tuples
        """
        if not self.tasks:
            return []
        
        self.logger.info(f"Starting quantum schedule optimization for {len(self.tasks)} tasks")
        
        # Initialize quantum superposition of all possible schedules
        current_schedule = await self._generate_initial_schedule()
        best_schedule = current_schedule.copy()
        best_energy = await self._calculate_schedule_energy(current_schedule)
        
        # Quantum annealing optimization
        for iteration in range(self.max_iterations):
            temperature = self._get_temperature(iteration)
            
            # Generate quantum perturbation
            candidate_schedule = await self._quantum_perturbation(current_schedule)
            candidate_energy = await self._calculate_schedule_energy(candidate_schedule)
            
            # Quantum acceptance probability
            if await self._quantum_accept(current_schedule, candidate_schedule, 
                                        best_energy, candidate_energy, temperature):
                current_schedule = candidate_schedule
                
                if candidate_energy < best_energy:
                    best_schedule = candidate_schedule.copy()
                    best_energy = candidate_energy
                    self.logger.debug(f"New best energy: {best_energy:.4f} at iteration {iteration}")
            
            # Apply quantum interference effects
            if iteration % 100 == 0:
                await self._apply_quantum_interference(current_schedule)
        
        self.scheduled_tasks = best_schedule
        self._record_optimization_metrics(best_energy, self.max_iterations)
        
        self.logger.info(f"Optimization complete. Final energy: {best_energy:.4f}")
        return best_schedule
    
    async def _generate_initial_schedule(self) -> List[Tuple[datetime, QuantumTask]]:
        """Generate initial schedule using quantum superposition sampling"""
        schedule = []
        current_time = datetime.utcnow()
        
        # Sort tasks by quantum-weighted priority
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: self._calculate_quantum_priority(t),
            reverse=True
        )
        
        for task in sorted_tasks:
            # Apply quantum uncertainty to start time
            uncertainty_minutes = np.random.exponential(30)  # Exponential distribution
            start_time = current_time + timedelta(minutes=uncertainty_minutes)
            
            schedule.append((start_time, task))
            
            # Update current time considering task duration
            if task.estimated_duration:
                current_time = start_time + task.estimated_duration
            else:
                # Estimate duration using quantum probability
                estimated_minutes = np.random.gamma(60, 1)  # Gamma distribution
                current_time = start_time + timedelta(minutes=estimated_minutes)
        
        return schedule
    
    def _calculate_quantum_priority(self, task: QuantumTask) -> float:
        """Calculate quantum-weighted priority score"""
        base_priority = task.priority.probability_weight
        completion_prob = task.get_completion_probability()
        coherence_factor = task.quantum_coherence
        
        # Quantum interference from entangled tasks
        entanglement_boost = len(task.entangled_tasks) * 0.05
        
        return base_priority * completion_prob * coherence_factor + entanglement_boost
    
    async def _calculate_schedule_energy(self, schedule: List[Tuple[datetime, QuantumTask]]) -> float:
        """
        Calculate total energy (cost) of a schedule using quantum mechanics principles
        
        Lower energy indicates better schedule
        """
        total_energy = 0.0
        resource_usage = defaultdict(float)
        time_penalties = 0.0
        
        for start_time, task in schedule:
            # Resource constraint energy
            for resource in task.resources:
                required = resource.get_expected_requirement()
                resource_usage[resource.resource_type] += required
                
                if resource_usage[resource.resource_type] > self.resource_pools[resource.resource_type]:
                    # Penalty for resource overuse
                    overuse = resource_usage[resource.resource_type] - self.resource_pools[resource.resource_type]
                    total_energy += overuse * 10.0  # Heavy penalty
            
            # Time constraint energy
            if task.due_date and start_time > task.due_date:
                delay = (start_time - task.due_date).total_seconds() / 3600  # Hours
                time_penalties += delay ** 2  # Quadratic penalty
            
            # Quantum coherence energy (lower coherence = higher energy)
            coherence_energy = (1.0 - task.quantum_coherence) * 5.0
            total_energy += coherence_energy
            
            # Priority-based energy (lower priority = higher energy cost)
            priority_energy = (1.0 - task.priority.probability_weight) * 3.0
            total_energy += priority_energy
        
        # Constraint satisfaction energy
        for constraint in self.constraints:
            constraint_violations = 0.0
            for _, task in schedule:
                satisfaction = constraint.evaluate(task, {"schedule": schedule})
                constraint_violations += (1.0 - satisfaction) * constraint.weight
            total_energy += constraint_violations
        
        # Entanglement correlation energy
        entanglement_energy = await self._calculate_entanglement_energy(schedule)
        total_energy += entanglement_energy
        
        return total_energy + time_penalties
    
    async def _calculate_entanglement_energy(self, schedule: List[Tuple[datetime, QuantumTask]]) -> float:
        """Calculate energy contribution from quantum entanglement"""
        entanglement_energy = 0.0
        task_positions = {task.task_id: i for i, (_, task) in enumerate(schedule)}
        
        for _, task in schedule:
            for entangled_id in task.entangled_tasks:
                if entangled_id in task_positions:
                    # Calculate separation in schedule
                    separation = abs(task_positions[task.task_id] - task_positions[entangled_id])
                    # Entangled tasks should be scheduled close together
                    entanglement_energy += separation * 0.5
        
        return entanglement_energy
    
    async def _quantum_perturbation(self, schedule: List[Tuple[datetime, QuantumTask]]) -> List[Tuple[datetime, QuantumTask]]:
        """Apply quantum perturbation to generate new schedule candidate"""
        new_schedule = schedule.copy()
        
        # Random quantum operation
        operation = np.random.choice(['swap', 'shift', 'quantum_tunnel'], p=[0.4, 0.4, 0.2])
        
        if operation == 'swap' and len(new_schedule) >= 2:
            # Quantum swap operation
            i, j = np.random.choice(len(new_schedule), 2, replace=False)
            new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
        
        elif operation == 'shift':
            # Quantum time shift
            idx = np.random.randint(len(new_schedule))
            start_time, task = new_schedule[idx]
            # Apply quantum uncertainty to time shift
            shift_minutes = np.random.normal(0, 60)  # Normal distribution
            new_start_time = start_time + timedelta(minutes=shift_minutes)
            new_schedule[idx] = (new_start_time, task)
        
        elif operation == 'quantum_tunnel':
            # Quantum tunneling - tasks can tunnel through barriers
            if len(new_schedule) >= 3:
                # Move a task to a dramatically different position
                idx = np.random.randint(len(new_schedule))
                task_entry = new_schedule.pop(idx)
                new_idx = np.random.randint(len(new_schedule) + 1)
                new_schedule.insert(new_idx, task_entry)
        
        return new_schedule
    
    async def _quantum_accept(self, current: List, candidate: List, 
                            current_energy: float, candidate_energy: float, 
                            temperature: float) -> bool:
        """Quantum acceptance probability using Boltzmann distribution"""
        if candidate_energy < current_energy:
            return True  # Always accept improvements
        
        # Quantum acceptance probability
        energy_diff = candidate_energy - current_energy
        if temperature > 0:
            acceptance_prob = np.exp(-energy_diff / temperature)
            # Add quantum fluctuation
            quantum_noise = np.random.normal(0, 0.01)
            acceptance_prob += quantum_noise
            return np.random.random() < acceptance_prob
        
        return False
    
    def _get_temperature(self, iteration: int) -> float:
        """Calculate temperature for quantum annealing schedule"""
        progress = iteration / self.max_iterations
        
        if self.temperature_schedule == "exponential":
            return self.initial_temperature * np.exp(-5 * progress)
        elif self.temperature_schedule == "linear":
            return self.initial_temperature * (1 - progress)
        elif self.temperature_schedule == "quantum":
            # Quantum-inspired temperature schedule with oscillations
            base_temp = self.initial_temperature * (1 - progress)
            quantum_oscillation = 0.1 * np.sin(10 * np.pi * progress)
            return base_temp + quantum_oscillation
        else:
            return self.initial_temperature * (1 - progress)
    
    async def _apply_quantum_interference(self, schedule: List[Tuple[datetime, QuantumTask]]):
        """Apply quantum interference effects to task states"""
        for _, task in schedule:
            # Interference between different task states
            for state1 in task.state_amplitudes:
                for state2 in task.state_amplitudes:
                    if state1 != state2:
                        amp1 = task.state_amplitudes[state1].amplitude
                        amp2 = task.state_amplitudes[state2].amplitude
                        
                        # Calculate interference phase
                        phase_diff = np.angle(amp1) - np.angle(amp2)
                        interference = 0.01 * np.cos(phase_diff)
                        
                        # Apply interference to amplitudes
                        task.state_amplitudes[state1].amplitude *= (1 + interference)
                        task.state_amplitudes[state2].amplitude *= (1 - interference)
            
            # Renormalize after interference
            task._normalize_amplitudes()
    
    def _record_optimization_metrics(self, final_energy: float, iterations: int):
        """Record optimization performance metrics"""
        metrics = {
            "final_energy": final_energy,
            "iterations": iterations,
            "tasks_scheduled": len(self.scheduled_tasks),
            "average_completion_probability": np.mean([
                task.get_completion_probability() for _, task in self.scheduled_tasks
            ]),
            "quantum_coherence_avg": np.mean([
                task.quantum_coherence for _, task in self.scheduled_tasks
            ]),
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
        
        self.optimization_metrics = metrics
        self.scheduling_history.append(metrics)
    
    async def get_next_tasks(self, count: int = 5) -> List[QuantumTask]:
        """Get next tasks to execute based on quantum measurement"""
        if not self.scheduled_tasks:
            await self.optimize_schedule()
        
        current_time = datetime.utcnow()
        upcoming_tasks = []
        
        for start_time, task in self.scheduled_tasks:
            if len(upcoming_tasks) >= count:
                break
            
            # Quantum measurement to determine if task should be executed
            measured_state = task.measure_state()
            
            if measured_state in [TaskState.PENDING, TaskState.IN_PROGRESS]:
                # Check if it's time to start the task (with quantum uncertainty)
                time_buffer = timedelta(minutes=np.random.exponential(15))
                if current_time >= start_time - time_buffer:
                    upcoming_tasks.append(task)
        
        return upcoming_tasks
    
    def get_schedule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive schedule statistics"""
        if not self.scheduled_tasks:
            return {}
        
        tasks = [task for _, task in self.scheduled_tasks]
        
        return {
            "total_tasks": len(tasks),
            "completion_probability_avg": np.mean([t.get_completion_probability() for t in tasks]),
            "completion_probability_std": np.std([t.get_completion_probability() for t in tasks]),
            "quantum_coherence_avg": np.mean([t.quantum_coherence for t in tasks]),
            "priority_distribution": {
                priority.name: sum(1 for t in tasks if t.priority == priority)
                for priority in TaskPriority
            },
            "entanglement_count": sum(len(t.entangled_tasks) for t in tasks) // 2,
            "estimated_total_duration": sum(
                t.estimated_duration.total_seconds() if t.estimated_duration else 3600
                for t in tasks
            ) / 3600,  # Hours
            "optimization_metrics": self.optimization_metrics
        }