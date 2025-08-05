"""
Task Entanglement Manager

Manages quantum entanglement relationships between tasks, implementing
non-local correlations, Bell state preparations, and entanglement-based
dependency resolution.
"""

import asyncio
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from enum import Enum
import logging

from .quantum_task import QuantumTask, TaskState


class EntanglementType(Enum):
    """Types of quantum entanglement between tasks"""
    BELL_STATE = "bell_state"           # Strong correlation
    GHZ_STATE = "ghz_state"            # Multi-task entanglement  
    CLUSTER_STATE = "cluster_state"     # Network entanglement
    DEPENDENCY = "dependency"           # Causal entanglement
    RESOURCE_SHARED = "resource_shared" # Resource-based entanglement
    TEMPORAL = "temporal"               # Time-based entanglement


@dataclass
class EntanglementBond:
    """Represents quantum entanglement between tasks"""
    task_ids: Set[str]
    entanglement_type: EntanglementType
    strength: float  # 0-1, strength of entanglement
    created_at: datetime
    last_measured: Optional[datetime] = None
    correlation_matrix: Optional[np.ndarray] = None
    decoherence_rate: float = 0.01
    
    def __post_init__(self):
        if self.correlation_matrix is None:
            n = len(self.task_ids)
            # Initialize with maximally entangled state
            self.correlation_matrix = np.ones((n, n)) * self.strength
            np.fill_diagonal(self.correlation_matrix, 1.0)


@dataclass  
class QuantumChannel:
    """Communication channel between entangled tasks"""
    source_task_id: str
    target_task_id: str
    channel_capacity: float  # bits per second
    noise_level: float = 0.1
    fidelity: float = 0.99
    last_transmission: Optional[datetime] = None


class TaskEntanglementManager:
    """
    Manages quantum entanglement networks between tasks, implementing
    sophisticated quantum correlation effects and non-local dependencies.
    """
    
    def __init__(self, decoherence_time: float = 3600.0):
        self.entanglement_bonds: Dict[str, EntanglementBond] = {}
        self.task_entanglement_map: Dict[str, Set[str]] = defaultdict(set)
        self.quantum_channels: Dict[Tuple[str, str], QuantumChannel] = {}
        self.entanglement_history: List[Dict[str, Any]] = []
        
        self.decoherence_time = decoherence_time  # seconds
        self.max_entanglement_distance = 3  # Maximum separation for entanglement
        
        self.logger = logging.getLogger(__name__)
    
    async def create_entanglement(self, tasks: List[QuantumTask], 
                                entanglement_type: EntanglementType = EntanglementType.BELL_STATE,
                                strength: float = 0.8) -> str:
        """
        Create quantum entanglement between multiple tasks
        
        Args:
            tasks: List of tasks to entangle
            entanglement_type: Type of entanglement to create
            strength: Strength of entanglement (0-1)
        
        Returns:
            Entanglement bond ID
        """
        if len(tasks) < 2:
            raise ValueError("At least 2 tasks required for entanglement")
        
        task_ids = {task.task_id for task in tasks}
        bond_id = f"bond_{datetime.utcnow().timestamp()}"
        
        # Create entanglement bond
        bond = EntanglementBond(
            task_ids=task_ids,
            entanglement_type=entanglement_type,
            strength=strength,
            created_at=datetime.utcnow()
        )
        
        self.entanglement_bonds[bond_id] = bond
        
        # Update task entanglement maps
        for task_id in task_ids:
            self.task_entanglement_map[task_id].add(bond_id)
        
        # Apply entanglement effects to tasks
        await self._apply_entanglement_effects(tasks, bond)
        
        # Create quantum channels between tasks
        await self._establish_quantum_channels(tasks)
        
        # Record entanglement creation
        self._record_entanglement_event("creation", bond_id, tasks)
        
        self.logger.info(f"Created {entanglement_type.value} entanglement {bond_id} "
                        f"between {len(tasks)} tasks with strength {strength}")
        
        return bond_id
    
    async def _apply_entanglement_effects(self, tasks: List[QuantumTask], bond: EntanglementBond):
        """Apply quantum entanglement effects to task states"""
        
        if bond.entanglement_type == EntanglementType.BELL_STATE:
            await self._create_bell_state(tasks, bond.strength)
        elif bond.entanglement_type == EntanglementType.GHZ_STATE:
            await self._create_ghz_state(tasks, bond.strength)
        elif bond.entanglement_type == EntanglementType.CLUSTER_STATE:
            await self._create_cluster_state(tasks, bond.strength)
        elif bond.entanglement_type == EntanglementType.DEPENDENCY:
            await self._create_dependency_entanglement(tasks, bond.strength)
        elif bond.entanglement_type == EntanglementType.RESOURCE_SHARED:
            await self._create_resource_entanglement(tasks, bond.strength)
        elif bond.entanglement_type == EntanglementType.TEMPORAL:
            await self._create_temporal_entanglement(tasks, bond.strength)
    
    async def _create_bell_state(self, tasks: List[QuantumTask], strength: float):
        """Create Bell state entanglement between two tasks"""
        if len(tasks) != 2:
            raise ValueError("Bell state requires exactly 2 tasks")
        
        task1, task2 = tasks
        
        # Create maximally entangled Bell state
        for state in task1.state_amplitudes:
            if state in task2.state_amplitudes:
                # Bell state: |00⟩ + |11⟩ (correlated states)
                correlation_factor = strength / np.sqrt(2)
                
                # Synchronize amplitudes
                combined_amplitude = (
                    task1.state_amplitudes[state].amplitude + 
                    task2.state_amplitudes[state].amplitude
                ) * correlation_factor
                
                task1.state_amplitudes[state].amplitude = combined_amplitude
                task2.state_amplitudes[state].amplitude = combined_amplitude
        
        # Normalize states
        task1._normalize_amplitudes()
        task2._normalize_amplitudes()
        
        # Update task entanglement sets
        task1.entangled_tasks.add(task2.task_id)
        task2.entangled_tasks.add(task1.task_id)
    
    async def _create_ghz_state(self, tasks: List[QuantumTask], strength: float):
        """Create GHZ (Greenberger-Horne-Zeilinger) state for multiple tasks"""
        if len(tasks) < 3:
            raise ValueError("GHZ state requires at least 3 tasks")
        
        # GHZ state: |000...⟩ + |111...⟩ (all tasks correlated)
        correlation_factor = strength / np.sqrt(2)
        
        # Find common states across all tasks
        common_states = set(tasks[0].state_amplitudes.keys())
        for task in tasks[1:]:
            common_states &= set(task.state_amplitudes.keys())
        
        for state in common_states:
            # Average amplitude across all tasks
            avg_amplitude = np.mean([
                task.state_amplitudes[state].amplitude for task in tasks
            ]) * correlation_factor
            
            # Set same amplitude for all tasks (GHZ correlation)
            for task in tasks:
                task.state_amplitudes[state].amplitude = avg_amplitude
        
        # Normalize all tasks
        for task in tasks:
            task._normalize_amplitudes()
        
        # Update entanglement relationships
        task_ids = [task.task_id for task in tasks]
        for i, task in enumerate(tasks):
            for j, other_task in enumerate(tasks):
                if i != j:
                    task.entangled_tasks.add(other_task.task_id)
    
    async def _create_cluster_state(self, tasks: List[QuantumTask], strength: float):
        """Create cluster state entanglement network"""
        # Cluster state: each task entangled with nearest neighbors
        for i, task in enumerate(tasks):
            neighbors = []
            
            # Connect to adjacent tasks (ring topology)
            if i > 0:
                neighbors.append(tasks[i-1])
            if i < len(tasks) - 1:
                neighbors.append(tasks[i+1])
            
            # Apply cluster state entanglement with neighbors
            for neighbor in neighbors:
                await self._apply_controlled_phase_gate(task, neighbor, strength)
    
    async def _apply_controlled_phase_gate(self, control_task: QuantumTask, 
                                         target_task: QuantumTask, strength: float):
        """Apply controlled phase gate between two tasks"""
        phase_shift = strength * np.pi / 2
        
        for state in control_task.state_amplitudes:
            if state in target_task.state_amplitudes:
                # Apply conditional phase shift
                if control_task.state_amplitudes[state].probability > 0.5:
                    current_amplitude = target_task.state_amplitudes[state].amplitude
                    new_amplitude = current_amplitude * np.exp(1j * phase_shift)
                    target_task.state_amplitudes[state].amplitude = new_amplitude
        
        target_task._normalize_amplitudes()
        
        # Update entanglement
        control_task.entangled_tasks.add(target_task.task_id)
        target_task.entangled_tasks.add(control_task.task_id)
    
    async def _create_dependency_entanglement(self, tasks: List[QuantumTask], strength: float):
        """Create dependency-based entanglement (causal relationships)"""
        # First task must complete before others can progress
        primary_task = tasks[0]
        dependent_tasks = tasks[1:]
        
        for dependent_task in dependent_tasks:
            # Create dependency correlation
            if TaskState.COMPLETED in primary_task.state_amplitudes:
                primary_completion_prob = primary_task.state_amplitudes[TaskState.COMPLETED].probability
                
                # Dependent task can only start when primary is likely to complete
                for state in [TaskState.PENDING, TaskState.IN_PROGRESS]:
                    if state in dependent_task.state_amplitudes:
                        dependent_task.state_amplitudes[state].amplitude *= (
                            np.sqrt(primary_completion_prob * strength)
                        )
            
            dependent_task._normalize_amplitudes()
            
            # Add dependency relationship
            dependent_task.dependencies.add(primary_task.task_id)
            primary_task.entangled_tasks.add(dependent_task.task_id)
            dependent_task.entangled_tasks.add(primary_task.task_id)
    
    async def _create_resource_entanglement(self, tasks: List[QuantumTask], strength: float):
        """Create resource-sharing entanglement"""
        # Tasks sharing resources become entangled
        shared_resources = set()
        
        # Find shared resource types
        for task in tasks:
            for resource in task.resources:
                shared_resources.add(resource.resource_type)
        
        # Apply resource-based correlation
        for resource_type in shared_resources:
            resource_tasks = [
                task for task in tasks 
                if any(res.resource_type == resource_type for res in task.resources)
            ]
            
            if len(resource_tasks) > 1:
                # Resource constraint creates entanglement
                total_resource_demand = sum(
                    res.get_expected_requirement()
                    for task in resource_tasks
                    for res in task.resources
                    if res.resource_type == resource_type
                )
                
                # Normalize resource allocation probabilities
                for task in resource_tasks:
                    task_demand = sum(
                        res.get_expected_requirement()
                        for res in task.resources
                        if res.resource_type == resource_type
                    )
                    
                    allocation_factor = (task_demand / total_resource_demand) * strength
                    
                    # Apply resource constraint to state probabilities
                    for state in task.state_amplitudes:
                        task.state_amplitudes[state].amplitude *= np.sqrt(allocation_factor)
                    
                    task._normalize_amplitudes()
                    
                    # Update entanglement relationships
                    for other_task in resource_tasks:
                        if other_task.task_id != task.task_id:
                            task.entangled_tasks.add(other_task.task_id)
    
    async def _create_temporal_entanglement(self, tasks: List[QuantumTask], strength: float):
        """Create temporal entanglement based on timing relationships"""
        # Sort tasks by due date or creation time
        sorted_tasks = sorted(tasks, key=lambda t: t.due_date or t.created_at)
        
        for i, task in enumerate(sorted_tasks):
            # Temporal correlation with nearby tasks
            temporal_neighbors = []
            
            # Connect to temporally adjacent tasks
            if i > 0:
                temporal_neighbors.append(sorted_tasks[i-1])
            if i < len(sorted_tasks) - 1:
                temporal_neighbors.append(sorted_tasks[i+1])
            
            for neighbor in temporal_neighbors:
                # Time-based correlation strength
                if task.due_date and neighbor.due_date:
                    time_diff = abs((task.due_date - neighbor.due_date).total_seconds())
                    temporal_correlation = strength * np.exp(-time_diff / 86400)  # Decay over days
                else:
                    temporal_correlation = strength * 0.5
                
                # Apply temporal phase correlation
                phase_shift = temporal_correlation * np.pi / 4
                
                for state in task.state_amplitudes:
                    if state in neighbor.state_amplitudes:
                        task.state_amplitudes[state].amplitude *= np.exp(1j * phase_shift)
                        neighbor.state_amplitudes[state].amplitude *= np.exp(-1j * phase_shift)
                
                task._normalize_amplitudes()
                neighbor._normalize_amplitudes()
                
                # Update entanglement
                task.entangled_tasks.add(neighbor.task_id)
                neighbor.entangled_tasks.add(task.task_id)
    
    async def _establish_quantum_channels(self, tasks: List[QuantumTask]):
        """Establish quantum communication channels between entangled tasks"""
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks):
                if i < j:  # Avoid duplicate channels
                    channel = QuantumChannel(
                        source_task_id=task1.task_id,
                        target_task_id=task2.task_id,
                        channel_capacity=100.0,  # Default capacity
                        noise_level=0.05,
                        fidelity=0.98
                    )
                    
                    self.quantum_channels[(task1.task_id, task2.task_id)] = channel
                    self.quantum_channels[(task2.task_id, task1.task_id)] = channel  # Bidirectional
    
    async def measure_entanglement(self, bond_id: str, observer_effect: float = 0.1) -> Dict[str, Any]:
        """
        Perform quantum measurement on entangled tasks
        
        Args:
            bond_id: ID of entanglement bond to measure
            observer_effect: Measurement disturbance (0-1)
        
        Returns:
            Measurement results for all entangled tasks
        """
        if bond_id not in self.entanglement_bonds:
            raise ValueError(f"Entanglement bond {bond_id} not found")
        
        bond = self.entanglement_bonds[bond_id]
        measurement_results = {}
        
        # Perform correlated measurements
        measurement_time = datetime.utcnow()
        
        # Apply measurement to all entangled tasks
        for task_id in bond.task_ids:
            # Note: In a real implementation, you'd need to fetch the actual task
            # For now, we'll record the measurement intent
            measurement_results[task_id] = {
                "measurement_time": measurement_time.isoformat(),
                "entanglement_strength": bond.strength,
                "correlation_preserved": bond.strength > observer_effect
            }
        
        # Apply decoherence due to measurement
        bond.strength *= (1.0 - observer_effect)
        bond.last_measured = measurement_time
        
        # Record measurement event
        self._record_entanglement_event("measurement", bond_id, measurement_results)
        
        self.logger.info(f"Measured entanglement bond {bond_id} with observer effect {observer_effect}")
        
        return measurement_results
    
    async def break_entanglement(self, bond_id: str) -> bool:
        """Break quantum entanglement bond"""
        if bond_id not in self.entanglement_bonds:
            return False
        
        bond = self.entanglement_bonds[bond_id]
        
        # Remove from task entanglement maps
        for task_id in bond.task_ids:
            self.task_entanglement_map[task_id].discard(bond_id)
        
        # Remove quantum channels
        task_list = list(bond.task_ids)
        for i in range(len(task_list)):
            for j in range(i + 1, len(task_list)):
                channel_key = (task_list[i], task_list[j])
                self.quantum_channels.pop(channel_key, None)
                self.quantum_channels.pop((task_list[j], task_list[i]), None)
        
        # Remove bond
        del self.entanglement_bonds[bond_id]
        
        # Record breaking event
        self._record_entanglement_event("break", bond_id, list(bond.task_ids))
        
        self.logger.info(f"Broke entanglement bond {bond_id}")
        return True
    
    async def apply_decoherence(self, time_elapsed: float):
        """Apply decoherence to all entanglement bonds"""
        bonds_to_remove = []
        
        for bond_id, bond in self.entanglement_bonds.items():
            # Apply exponential decoherence
            decoherence_factor = np.exp(-time_elapsed * bond.decoherence_rate)
            bond.strength *= decoherence_factor
            
            # Remove bonds that have decohered too much
            if bond.strength < 0.1:
                bonds_to_remove.append(bond_id)
        
        # Remove highly decohered bonds
        for bond_id in bonds_to_remove:
            await self.break_entanglement(bond_id)
        
        self.logger.debug(f"Applied decoherence, removed {len(bonds_to_remove)} bonds")
    
    def get_entangled_tasks(self, task_id: str) -> Set[str]:
        """Get all tasks entangled with given task"""
        entangled_tasks = set()
        
        for bond_id in self.task_entanglement_map.get(task_id, set()):
            if bond_id in self.entanglement_bonds:
                bond = self.entanglement_bonds[bond_id]
                entangled_tasks.update(bond.task_ids - {task_id})
        
        return entangled_tasks
    
    def get_entanglement_strength(self, task_id1: str, task_id2: str) -> float:
        """Get entanglement strength between two specific tasks"""
        max_strength = 0.0
        
        bonds1 = self.task_entanglement_map.get(task_id1, set())
        bonds2 = self.task_entanglement_map.get(task_id2, set())
        
        # Find common bonds
        common_bonds = bonds1 & bonds2
        
        for bond_id in common_bonds:
            bond = self.entanglement_bonds.get(bond_id)
            if bond:
                max_strength = max(max_strength, bond.strength)
        
        return max_strength
    
    def _record_entanglement_event(self, event_type: str, bond_id: str, data: Any):
        """Record entanglement event in history"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "bond_id": bond_id,
            "data": data
        }
        
        self.entanglement_history.append(event)
    
    def get_entanglement_statistics(self) -> Dict[str, Any]:
        """Get comprehensive entanglement statistics"""
        active_bonds = len(self.entanglement_bonds)
        total_entangled_tasks = len(self.task_entanglement_map)
        
        if active_bonds == 0:
            return {
                "active_bonds": 0,
                "total_entangled_tasks": 0,
                "average_strength": 0.0,
                "entanglement_types": {}
            }
        
        strengths = [bond.strength for bond in self.entanglement_bonds.values()]
        type_counts = defaultdict(int)
        
        for bond in self.entanglement_bonds.values():
            type_counts[bond.entanglement_type.value] += 1
        
        return {
            "active_bonds": active_bonds,
            "total_entangled_tasks": total_entangled_tasks,
            "average_strength": np.mean(strengths),
            "strength_std": np.std(strengths),
            "max_strength": np.max(strengths),
            "min_strength": np.min(strengths),
            "entanglement_types": dict(type_counts),
            "quantum_channels": len(self.quantum_channels),
            "total_events": len(self.entanglement_history)
        }