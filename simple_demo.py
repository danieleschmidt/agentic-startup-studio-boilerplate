#!/usr/bin/env python3
"""
Simple Quantum Task Planner Demo

A basic demonstration that works without external dependencies.
"""

import sys
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


class TaskState(Enum):
    """Simple task states"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Task priorities with quantum weights"""
    CRITICAL = ("critical", 0.95)
    HIGH = ("high", 0.80)
    MEDIUM = ("medium", 0.60)
    LOW = ("low", 0.40)
    
    def __init__(self, name: str, probability_weight: float):
        self.probability_weight = probability_weight


@dataclass
class SimpleQuantumTask:
    """Simple quantum-inspired task"""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    created_at: datetime
    state: TaskState = TaskState.PENDING
    quantum_coherence: float = 1.0
    completion_probability: float = 0.8
    entangled_tasks: Set[str] = None
    
    def __post_init__(self):
        if self.entangled_tasks is None:
            self.entangled_tasks = set()
    
    def get_completion_probability(self) -> float:
        """Calculate completion probability"""
        base_prob = self.completion_probability
        priority_boost = self.priority.probability_weight * 0.2
        coherence_bonus = self.quantum_coherence * 0.1
        return min(1.0, base_prob + priority_boost + coherence_bonus)
    
    def measure_state(self) -> TaskState:
        """Quantum measurement simulation"""
        import random
        # Quantum measurement with probability bias
        if random.random() < self.get_completion_probability():
            return TaskState.IN_PROGRESS if self.state == TaskState.PENDING else self.state
        return self.state
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.name,
            "state": self.state.value,
            "quantum_coherence": self.quantum_coherence,
            "completion_probability": self.get_completion_probability(),
            "entangled_tasks": len(self.entangled_tasks),
            "created_at": self.created_at.isoformat()
        }


class SimpleQuantumPlanner:
    """Simple quantum task planner"""
    
    def __init__(self):
        self.tasks: Dict[str, SimpleQuantumTask] = {}
        self.entanglement_bonds: Dict[str, List[str]] = {}
    
    def create_task(self, title: str, description: str, priority: str = "medium") -> SimpleQuantumTask:
        """Create a new quantum task"""
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }
        
        task = SimpleQuantumTask(
            task_id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=priority_map.get(priority, TaskPriority.MEDIUM),
            created_at=datetime.utcnow()
        )
        
        self.tasks[task.task_id] = task
        return task
    
    def entangle_tasks(self, task_ids: List[str], strength: float = 0.8) -> str:
        """Create quantum entanglement between tasks"""
        if len(task_ids) < 2:
            raise ValueError("Need at least 2 tasks for entanglement")
        
        bond_id = str(uuid.uuid4())
        self.entanglement_bonds[bond_id] = task_ids.copy()
        
        # Update task entanglement sets
        for task_id in task_ids:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for other_id in task_ids:
                    if other_id != task_id:
                        task.entangled_tasks.add(other_id)
                
                # Apply entanglement effects
                task.quantum_coherence *= strength
        
        return bond_id
    
    def optimize_schedule(self) -> List[SimpleQuantumTask]:
        """Simple quantum optimization"""
        if not self.tasks:
            return []
        
        # Sort by quantum-weighted priority
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.get_completion_probability() * t.priority.probability_weight,
            reverse=True
        )
        
        return sorted_tasks
    
    def measure_all_tasks(self) -> Dict[str, TaskState]:
        """Perform quantum measurement on all tasks"""
        measurements = {}
        for task_id, task in self.tasks.items():
            measurements[task_id] = task.measure_state()
        return measurements
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        if not self.tasks:
            return {"total_tasks": 0}
        
        total_tasks = len(self.tasks)
        avg_coherence = sum(task.quantum_coherence for task in self.tasks.values()) / total_tasks
        avg_completion_prob = sum(task.get_completion_probability() for task in self.tasks.values()) / total_tasks
        
        state_distribution = {}
        for task in self.tasks.values():
            state = task.state.value
            state_distribution[state] = state_distribution.get(state, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "average_quantum_coherence": avg_coherence,
            "average_completion_probability": avg_completion_prob,
            "active_entanglements": len(self.entanglement_bonds),
            "state_distribution": state_distribution
        }


def demo():
    """Run interactive demo"""
    print("🌌 Quantum Task Planner - Simple Demo")
    print("=" * 50)
    
    planner = SimpleQuantumPlanner()
    
    # Create sample tasks
    task1 = planner.create_task("Build Authentication System", "Implement JWT-based auth", "high")
    task2 = planner.create_task("Design Database Schema", "Create user and task tables", "high")  
    task3 = planner.create_task("Setup CI/CD Pipeline", "Configure GitHub Actions", "medium")
    task4 = planner.create_task("Write Unit Tests", "Add comprehensive test coverage", "medium")
    task5 = planner.create_task("Deploy to Production", "Setup production environment", "critical")
    
    print(f"✨ Created {len(planner.tasks)} quantum tasks")
    
    # Create entanglements
    bond1 = planner.entangle_tasks([task1.task_id, task2.task_id], 0.9)  # Auth depends on DB
    bond2 = planner.entangle_tasks([task3.task_id, task4.task_id], 0.7)  # CI/CD with testing
    bond3 = planner.entangle_tasks([task4.task_id, task5.task_id], 0.8)  # Testing before deploy
    
    print(f"🔗 Created {len(planner.entanglement_bonds)} quantum entanglements")
    
    # Show task list
    print("\\n📋 Quantum Task Universe:")
    print("-" * 80)
    print(f"{'ID':<8} {'Title':<25} {'Priority':<10} {'Coherence':<10} {'Completion':<12} {'State'}")
    print("-" * 80)
    
    for task in planner.tasks.values():
        print(f"{task.task_id[:8]:<8} {task.title[:25]:<25} {task.priority.name:<10} "
              f"{task.quantum_coherence:.3f}     {task.get_completion_probability():.1%}        {task.state.value}")
    
    # Optimize schedule
    print("\\n🚀 Quantum Schedule Optimization:")
    optimized_tasks = planner.optimize_schedule()
    print("-" * 50)
    
    for i, task in enumerate(optimized_tasks, 1):
        print(f"{i}. {task.title} (Priority: {task.priority.name}, "
              f"Completion Prob: {task.get_completion_probability():.1%})")
    
    # Quantum measurements
    print("\\n🔬 Quantum State Measurements:")
    measurements = planner.measure_all_tasks()
    print("-" * 40)
    
    for task_id, measured_state in measurements.items():
        task = planner.tasks[task_id]
        print(f"{task.title[:30]}: {measured_state.value}")
    
    # System statistics
    print("\\n📊 System Statistics:")
    stats = planner.get_statistics()
    print("-" * 30)
    
    for key, value in stats.items():
        if key == "state_distribution":
            print(f"State Distribution:")
            for state, count in value.items():
                print(f"  {state}: {count}")
        else:
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\\n✨ Quantum Task Planner Demo Complete!")
    print("Install full dependencies for advanced features:")
    print("pip install fastapi uvicorn pydantic numpy rich click")


if __name__ == "__main__":
    demo()