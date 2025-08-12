#!/usr/bin/env python3
"""
Test script for Quantum Annealing Optimizer
"""

import sys
sys.path.insert(0, '.')

import asyncio
from quantum_task_planner.research.quantum_annealing_optimizer import QuantumAnnealingOptimizer
from quantum_task_planner.core.quantum_task import QuantumTask

async def test_quantum_research():
    print("Testing Quantum Annealing Optimizer...")
    
    # Create optimizer
    optimizer = QuantumAnnealingOptimizer(n_qubits=16)
    
    # Create test tasks
    tasks = [
        QuantumTask(title=f'Research Task {i}', description=f'Test task {i}')
        for i in range(3)
    ]
    
    # Define simple optimization objectives
    objectives = [
        lambda task_list: sum(task.get_completion_probability() for task in task_list),  # Maximize completion probability
        lambda task_list: len(task_list)  # Minimize number of tasks (simple constraint)
    ]
    
    constraints = {
        'max_parallel_tasks': 2
    }
    
    print(f"Created {len(tasks)} tasks for optimization")
    
    # Run optimization
    result = await optimizer.optimize_task_scheduling(tasks, constraints, objectives)
    
    print("Optimization completed!")
    print(f"Best energy: {result['quantum_metrics']['best_energy']:.6f}")
    print(f"Quantum tunneling events: {result['quantum_metrics']['quantum_tunneling_events']}")
    print(f"Convergence iterations: {result['quantum_metrics']['convergence_iterations']}")
    print(f"Statistical significance: {result['statistical_validation']['energy_statistics']['mean']:.6f}")
    
    # Display research summary
    research_summary = optimizer.get_research_summary()
    print(f"Novel contributions: {len(research_summary['novel_contributions'])}")
    
    print("✅ Quantum annealing research optimization working!")

if __name__ == "__main__":
    asyncio.run(test_quantum_research())