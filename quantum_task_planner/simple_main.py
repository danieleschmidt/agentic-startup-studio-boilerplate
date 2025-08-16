#!/usr/bin/env python3
"""
Simple main entry point for testing core functionality without complex dependencies
"""

from .core.quantum_task import QuantumTask, TaskState
from .core.simple_optimizer import SimpleQuantumOptimizer
from rich.console import Console
from rich.table import Table
import click

console = Console()

@click.group()
def simple_cli():
    """Simple Quantum Task Planner CLI"""
    pass

@simple_cli.command()
@click.option('--task-name', default='Sample Task', help='Name of the task to create')
@click.option('--priority', default=1.0, help='Priority of the task (0.0-1.0)')
def create_task(task_name: str, priority: float):
    """Create a simple quantum task"""
    task = QuantumTask(
        task_id=f"task_{task_name.lower().replace(' ', '_')}",
        title=task_name,
        description=f"A sample quantum task: {task_name}"
    )
    
    console.print(f"[green]✅ Created quantum task: {task.title}[/green]")
    console.print(f"Task ID: {task.task_id}")
    console.print(f"Priority: {task.priority}")
    console.print(f"State: {task.state}")
    
    # Test quantum superposition
    try:
        superposition_prob = task.get_superposition_probability()
        console.print(f"Superposition Probability: {superposition_prob:.3f}")
    except AttributeError:
        console.print("Superposition calculation not available in simplified mode")

@simple_cli.command()
def demo():
    """Run a simple quantum task planning demo"""
    console.print("[bold blue]🌌 Quantum Task Planner Demo[/bold blue]")
    
    # Create multiple tasks
    tasks = [
        QuantumTask(task_id="task_1", title="Data Analysis", description="Analyze customer data"),
        QuantumTask(task_id="task_2", title="Report Generation", description="Generate monthly report"),
        QuantumTask(task_id="task_3", title="Model Training", description="Train ML model")
    ]
    
    # Create optimizer
    optimizer = SimpleQuantumOptimizer()
    
    # Display tasks table
    table = Table(title="Quantum Task Queue")
    table.add_column("Task ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Priority", style="yellow")
    table.add_column("State", style="magenta")
    table.add_column("Superposition", style="blue")
    
    for task in tasks:
        try:
            superposition = task.get_superposition_probability()
        except AttributeError:
            superposition = 0.5  # Default superposition value
        
        table.add_row(
            task.task_id,
            task.title,
            str(task.priority.probability_weight),
            task.state.name,
            f"{superposition:.3f}"
        )
    
    console.print(table)
    
    # Optimize task order
    console.print("\n[bold yellow]⚡ Quantum Optimization[/bold yellow]")
    optimized_order = optimizer.optimize_task_order(tasks)
    
    console.print("Optimized execution order:")
    for i, task in enumerate(optimized_order, 1):
        console.print(f"{i}. {task.title} (Priority: {task.priority.probability_weight:.2f})")

if __name__ == "__main__":
    simple_cli()