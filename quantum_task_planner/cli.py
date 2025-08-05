"""
Quantum Task Planner CLI

Command-line interface for quantum task planning with interactive
quantum state visualization and real-time optimization feedback.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Optional, List
import click
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import numpy as np

from .core.quantum_task import QuantumTask, TaskState, TaskPriority
from .core.quantum_scheduler import QuantumTaskScheduler
from .core.quantum_optimizer import QuantumProbabilityOptimizer
from .core.entanglement_manager import TaskEntanglementManager, EntanglementType
from .api.quantum_api import quantum_api


console = Console()


@click.group()
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx):
    """Quantum Task Planner - Advanced quantum-inspired task management system"""
    ctx.ensure_object(dict)
    ctx.obj['scheduler'] = QuantumTaskScheduler()
    ctx.obj['optimizer'] = QuantumProbabilityOptimizer()
    ctx.obj['entanglement_manager'] = TaskEntanglementManager()


@cli.group()
def task():
    """Task management commands"""
    pass


@task.command()
@click.option('--title', '-t', required=True, help='Task title')
@click.option('--description', '-d', required=True, help='Task description')
@click.option('--priority', '-p', 
              type=click.Choice(['critical', 'high', 'medium', 'low', 'minimal']),
              default='medium', help='Task priority')
@click.option('--duration', '-dur', type=float, help='Estimated duration in hours')
@click.option('--due', type=click.DateTime(), help='Due date (YYYY-MM-DD HH:MM:SS)')
@click.option('--complexity', '-c', type=float, default=1.0, 
              help='Complexity factor (0.1-10.0)')
@click.option('--tags', multiple=True, help='Task tags')
@click.pass_context
def create(ctx, title, description, priority, duration, due, complexity, tags):
    """Create a new quantum task"""
    
    priority_map = {
        'critical': TaskPriority.CRITICAL,
        'high': TaskPriority.HIGH,
        'medium': TaskPriority.MEDIUM,
        'low': TaskPriority.LOW,
        'minimal': TaskPriority.MINIMAL
    }
    
    estimated_duration = timedelta(hours=duration) if duration else None
    
    task = QuantumTask(
        title=title,
        description=description,
        priority=priority_map[priority],
        estimated_duration=estimated_duration,
        due_date=due,
        complexity_factor=complexity,
        tags=list(tags)
    )
    
    scheduler = ctx.obj['scheduler']
    scheduler.add_task(task)
    
    console.print(f"✨ Created quantum task: [bold green]{task.task_id}[/bold green]")
    console.print(f"📊 Completion probability: {task.get_completion_probability():.2%}")
    console.print(f"🔗 Quantum coherence: {task.quantum_coherence:.3f}")


@task.command()
@click.pass_context
def list_tasks(ctx):
    """List all quantum tasks"""
    scheduler = ctx.obj['scheduler']
    
    if not scheduler.tasks:
        console.print("No tasks found. Create some tasks first!")
        return
    
    table = Table(title="🌌 Quantum Task Universe")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="bold white")
    table.add_column("Priority", style="yellow")
    table.add_column("Completion Prob", style="green")
    table.add_column("Coherence", style="blue")
    table.add_column("Entangled", style="magenta")
    table.add_column("State", style="red")
    
    for task in scheduler.tasks.values():
        # Get most likely state
        if task.state_amplitudes:
            most_likely_state = max(
                task.state_amplitudes.items(),
                key=lambda x: x[1].probability
            )[0].value
        else:
            most_likely_state = "unknown"
        
        table.add_row(
            task.task_id[:8],
            task.title[:30],
            task.priority.name,
            f"{task.get_completion_probability():.1%}",
            f"{task.quantum_coherence:.3f}",
            str(len(task.entangled_tasks)),
            most_likely_state
        )
    
    console.print(table)


@task.command()
@click.argument('task_id')
@click.option('--observer-effect', '-o', type=float, default=0.1,
              help='Observer effect strength (0.0-1.0)')
@click.pass_context
def measure(ctx, task_id, observer_effect):
    """Perform quantum measurement on a task"""
    scheduler = ctx.obj['scheduler']
    
    # Find task by partial ID match
    matching_tasks = [
        task for task in scheduler.tasks.values()
        if task.task_id.startswith(task_id)
    ]
    
    if not matching_tasks:
        console.print(f"❌ Task not found: {task_id}")
        return
    
    if len(matching_tasks) > 1:
        console.print(f"⚠️  Multiple tasks match {task_id}:")
        for task in matching_tasks:
            console.print(f"  - {task.task_id[:8]}: {task.title}")
        return
    
    task = matching_tasks[0]
    
    with console.status("[bold blue]Performing quantum measurement..."):
        measured_state = task.measure_state(observer_effect)
    
    console.print(f"🔬 Quantum measurement complete!")
    console.print(f"📊 Measured state: [bold green]{measured_state.value}[/bold green]")
    console.print(f"🌊 Observer effect: {observer_effect}")
    console.print(f"⚛️  Remaining coherence: {task.quantum_coherence:.3f}")
    
    # Show state probabilities
    if task.state_amplitudes:
        table = Table(title="Quantum State Probabilities")
        table.add_column("State", style="cyan")
        table.add_column("Probability", style="green")
        table.add_column("Amplitude", style="blue")
        
        for state, amplitude_obj in task.state_amplitudes.items():
            table.add_row(
                state.value,
                f"{amplitude_obj.probability:.3f}",
                f"{abs(amplitude_obj.amplitude):.3f}∠{np.angle(amplitude_obj.amplitude):.2f}"
            )
        
        console.print(table)


@cli.group()
def quantum():
    """Quantum operations and entanglement management"""
    pass


@quantum.command()
@click.argument('task_ids', nargs=-1, required=True)
@click.option('--type', '-t', 'entanglement_type',
              type=click.Choice(['bell_state', 'ghz_state', 'cluster_state', 
                               'dependency', 'resource_shared', 'temporal']),
              default='bell_state', help='Entanglement type')
@click.option('--strength', '-s', type=float, default=0.8,
              help='Entanglement strength (0.0-1.0)')
@click.pass_context
def entangle(ctx, task_ids, entanglement_type, strength):
    """Create quantum entanglement between tasks"""
    scheduler = ctx.obj['scheduler']
    entanglement_manager = ctx.obj['entanglement_manager']
    
    # Resolve task IDs
    tasks = []
    for task_id in task_ids:
        matching_tasks = [
            task for task in scheduler.tasks.values()
            if task.task_id.startswith(task_id)
        ]
        
        if not matching_tasks:
            console.print(f"❌ Task not found: {task_id}")
            return
        
        if len(matching_tasks) > 1:
            console.print(f"⚠️  Multiple tasks match {task_id}")
            return
        
        tasks.append(matching_tasks[0])
    
    if len(tasks) < 2:
        console.print("❌ Need at least 2 tasks for entanglement")
        return
    
    async def create_entanglement():
        entanglement_type_enum = EntanglementType(entanglement_type)
        bond_id = await entanglement_manager.create_entanglement(
            tasks, entanglement_type_enum, strength
        )
        return bond_id
    
    with console.status("[bold blue]Creating quantum entanglement..."):
        bond_id = asyncio.run(create_entanglement())
    
    console.print(f"🔗 Quantum entanglement created!")
    console.print(f"🆔 Bond ID: [bold green]{bond_id}[/bold green]")
    console.print(f"🌀 Type: {entanglement_type}")
    console.print(f"💪 Strength: {strength}")
    console.print(f"🎯 Entangled tasks: {len(tasks)}")
    
    for task in tasks:
        console.print(f"  - {task.task_id[:8]}: {task.title}")


@quantum.command()
@click.pass_context
def entanglement_stats(ctx):
    """Show quantum entanglement statistics"""
    entanglement_manager = ctx.obj['entanglement_manager']
    stats = entanglement_manager.get_entanglement_statistics()
    
    if stats['active_bonds'] == 0:
        console.print("No active quantum entanglements")
        return
    
    console.print(Panel.fit(
        f"🌌 Quantum Entanglement Network Statistics\n\n"
        f"Active Bonds: {stats['active_bonds']}\n"
        f"Entangled Tasks: {stats['total_entangled_tasks']}\n"
        f"Average Strength: {stats['average_strength']:.3f}\n"
        f"Quantum Channels: {stats['quantum_channels']}\n"
        f"Total Events: {stats['total_events']}",
        title="Entanglement Network"
    ))
    
    if stats['entanglement_types']:
        table = Table(title="Entanglement Type Distribution")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="green")
        
        for ent_type, count in stats['entanglement_types'].items():
            table.add_row(ent_type, str(count))
        
        console.print(table)


@cli.group()
def schedule():
    """Quantum scheduling operations"""
    pass


@schedule.command()
@click.option('--iterations', '-i', type=int, default=1000,
              help='Optimization iterations')
@click.option('--show-progress/--no-progress', default=True,
              help='Show optimization progress')
@click.pass_context
def optimize(ctx, iterations, show_progress):
    """Optimize task schedule using quantum annealing"""
    scheduler = ctx.obj['scheduler']
    scheduler.max_iterations = iterations
    
    if not scheduler.tasks:
        console.print("❌ No tasks to optimize. Create some tasks first!")
        return
    
    async def run_optimization():
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task("🌀 Quantum annealing optimization...", total=None)
                optimized_schedule = await scheduler.optimize_schedule()
                progress.update(task, description="✅ Optimization complete!")
        else:
            optimized_schedule = await scheduler.optimize_schedule()
        
        return optimized_schedule
    
    optimized_schedule = asyncio.run(run_optimization())
    
    console.print(f"✨ Quantum optimization complete!")
    console.print(f"📊 Optimized {len(optimized_schedule)} tasks")
    
    if scheduler.optimization_metrics:
        metrics = scheduler.optimization_metrics
        console.print(f"⚡ Final energy: {metrics['final_energy']:.4f}")
        console.print(f"🎯 Average completion probability: {metrics['average_completion_probability']:.2%}")
        console.print(f"🌊 Average quantum coherence: {metrics['quantum_coherence_avg']:.3f}")
    
    # Show optimized schedule
    table = Table(title="🚀 Optimized Quantum Schedule")
    table.add_column("Start Time", style="cyan")
    table.add_column("Task", style="bold white")
    table.add_column("Priority", style="yellow")
    table.add_column("Completion Prob", style="green")
    
    for start_time, task in optimized_schedule[:10]:  # Show first 10
        table.add_row(
            start_time.strftime("%Y-%m-%d %H:%M"),
            task.title[:30],
            task.priority.name,
            f"{task.get_completion_probability():.1%}"
        )
    
    console.print(table)
    
    if len(optimized_schedule) > 10:
        console.print(f"... and {len(optimized_schedule) - 10} more tasks")


@schedule.command()
@click.option('--count', '-c', type=int, default=5,
              help='Number of next tasks to show')
@click.pass_context
def next_tasks(ctx, count):
    """Get next tasks to execute based on quantum measurement"""
    scheduler = ctx.obj['scheduler']
    
    async def get_next():
        return await scheduler.get_next_tasks(count)
    
    with console.status("[bold blue]Measuring quantum states..."):
        next_tasks = asyncio.run(get_next())
    
    if not next_tasks:
        console.print("🎯 No tasks ready for execution")
        return
    
    console.print(f"🚀 Next {len(next_tasks)} tasks to execute:")
    
    table = Table()
    table.add_column("Task", style="bold white")
    table.add_column("Priority", style="yellow")
    table.add_column("Completion Prob", style="green")
    table.add_column("Coherence", style="blue")
    
    for task in next_tasks:
        table.add_row(
            task.title[:40],
            task.priority.name,
            f"{task.get_completion_probability():.1%}",
            f"{task.quantum_coherence:.3f}"
        )
    
    console.print(table)


@cli.command()
@click.option('--host', default='127.0.0.1', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--reload/--no-reload', default=False, help='Auto-reload on changes')
def serve(host, port, reload):
    """Start the Quantum Task Planner API server"""
    console.print(f"🚀 Starting Quantum Task Planner API on {host}:{port}")
    console.print(f"📚 API docs available at: http://{host}:{port}/docs")
    console.print(f"🔬 Quantum state monitor at: http://{host}:{port}/api/v1/quantum/state")
    
    quantum_api.run(host=host, port=port, reload=reload)


@cli.command()
@click.pass_context
def dashboard(ctx):
    """Interactive quantum task dashboard"""
    scheduler = ctx.obj['scheduler']
    entanglement_manager = ctx.obj['entanglement_manager']
    
    def make_layout():
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="tasks"),
            Layout(name="quantum")
        )
        
        # Header
        layout["header"].update(
            Panel("🌌 Quantum Task Planner Dashboard", style="bold blue")
        )
        
        # Tasks table
        if scheduler.tasks:
            tasks_table = Table(title="Active Tasks")
            tasks_table.add_column("ID", style="cyan")
            tasks_table.add_column("Title", style="white")
            tasks_table.add_column("State", style="green")
            tasks_table.add_column("Coherence", style="blue")
            
            for task in list(scheduler.tasks.values())[:10]:
                most_likely_state = "unknown"
                if task.state_amplitudes:
                    most_likely_state = max(
                        task.state_amplitudes.items(),
                        key=lambda x: x[1].probability
                    )[0].value
                
                tasks_table.add_row(
                    task.task_id[:8],
                    task.title[:20],
                    most_likely_state,
                    f"{task.quantum_coherence:.3f}"
                )
            
            layout["tasks"].update(Panel(tasks_table, title="Tasks"))
        else:
            layout["tasks"].update(Panel("No tasks found", title="Tasks"))
        
        # Quantum stats
        stats = entanglement_manager.get_entanglement_statistics()
        quantum_info = (
            f"Active Entanglements: {stats['active_bonds']}\n"
            f"Entangled Tasks: {stats['total_entangled_tasks']}\n"
            f"Quantum Channels: {stats['quantum_channels']}\n"
            f"Average Strength: {stats.get('average_strength', 0):.3f}"
        )
        
        layout["quantum"].update(
            Panel(quantum_info, title="Quantum Network")
        )
        
        # Footer
        layout["footer"].update(
            Panel(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Tasks: {len(scheduler.tasks)} | "
                  f"Entanglements: {stats['active_bonds']}")
        )
        
        return layout
    
    try:
        with Live(make_layout(), refresh_per_second=1) as live:
            while True:
                asyncio.sleep(1)
                live.update(make_layout())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye from the quantum realm!")


if __name__ == '__main__':
    cli()