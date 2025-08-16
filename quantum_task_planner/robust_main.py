#!/usr/bin/env python3
"""
Robust main entry point with enhanced error handling and validation
"""

from .core.robust_quantum_task import RobustQuantumTask, RobustTaskManager
from .core.quantum_task import TaskState, TaskPriority
from .core.simple_optimizer import SimpleQuantumOptimizer
from .utils.simple_error_handling import (
    health_checker, safe_quantum_operation, 
    ErrorHandler, CircuitBreakerOpen
)
from .utils.simple_validation import TaskValidationError
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import click
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

@click.group()
def robust_cli():
    """Robust Quantum Task Planner CLI with Enhanced Error Handling"""
    pass

@robust_cli.command()
@click.option('--task-name', required=True, help='Name of the task to create')
@click.option('--description', required=True, help='Task description')
@click.option('--priority', default='medium', help='Priority level (critical/high/medium/low/minimal)')
@click.option('--coherence', default=0.8, help='Initial quantum coherence (0.0-1.0)')
def create_task(task_name: str, description: str, priority: str, coherence: float):
    """Create a robust quantum task with validation"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("Creating quantum task...", total=None)
        
        try:
            # Validate priority
            priority_map = {
                'critical': TaskPriority.CRITICAL,
                'high': TaskPriority.HIGH,
                'medium': TaskPriority.MEDIUM,
                'low': TaskPriority.LOW,
                'minimal': TaskPriority.MINIMAL
            }
            
            if priority.lower() not in priority_map:
                console.print(f"[red]❌ Invalid priority: {priority}[/red]")
                console.print("Valid priorities: critical, high, medium, low, minimal")
                return
            
            # Create task manager
            task_manager = RobustTaskManager()
            
            # Prepare task data
            task_data = {
                'title': task_name,
                'description': description,
                'priority': priority_map[priority.lower()],
                'quantum_coherence': coherence
            }
            
            # Create task
            task = task_manager.create_task(task_data)
            
            progress.update(task_progress, description="Task created successfully!")
            
            # Display success
            console.print(Panel(
                f"[green]✅ Quantum task created successfully![/green]\\n\\n"
                f"[bold]Task ID:[/bold] {task.task_id}\\n"
                f"[bold]Title:[/bold] {task.title}\\n"
                f"[bold]Description:[/bold] {task.description}\\n"
                f"[bold]Priority:[/bold] {task.priority.name} ({task.priority.probability_weight})\\n"
                f"[bold]State:[/bold] {task.state.name}\\n"
                f"[bold]Quantum Coherence:[/bold] {task.quantum_coherence:.3f}\\n"
                f"[bold]Completion Probability:[/bold] {task.get_completion_probability():.3f}",
                title="🌌 Quantum Task Created",
                border_style="green"
            ))
            
        except TaskValidationError as e:
            console.print(f"[red]❌ Validation Error: {e}[/red]")
        except CircuitBreakerOpen:
            console.print("[red]❌ System temporarily unavailable (circuit breaker open)[/red]")
        except Exception as e:
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
            logger.error(f"Task creation failed: {e}", exc_info=True)

@robust_cli.command()
def health_check():
    """Perform comprehensive system health check"""
    
    console.print("[bold blue]🔍 Performing System Health Check[/bold blue]\\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        health_task = progress.add_task("Checking system health...", total=None)
        
        try:
            # Create task manager and sample tasks
            task_manager = RobustTaskManager()
            
            # Create some test tasks
            test_tasks = [
                {'title': 'Health Check Task 1', 'description': 'Test task for health monitoring'},
                {'title': 'Health Check Task 2', 'description': 'Another test task'},
            ]
            
            for task_data in test_tasks:
                task_manager.create_task(task_data)
            
            # Check component health
            def check_task_manager():
                return len(task_manager.get_all_tasks()) > 0
            
            def check_optimizer():
                optimizer = SimpleQuantumOptimizer()
                return hasattr(optimizer, 'optimize_task_order')
            
            progress.update(health_task, description="Checking components...")
            
            # Run health checks
            health_checker.check_component_health("task_manager", check_task_manager)
            health_checker.check_component_health("quantum_optimizer", check_optimizer)
            
            # Get system health
            system_health = task_manager.get_system_health()
            overall_health = health_checker.get_overall_health()
            
            progress.update(health_task, description="Health check complete!")
            
            # Display results
            if overall_health["healthy"] and system_health["healthy"]:
                status_color = "green"
                status_icon = "✅"
                status_text = "HEALTHY"
            else:
                status_color = "red"
                status_icon = "❌"
                status_text = "UNHEALTHY"
            
            console.print(Panel(
                f"[{status_color}]{status_icon} System Status: {status_text}[/{status_color}]\\n\\n"
                f"[bold]Component Health:[/bold]\\n"
                f"• Task Manager: {'✅' if overall_health['components'].get('task_manager', {}).get('healthy') else '❌'}\\n"
                f"• Quantum Optimizer: {'✅' if overall_health['components'].get('quantum_optimizer', {}).get('healthy') else '❌'}\\n\\n"
                f"[bold]System Metrics:[/bold]\\n"
                f"• Total Tasks: {system_health['total_tasks']}\\n"
                f"• Healthy Tasks: {system_health['healthy_tasks']}\\n"
                f"• Average Coherence: {system_health['average_coherence']:.3f}\\n"
                f"• Average Performance: {system_health['average_performance']:.3f}\\n"
                f"• Error Count: {system_health['error_count']}",
                title="🏥 System Health Report",
                border_style=status_color
            ))
            
        except Exception as e:
            console.print(f"[red]❌ Health check failed: {e}[/red]")
            logger.error(f"Health check failed: {e}", exc_info=True)

@robust_cli.command()
def robust_demo():
    """Run a comprehensive robust quantum task planning demo"""
    
    console.print("[bold blue]🌌 Robust Quantum Task Planner Demo[/bold blue]\\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        demo_task = progress.add_task("Initializing demo...", total=None)
        
        try:
            # Create task manager
            task_manager = RobustTaskManager()
            optimizer = SimpleQuantumOptimizer()
            
            progress.update(demo_task, description="Creating sample tasks...")
            
            # Create sample tasks with different priorities
            sample_tasks_data = [
                {
                    'title': 'Critical Security Audit',
                    'description': 'Comprehensive security audit of quantum systems',
                    'priority': TaskPriority.CRITICAL,
                    'quantum_coherence': 0.95
                },
                {
                    'title': 'Data Analysis Pipeline',
                    'description': 'Analyze customer behavior data',
                    'priority': TaskPriority.HIGH,
                    'quantum_coherence': 0.8
                },
                {
                    'title': 'Documentation Update',
                    'description': 'Update API documentation',
                    'priority': TaskPriority.MEDIUM,
                    'quantum_coherence': 0.7
                },
                {
                    'title': 'Performance Optimization',
                    'description': 'Optimize quantum algorithm performance',
                    'priority': TaskPriority.HIGH,
                    'quantum_coherence': 0.85
                }
            ]
            
            tasks = []
            for task_data in sample_tasks_data:
                task = task_manager.create_task(task_data)
                tasks.append(task)
            
            progress.update(demo_task, description="Displaying task queue...")
            
            # Display tasks table
            table = Table(title="🌌 Robust Quantum Task Queue")
            table.add_column("Task ID", style="cyan", no_wrap=True)
            table.add_column("Title", style="green")
            table.add_column("Priority", style="yellow")
            table.add_column("State", style="magenta")
            table.add_column("Coherence", style="blue")
            table.add_column("Completion Prob", style="bright_green")
            table.add_column("Health", style="white")
            
            for task in tasks:
                health_metrics = task.get_health_metrics()
                health_icon = "✅" if health_metrics.get("is_healthy") else "❌"
                
                table.add_row(
                    task.task_id[:8] + "...",
                    task.title,
                    f"{task.priority.name}\\n({task.priority.probability_weight:.2f})",
                    task.state.name,
                    f"{task.quantum_coherence:.3f}",
                    f"{task.get_completion_probability():.3f}",
                    health_icon
                )
            
            console.print(table)
            
            progress.update(demo_task, description="Optimizing task order...")
            
            # Optimize task order
            console.print("\\n[bold yellow]⚡ Quantum Optimization Results[/bold yellow]")
            optimized_order = optimizer.optimize_task_order(tasks)
            
            for i, task in enumerate(optimized_order, 1):
                priority_color = {
                    TaskPriority.CRITICAL: "red",
                    TaskPriority.HIGH: "orange1",
                    TaskPriority.MEDIUM: "yellow",
                    TaskPriority.LOW: "green",
                    TaskPriority.MINIMAL: "blue"
                }.get(task.priority, "white")
                
                console.print(
                    f"{i}. [{priority_color}]{task.title}[/{priority_color}] "
                    f"(Priority: {task.priority.probability_weight:.2f}, "
                    f"Completion: {task.get_completion_probability():.3f})"
                )
            
            progress.update(demo_task, description="Demonstrating state transitions...")
            
            # Demonstrate state transitions
            console.print("\\n[bold cyan]🔄 State Transition Demo[/bold cyan]")
            demo_task_obj = tasks[0]  # Use first task for demo
            
            try:
                console.print(f"Initial state: {demo_task_obj.state.name}")
                demo_task_obj.transition_state(TaskState.IN_PROGRESS)
                console.print(f"Transitioned to: {demo_task_obj.state.name}")
                demo_task_obj.transition_state(TaskState.COMPLETED)
                console.print(f"Transitioned to: {demo_task_obj.state.name}")
            except Exception as e:
                console.print(f"[red]Transition error: {e}[/red]")
            
            progress.update(demo_task, description="Demo completed successfully!")
            
            # Final system health
            console.print("\\n[bold green]📊 Final System Health[/bold green]")
            system_health = task_manager.get_system_health()
            console.print(f"System Health: {'✅ Healthy' if system_health['healthy'] else '❌ Unhealthy'}")
            console.print(f"Total Tasks: {system_health['total_tasks']}")
            console.print(f"Average Performance: {system_health['average_performance']:.3f}")
            
        except Exception as e:
            console.print(f"[red]❌ Demo failed: {e}[/red]")
            logger.error(f"Demo failed: {e}", exc_info=True)

if __name__ == "__main__":
    robust_cli()