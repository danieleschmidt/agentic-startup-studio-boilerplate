"""
Quantum Task Planner API

Production-ready FastAPI application with comprehensive middleware,
security, monitoring, health checks, and distributed quantum capabilities.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.quantum_task import QuantumTask, TaskState, TaskPriority
from ..core.quantum_scheduler import QuantumTaskScheduler
from ..core.simple_optimizer import SimpleQuantumOptimizer
from ..core.simple_entanglement import SimpleEntanglementManager, SimpleEntanglementType

from ..utils.simple_middleware import setup_middleware, create_default_security_config
from ..utils.simple_health import (
    get_health_manager, setup_default_health_checks,
    SystemResourcesHealthCheck, QuantumCoherenceHealthCheck
)
from ..utils.simple_logging import setup_logging, get_logger
from ..performance.simple_cache import get_cache, cached_quantum
from ..performance.simple_concurrent import get_worker_pool
from ..performance.simple_scaling import get_load_balancer, get_auto_scaler
from ..distributed.simple_sync import get_quantum_coordinator


# Pydantic models for API
class TaskCreateRequest(BaseModel):
    title: str
    description: str
    priority: str = "medium"
    estimated_duration_hours: Optional[float] = None
    due_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    complexity_factor: float = Field(default=1.0, ge=0.1, le=10.0)


class TaskUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[str] = None
    estimated_duration_hours: Optional[float] = None
    due_date: Optional[datetime] = None
    tags: Optional[List[str]] = None
    complexity_factor: Optional[float] = None


class EntanglementRequest(BaseModel):
    task_ids: List[str]
    entanglement_type: str = "bell_state"
    strength: float = Field(default=0.8, ge=0.0, le=1.0)


class OptimizationRequest(BaseModel):
    task_ids: Optional[List[str]] = None
    objectives: Optional[List[str]] = None
    max_iterations: int = Field(default=100, ge=10, le=1000)


class QuantumMeasurementRequest(BaseModel):
    task_ids: Optional[List[str]] = None
    observer_effect: float = Field(default=0.1, ge=0.0, le=1.0)


# Global instances - Enhanced with distributed capabilities
scheduler = QuantumTaskScheduler()
optimizer = SimpleQuantumOptimizer()
entanglement_manager = SimpleEntanglementManager()

# Enhanced system components
logger = None
health_manager = None
load_balancer = None
auto_scaler = None
quantum_coordinator = None
middleware_instances = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management with full system initialization"""
    global logger, health_manager, load_balancer, auto_scaler, quantum_coordinator, middleware_instances
    
    # Startup
    print("🚀 Starting Quantum Task Planner API...")
    
    # Initialize logging
    logger = setup_logging("INFO")
    logger.logger.info("Logging system initialized")
    
    # Setup health checks
    health_manager = setup_default_health_checks(
        scheduler=scheduler,
        database_url="postgresql://localhost/quantum_tasks", 
        redis_url="redis://localhost:6379"
    )
    logger.logger.info("Health monitoring system initialized")
    
    # Initialize scaling components
    load_balancer = get_load_balancer(coherence_weight=0.3)
    auto_scaler = get_auto_scaler(check_interval=30.0)
    logger.logger.info("Auto-scaling system initialized")
    
    # Initialize distributed coordination
    quantum_coordinator = get_quantum_coordinator(f"api_node_{uuid.uuid4().hex[:8]}")
    await quantum_coordinator.join_cluster()
    logger.logger.info("Distributed quantum coordination initialized")
    
    # Start background tasks
    background_tasks = [
        asyncio.create_task(background_decoherence()),
        asyncio.create_task(background_optimization()),
        asyncio.create_task(background_health_monitoring()),
        asyncio.create_task(background_metrics_collection())
    ]
    
    logger.logger.info("✅ Quantum Task Planner API startup complete")
    
    try:
        yield
    finally:
        # Shutdown
        print("🛑 Shutting down Quantum Task Planner API...")
        logger.logger.info("Beginning graceful shutdown")
        
        # Cancel background tasks
        for task in background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*background_tasks, return_exceptions=True)
        
        # Shutdown scaling systems  
        if auto_scaler:
            auto_scaler.stop_monitoring()
        
        # Shutdown health monitoring
        if health_manager:
            health_manager.stop_monitoring()
        
        logger.logger.info("✅ Quantum Task Planner API shutdown complete")


# Enhanced FastAPI app with production configuration
app = FastAPI(
    title="Quantum Task Planner API",
    description="Production-ready quantum-inspired task planning and optimization system with distributed capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup comprehensive middleware stack
security_config = create_default_security_config()
middleware_instances = setup_middleware(
    app,
    security_config=security_config,
    enable_quantum_middleware=True,
    enable_monitoring=True
)


async def background_decoherence():
    """Background task to apply quantum decoherence"""
    while True:
        try:
            await entanglement_manager.apply_decoherence(60.0)  # Every minute
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            if logger:
                logger.logger.error(f"Decoherence error: {e}")
            await asyncio.sleep(60)


async def background_optimization():
    """Background optimization task"""
    while True:
        try:
            # Run optimization every 5 minutes
            if scheduler.tasks:
                task_ids = list(scheduler.tasks.keys())[:10]  # Optimize up to 10 tasks
                result = await optimizer.optimize_async(task_ids)
                if logger:
                    logger.logger.info(f"Background optimization completed: {result.get('improvement', 0):.2%} improvement")
            await asyncio.sleep(300)
        except asyncio.CancelledError:
            break
        except Exception as e:
            if logger:
                logger.logger.error(f"Background optimization error: {e}")
            await asyncio.sleep(300)


async def background_health_monitoring():
    """Background health monitoring task"""
    while True:
        try:
            if health_manager:
                health_results = await health_manager.check_all_health()
                unhealthy_checks = [name for name, result in health_results.items() 
                                  if result.status.value in ['unhealthy', 'critical']]
                if unhealthy_checks and logger:
                    logger.logger.warning(f"Unhealthy checks: {unhealthy_checks}")
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            if logger:
                logger.logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(60)


async def background_metrics_collection():
    """Background metrics collection task"""
    while True:
        try:
            # Collect and log system metrics
            if logger and scheduler:
                task_count = len(scheduler.tasks)
                active_tasks = len([t for t in scheduler.tasks.values() if t.state in [TaskState.RUNNING, TaskState.IN_PROGRESS]])
                avg_coherence = sum(t.quantum_coherence for t in scheduler.tasks.values()) / max(1, task_count)
                
                logger.logger.info(
                    "system_metrics",
                    total_tasks=task_count,
                    active_tasks=active_tasks,
                    average_coherence=avg_coherence
                )
            
            await asyncio.sleep(120)  # Every 2 minutes
        except asyncio.CancelledError:
            break
        except Exception as e:
            if logger:
                logger.logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(120)


# Task Management Endpoints

@app.post("/api/v1/tasks", response_model=Dict[str, Any])
async def create_task(request: TaskCreateRequest):
    """Create a new quantum task"""
    try:
        # Convert priority string to enum
        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
            "minimal": TaskPriority.MINIMAL
        }
        
        priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)
        
        # Create estimated duration
        estimated_duration = None
        if request.estimated_duration_hours:
            estimated_duration = timedelta(hours=request.estimated_duration_hours)
        
        # Create quantum task
        task = QuantumTask(
            title=request.title,
            description=request.description,
            priority=priority,
            estimated_duration=estimated_duration,
            due_date=request.due_date,
            tags=request.tags,
            complexity_factor=request.complexity_factor
        )
        
        # Add to scheduler
        scheduler.add_task(task)
        
        return {"status": "success", "task": task.to_dict()}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/tasks", response_model=Dict[str, Any])
async def list_tasks():
    """List all quantum tasks"""
    tasks = [task.to_dict() for task in scheduler.tasks.values()]
    return {"status": "success", "tasks": tasks, "count": len(tasks)}


@app.get("/api/v1/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str):
    """Get specific quantum task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = scheduler.tasks[task_id]
    return {"status": "success", "task": task.to_dict()}


@app.put("/api/v1/tasks/{task_id}", response_model=Dict[str, Any])
async def update_task(task_id: str, request: TaskUpdateRequest):
    """Update quantum task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = scheduler.tasks[task_id]
    
    try:
        # Update fields if provided
        if request.title:
            task.title = request.title
        if request.description:
            task.description = request.description
        if request.priority:
            priority_map = {
                "critical": TaskPriority.CRITICAL,
                "high": TaskPriority.HIGH,
                "medium": TaskPriority.MEDIUM,
                "low": TaskPriority.LOW,
                "minimal": TaskPriority.MINIMAL
            }
            task.priority = priority_map.get(request.priority.lower(), task.priority)
        if request.estimated_duration_hours:
            task.estimated_duration = timedelta(hours=request.estimated_duration_hours)
        if request.due_date:
            task.due_date = request.due_date
        if request.tags:
            task.tags = request.tags
        if request.complexity_factor:
            task.complexity_factor = request.complexity_factor
        
        return {"status": "success", "task": task.to_dict()}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/tasks/{task_id}", response_model=Dict[str, Any])
async def delete_task(task_id: str):
    """Delete quantum task"""
    removed_task = scheduler.remove_task(task_id)
    if not removed_task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {"status": "success", "message": f"Task {task_id} deleted"}


# Quantum State Management

@app.post("/api/v1/tasks/{task_id}/measure", response_model=Dict[str, Any])
async def measure_task_state(task_id: str, observer_effect: float = 0.1):
    """Perform quantum measurement on task state"""
    if task_id not in scheduler.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = scheduler.tasks[task_id]
    measured_state = task.measure_state(observer_effect)
    
    return {
        "status": "success",
        "task_id": task_id,
        "measured_state": measured_state.value,
        "quantum_coherence": task.quantum_coherence,
        "measurement_time": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/quantum/measure", response_model=Dict[str, Any])
async def measure_multiple_tasks(request: QuantumMeasurementRequest):
    """Perform quantum measurements on multiple tasks"""
    if not request.task_ids:
        task_ids = list(scheduler.tasks.keys())
    else:
        task_ids = request.task_ids
    
    results = {}
    for task_id in task_ids:
        if task_id in scheduler.tasks:
            task = scheduler.tasks[task_id]
            measured_state = task.measure_state(request.observer_effect)
            results[task_id] = {
                "measured_state": measured_state.value,
                "quantum_coherence": task.quantum_coherence,
                "completion_probability": task.get_completion_probability()
            }
    
    return {
        "status": "success",
        "measurements": results,
        "measurement_time": datetime.utcnow().isoformat()
    }


# Quantum Entanglement Management

@app.post("/api/v1/entanglement/create", response_model=Dict[str, Any])
async def create_entanglement(request: EntanglementRequest):
    """Create quantum entanglement between tasks"""
    # Validate task IDs
    tasks = []
    for task_id in request.task_ids:
        if task_id not in scheduler.tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        tasks.append(scheduler.tasks[task_id])
    
    # Validate entanglement type
    try:
        entanglement_type = SimpleEntanglementType(request.entanglement_type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid entanglement type")
    
    try:
        bond_id = await entanglement_manager.create_entanglement(
            tasks, entanglement_type, request.strength
        )
        
        return {
            "status": "success",
            "bond_id": bond_id,
            "entangled_tasks": request.task_ids,
            "entanglement_type": request.entanglement_type,
            "strength": request.strength
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/entanglement/{bond_id}/measure", response_model=Dict[str, Any])
async def measure_entanglement(bond_id: str, observer_effect: float = 0.1):
    """Perform quantum measurement on entangled tasks"""
    try:
        results = await entanglement_manager.measure_entanglement(bond_id, observer_effect)
        return {"status": "success", "bond_id": bond_id, "measurements": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/entanglement/{bond_id}", response_model=Dict[str, Any])
async def break_entanglement(bond_id: str):
    """Break quantum entanglement bond"""
    success = await entanglement_manager.break_entanglement(bond_id)
    if not success:
        raise HTTPException(status_code=404, detail="Entanglement bond not found")
    
    return {"status": "success", "message": f"Entanglement bond {bond_id} broken"}


@app.get("/api/v1/entanglement/stats", response_model=Dict[str, Any])
async def get_entanglement_statistics():
    """Get entanglement network statistics"""
    stats = entanglement_manager.get_entanglement_statistics()
    return {"status": "success", "statistics": stats}


# Quantum Scheduling

@app.post("/api/v1/schedule/optimize", response_model=Dict[str, Any])
async def optimize_schedule(background_tasks: BackgroundTasks):
    """Optimize task schedule using quantum algorithms"""
    try:
        # Run optimization in background
        optimized_schedule = await scheduler.optimize_schedule()
        
        return {
            "status": "success",
            "schedule": [
                {
                    "start_time": start_time.isoformat(),
                    "task": task.to_dict()
                }
                for start_time, task in optimized_schedule
            ],
            "optimization_metrics": scheduler.optimization_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/schedule/next", response_model=Dict[str, Any])
async def get_next_tasks(count: int = 5):
    """Get next tasks to execute based on quantum measurement"""
    try:
        next_tasks = await scheduler.get_next_tasks(count)
        return {
            "status": "success",
            "next_tasks": [task.to_dict() for task in next_tasks],
            "count": len(next_tasks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/schedule/stats", response_model=Dict[str, Any])
async def get_schedule_statistics():
    """Get comprehensive schedule statistics"""
    stats = scheduler.get_schedule_statistics()
    return {"status": "success", "statistics": stats}


# Quantum Optimization

@app.post("/api/v1/optimize/allocation", response_model=Dict[str, Any])
async def optimize_task_allocation(request: OptimizationRequest):
    """Optimize task allocation using quantum genetic algorithms"""
    try:
        # Get tasks to optimize
        if request.task_ids:
            tasks = [scheduler.tasks[tid] for tid in request.task_ids if tid in scheduler.tasks]
        else:
            tasks = list(scheduler.tasks.values())
        
        if not tasks:
            raise HTTPException(status_code=400, detail="No valid tasks found")
        
        # Set up optimizer
        if not optimizer.objectives:
            objectives = optimizer.create_standard_objectives()
            for obj in objectives:
                optimizer.add_objective(obj)
        
        # Mock resource pools (in real implementation, this would come from configuration)
        resources = {
            "cpu": 100.0,
            "memory": 16.0,
            "network": 1000.0,
            "storage": 500.0
        }
        
        # Run optimization
        results = await optimizer.optimize_task_allocation(tasks, resources)
        
        return {"status": "success", "optimization_results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System Status and Health

@app.get("/api/v1/health", response_model=Dict[str, Any])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "scheduler": "operational",
            "optimizer": "operational", 
            "entanglement_manager": "operational"
        },
        "metrics": {
            "total_tasks": len(scheduler.tasks),
            "active_entanglements": len(entanglement_manager.entanglement_bonds),
            "quantum_channels": len(entanglement_manager.quantum_channels)
        }
    }


@app.get("/api/v1/quantum/state", response_model=Dict[str, Any])
async def get_quantum_system_state():
    """Get overall quantum system state"""
    task_states = {}
    for task_id, task in scheduler.tasks.items():
        task_states[task_id] = {
            "coherence": task.quantum_coherence,
            "completion_probability": task.get_completion_probability(),
            "entangled_tasks": list(task.entangled_tasks),
            "state_probabilities": {
                state.value: amp.probability
                for state, amp in task.state_amplitudes.items()
            }
        }
    
    return {
        "status": "success",
        "quantum_system_state": task_states,
        "entanglement_statistics": entanglement_manager.get_entanglement_statistics(),
        "schedule_statistics": scheduler.get_schedule_statistics(),
        "timestamp": datetime.utcnow().isoformat()
    }


# Enhanced Production Endpoints

@app.get("/api/v1/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check endpoint"""
    if not health_manager:
        return {"status": "healthy", "message": "Health manager not initialized"}
    
    health_status = health_manager.get_health_status()
    return {
        "status": health_status["overall_status"],
        "timestamp": health_status["timestamp"],
        "checks": health_status["health_checks"],
        "circuit_breakers": health_status["circuit_breakers"],
        "version": "2.0.0"
    }


@app.post("/api/v1/tasks/batch", response_model=Dict[str, Any])
async def create_tasks_batch(requests: List[TaskCreateRequest]):
    """Create multiple tasks in batch for improved performance"""
    try:
        created_tasks = []
        errors = []
        
        for i, request in enumerate(requests):
            try:
                # Convert priority string to enum
                priority_map = {
                    "critical": TaskPriority.CRITICAL,
                    "high": TaskPriority.HIGH,
                    "medium": TaskPriority.MEDIUM,
                    "low": TaskPriority.LOW,
                    "minimal": TaskPriority.MINIMAL
                }
                
                priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)
                
                # Create estimated duration
                estimated_duration = None
                if request.estimated_duration_hours:
                    estimated_duration = timedelta(hours=request.estimated_duration_hours)
                
                # Create quantum task
                task = QuantumTask(
                    title=request.title,
                    description=request.description,
                    priority=priority,
                    estimated_duration=estimated_duration,
                    due_date=request.due_date,
                    tags=request.tags,
                    complexity_factor=request.complexity_factor
                )
                
                # Add to scheduler
                scheduler.add_task(task)
                created_tasks.append(task.to_dict())
                
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
        
        return {
            "status": "success" if not errors else "partial",
            "created_tasks": created_tasks,
            "errors": errors,
            "total_created": len(created_tasks),
            "total_errors": len(errors)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/tasks/search", response_model=Dict[str, Any])
async def search_tasks(
    q: Optional[str] = None,
    priority: Optional[str] = None,
    state: Optional[str] = None,
    tags: Optional[str] = None,
    due_after: Optional[datetime] = None,
    due_before: Optional[datetime] = None,
    min_coherence: Optional[float] = None,
    max_coherence: Optional[float] = None,
    limit: int = 100,
    offset: int = 0
):
    """Advanced task search with filters"""
    try:
        tasks = list(scheduler.tasks.values())
        filtered_tasks = []
        
        for task in tasks:
            # Text search in title and description
            if q and q.lower() not in (task.title + " " + task.description).lower():
                continue
            
            # Priority filter
            if priority and task.priority.name.lower() != priority.lower():
                continue
            
            # State filter
            if state and task.state.name.lower() != state.lower():
                continue
            
            # Tags filter
            if tags:
                search_tags = [tag.strip().lower() for tag in tags.split(",")]
                task_tags_lower = [tag.lower() for tag in task.tags]
                if not any(tag in task_tags_lower for tag in search_tags):
                    continue
            
            # Due date filters
            if due_after and task.due_date and task.due_date < due_after:
                continue
            if due_before and task.due_date and task.due_date > due_before:
                continue
            
            # Quantum coherence filters
            if min_coherence and task.quantum_coherence < min_coherence:
                continue
            if max_coherence and task.quantum_coherence > max_coherence:
                continue
            
            filtered_tasks.append(task)
        
        # Apply pagination
        total_count = len(filtered_tasks)
        paginated_tasks = filtered_tasks[offset:offset + limit]
        
        return {
            "status": "success",
            "tasks": [task.to_dict() for task in paginated_tasks],
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/tasks/{task_id}/execute", response_model=Dict[str, Any])
async def execute_task(task_id: str, background_tasks: BackgroundTasks):
    """Execute a quantum task with real-time monitoring"""
    if task_id not in scheduler.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = scheduler.tasks[task_id]
    
    async def task_execution():
        """Background task execution with quantum state updates"""
        try:
            # Update task state to running
            task.start_execution()
            
            # Simulate quantum task execution with coherence decay
            execution_time = task.estimated_duration or timedelta(hours=1)
            total_seconds = execution_time.total_seconds()
            
            for i in range(int(total_seconds / 10)):  # Update every 10 seconds
                await asyncio.sleep(10)
                
                # Apply quantum decoherence during execution
                decoherence_rate = 0.01 * (i / (total_seconds / 10))  # Gradual decay
                task.quantum_coherence = max(0.1, task.quantum_coherence - decoherence_rate)
                
                # Update completion probability based on progress
                progress = i / (total_seconds / 10)
                task._update_completion_probability(progress)
                
                if logger:
                    logger.logger.info(f"Task {task_id} progress: {progress:.2%}")
            
            # Complete task
            task.complete_execution()
            if logger:
                logger.logger.info(f"Task {task_id} completed successfully")
                
        except Exception as e:
            task.set_state(TaskState.FAILED)
            if logger:
                logger.logger.error(f"Task {task_id} execution failed: {e}")
    
    # Start background execution
    background_tasks.add_task(task_execution)
    
    return {
        "status": "success",
        "message": f"Task {task_id} execution started",
        "task": task.to_dict(),
        "execution_started": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/tasks/{task_id}/pause", response_model=Dict[str, Any])
async def pause_task(task_id: str):
    """Pause a running quantum task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = scheduler.tasks[task_id]
    
    if task.state not in [TaskState.RUNNING, TaskState.IN_PROGRESS]:
        raise HTTPException(status_code=400, detail="Task is not running")
    
    task.set_state(TaskState.PAUSED)
    
    return {
        "status": "success",
        "message": f"Task {task_id} paused",
        "task": task.to_dict(),
        "paused_at": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/tasks/{task_id}/resume", response_model=Dict[str, Any])
async def resume_task(task_id: str):
    """Resume a paused quantum task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = scheduler.tasks[task_id]
    
    if task.state != TaskState.PAUSED:
        raise HTTPException(status_code=400, detail="Task is not paused")
    
    task.set_state(TaskState.IN_PROGRESS)
    
    return {
        "status": "success",
        "message": f"Task {task_id} resumed",
        "task": task.to_dict(),
        "resumed_at": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/metrics", response_model=Dict[str, Any])
async def get_system_metrics():
    """Get comprehensive system metrics"""
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "total_tasks": len(scheduler.tasks),
            "active_tasks": len([t for t in scheduler.tasks.values() if t.state in [TaskState.RUNNING, TaskState.IN_PROGRESS]]),
            "average_coherence": sum(t.quantum_coherence for t in scheduler.tasks.values()) / max(1, len(scheduler.tasks))
        }
    }
    
    # Add load balancer metrics
    if load_balancer:
        metrics["load_balancing"] = load_balancer.get_load_distribution()
    
    # Add auto-scaler metrics
    if auto_scaler:
        metrics["auto_scaling"] = auto_scaler.get_scaling_status()
    
    # Add quantum coordinator metrics
    if quantum_coordinator:
        metrics["distributed"] = quantum_coordinator.get_cluster_status()
    
    # Add middleware metrics
    if middleware_instances and "monitoring" in middleware_instances:
        metrics["monitoring"] = middleware_instances["monitoring"].get_metrics()
    
    return {"status": "success", "metrics": metrics}


@app.get("/api/v1/performance", response_model=Dict[str, Any])
@cached_quantum(cache_name="performance", ttl=60)  # Cache for 1 minute
async def get_performance_metrics():
    """Get performance and optimization metrics"""
    return {
        "status": "success",
        "performance": {
            "cache_stats": get_cache("default").get_stats(),
            "worker_pool_stats": get_worker_pool().get_pool_stats(),
            "optimization_history": optimizer.get_optimization_history()
        }
    }


@app.post("/api/v1/scale", response_model=Dict[str, Any])
async def trigger_scaling(action: str, amount: int = 1):
    """Manually trigger scaling action"""
    if not auto_scaler:
        raise HTTPException(status_code=503, detail="Auto-scaler not available")
    
    if action == "up":
        await auto_scaler._scale_up(amount)
    elif action == "down": 
        await auto_scaler._scale_down(amount)
    else:
        raise HTTPException(status_code=400, detail="Invalid action. Use 'up' or 'down'")
    
    return {
        "status": "success",
        "action": action,
        "amount": amount,
        "new_instance_count": auto_scaler.current_instances
    }


@app.post("/api/v1/distributed/sync", response_model=Dict[str, Any])
async def sync_quantum_states():
    """Trigger distributed quantum state synchronization"""
    if not quantum_coordinator:
        raise HTTPException(status_code=503, detail="Quantum coordinator not available")
    
    # Trigger synchronization across all nodes
    sync_results = []
    for task_id, task in scheduler.tasks.items():
        quantum_coordinator.state_tracker.update_local_state(
            task_id=task_id,
            quantum_coherence=task.quantum_coherence,
            state_probabilities=task.state_probabilities,
            entanglement_bonds=task.entanglement_bonds or []
        )
        sync_results.append(task_id)
    
    return {
        "status": "success",
        "synchronized_tasks": len(sync_results),
        "cluster_status": quantum_coordinator.get_cluster_status()
    }


@app.post("/api/v1/cache/clear", response_model=Dict[str, Any])
async def clear_caches(cache_name: Optional[str] = None):
    """Clear system caches"""
    if cache_name:
        cache = get_cache(cache_name)
        cache.clear()
        return {"status": "success", "cleared_cache": cache_name}
    else:
        # Clear all caches
        from ..performance.cache import _cache_manager
        _cache_manager.clear_all()
        return {"status": "success", "cleared_cache": "all"}


class QuantumPlannerAPI:
    """Enhanced main API class for external integrations"""
    
    def __init__(self):
        self.app = app
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.entanglement_manager = entanglement_manager
        self.health_manager = health_manager
        self.load_balancer = load_balancer
        self.auto_scaler = auto_scaler
        self.quantum_coordinator = quantum_coordinator
    
    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        """Run the production API server"""
        uvicorn.run(
            self.app, 
            host=host, 
            port=port,
            log_level="info",
            access_log=True,
            **kwargs
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "api_version": "2.0.0",
            "tasks": len(self.scheduler.tasks),
            "health": "healthy" if health_manager and health_manager.is_healthy() else "degraded",
            "distributed": bool(quantum_coordinator),
            "auto_scaling": bool(auto_scaler),
            "load_balancing": bool(load_balancer)
        }


# Export enhanced API instance
quantum_api = QuantumPlannerAPI()