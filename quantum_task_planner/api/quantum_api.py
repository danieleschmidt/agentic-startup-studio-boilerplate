"""
Quantum Task Planner API

FastAPI-based REST API for quantum task planning with real-time
quantum state monitoring, entanglement management, and optimization endpoints.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..core.quantum_task import QuantumTask, TaskState, TaskPriority
from ..core.quantum_scheduler import QuantumTaskScheduler
from ..core.quantum_optimizer import QuantumProbabilityOptimizer
from ..core.entanglement_manager import TaskEntanglementManager, EntanglementType


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


# Global instances
scheduler = QuantumTaskScheduler()
optimizer = QuantumProbabilityOptimizer()
entanglement_manager = TaskEntanglementManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with background tasks"""
    # Start background decoherence process
    decoherence_task = asyncio.create_task(background_decoherence())
    
    try:
        yield
    finally:
        # Clean shutdown
        decoherence_task.cancel()
        try:
            await decoherence_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Quantum Task Planner API",
    description="Advanced quantum-inspired task planning and optimization system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
            print(f"Decoherence error: {e}")
            await asyncio.sleep(60)


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
        entanglement_type = EntanglementType(request.entanglement_type)
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


class QuantumPlannerAPI:
    """Main API class for external integrations"""
    
    def __init__(self):
        self.app = app
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.entanglement_manager = entanglement_manager
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Export main API instance
quantum_api = QuantumPlannerAPI()