"""
Agent API endpoints.
Handles requests for AI agent operations and task execution.
"""

import logging
from typing import Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from app.agents.crew import AgentRequest, AgentResponse, get_agent_crew
from app.core.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class TaskSubmissionResponse(BaseModel):
    """Response model for task submission."""
    
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    
    task_id: str
    status: str
    progress: float
    result: str = None
    error: str = None


# In-memory task storage (in production, use Redis or database)
task_storage: Dict[str, Dict] = {}


@router.post("/research", response_model=AgentResponse)
async def execute_research_task(
    request: AgentRequest,
    current_user: User = Depends(get_current_user)
) -> AgentResponse:
    """
    Execute a research task using AI agents.
    
    Args:
        request: Research task request
        current_user: Current authenticated user
        
    Returns:
        AgentResponse: Task execution result
    """
    try:
        logger.info(f"User {current_user.id} requested research task: {request.task_description}")
        
        # Get agent crew instance
        crew = get_agent_crew()
        
        # Execute the research task
        response = await crew.execute_research_task(request)
        
        logger.info(f"Research task completed for user {current_user.id}: success={response.success}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error executing research task for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute research task: {str(e)}"
        )


@router.post("/research/async", response_model=TaskSubmissionResponse)
async def submit_research_task_async(
    request: AgentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> TaskSubmissionResponse:
    """
    Submit a research task for asynchronous execution.
    
    Args:
        request: Research task request
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        
    Returns:
        TaskSubmissionResponse: Task submission confirmation
    """
    try:
        import uuid
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        task_storage[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "user_id": current_user.id,
            "request": request.dict(),
            "result": None,
            "error": None
        }
        
        # Add background task
        background_tasks.add_task(
            execute_background_research_task,
            task_id,
            request
        )
        
        logger.info(f"Submitted async research task {task_id} for user {current_user.id}")
        
        return TaskSubmissionResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted successfully. Use /agents/tasks/{task_id} to check status."
        )
        
    except Exception as e:
        logger.error(f"Error submitting async research task for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit research task: {str(e)}"
        )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
) -> TaskStatusResponse:
    """
    Get the status of an asynchronous task.
    
    Args:
        task_id: Task identifier
        current_user: Current authenticated user
        
    Returns:
        TaskStatusResponse: Task status information
    """
    try:
        # Check if task exists
        if task_id not in task_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        task_data = task_storage[task_id]
        
        # Check if user owns the task
        if task_data["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: task belongs to another user"
            )
        
        return TaskStatusResponse(
            task_id=task_id,
            status=task_data["status"],
            progress=task_data["progress"],
            result=task_data["result"],
            error=task_data["error"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@router.get("/tasks", response_model=List[TaskStatusResponse])
async def list_user_tasks(
    current_user: User = Depends(get_current_user)
) -> List[TaskStatusResponse]:
    """
    List all tasks for the current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        List[TaskStatusResponse]: List of user's tasks
    """
    try:
        user_tasks = []
        
        for task_id, task_data in task_storage.items():
            if task_data["user_id"] == current_user.id:
                user_tasks.append(TaskStatusResponse(
                    task_id=task_id,
                    status=task_data["status"],
                    progress=task_data["progress"],
                    result=task_data["result"],
                    error=task_data["error"]
                ))
        
        return user_tasks
        
    except Exception as e:
        logger.error(f"Error listing tasks for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Cancel or delete a task.
    
    Args:
        task_id: Task identifier
        current_user: Current authenticated user
        
    Returns:
        Dict[str, str]: Cancellation confirmation
    """
    try:
        # Check if task exists
        if task_id not in task_storage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
        
        task_data = task_storage[task_id]
        
        # Check if user owns the task
        if task_data["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: task belongs to another user"
            )
        
        # Remove task from storage
        del task_storage[task_id]
        
        logger.info(f"Task {task_id} cancelled by user {current_user.id}")
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}"
        )


async def execute_background_research_task(task_id: str, request: AgentRequest):
    """
    Execute a research task in the background.
    
    Args:
        task_id: Task identifier
        request: Research task request
    """
    try:
        logger.info(f"Starting background execution of task {task_id}")
        
        # Update task status to running
        task_storage[task_id]["status"] = "running"
        task_storage[task_id]["progress"] = 0.1
        
        # Get agent crew instance
        crew = get_agent_crew()
        
        # Update progress
        task_storage[task_id]["progress"] = 0.3
        
        # Execute the research task
        response = await crew.execute_research_task(request)
        
        # Update task with results
        if response.success:
            task_storage[task_id]["status"] = "completed"
            task_storage[task_id]["progress"] = 1.0
            task_storage[task_id]["result"] = response.result
        else:
            task_storage[task_id]["status"] = "failed"
            task_storage[task_id]["progress"] = 1.0
            task_storage[task_id]["error"] = response.result
        
        logger.info(f"Background task {task_id} completed with status: {task_storage[task_id]['status']}")
        
    except Exception as e:
        logger.error(f"Error in background task {task_id}: {e}")
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["error"] = str(e)