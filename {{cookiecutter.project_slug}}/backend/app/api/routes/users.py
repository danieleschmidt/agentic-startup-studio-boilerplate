"""
User management endpoints.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_active_user, get_current_superuser
from app.core.database import get_async_db_dependency
from app.core.security import get_password_hash, verify_password
from app.models.user import User

router = APIRouter()


# Pydantic models for request/response
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class UserStats(BaseModel):
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_api_keys: int


@router.get("/me", response_model=UserResponse, summary="Get current user")
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """
    Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        UserResponse: Current user data
    """
    return UserResponse.model_validate(current_user)


@router.put("/me", response_model=UserResponse, summary="Update current user")
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> UserResponse:
    """
    Update current user information.
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        UserResponse: Updated user data
    """
    # Check if email/username already exists (if being updated)
    if user_update.email and user_update.email != current_user.email:
        result = await db.execute(
            select(User).where(User.email == user_update.email)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    if user_update.username and user_update.username != current_user.username:
        result = await db.execute(
            select(User).where(User.username == user_update.username)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Update user fields
    update_data = user_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(current_user)
    
    return UserResponse.model_validate(current_user)


@router.post("/me/change-password", summary="Change user password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> dict:
    """
    Change current user password.
    
    Args:
        password_data: Password change data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        dict: Success message
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    current_user.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Password updated successfully"}


@router.get("/me/stats", response_model=UserStats, summary="Get user statistics")
async def get_user_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> UserStats:
    """
    Get current user statistics.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        UserStats: User statistics
    """
    from app.models.agent_task import AgentTask, TaskStatus
    
    # Get task statistics
    total_tasks_result = await db.execute(
        select(AgentTask).where(AgentTask.user_id == current_user.id)
    )
    total_tasks = len(total_tasks_result.scalars().all())
    
    completed_tasks_result = await db.execute(
        select(AgentTask).where(
            AgentTask.user_id == current_user.id,
            AgentTask.status == TaskStatus.COMPLETED
        )
    )
    completed_tasks = len(completed_tasks_result.scalars().all())
    
    failed_tasks_result = await db.execute(
        select(AgentTask).where(
            AgentTask.user_id == current_user.id,
            AgentTask.status == TaskStatus.FAILED
        )
    )
    failed_tasks = len(failed_tasks_result.scalars().all())
    
    # Get active API keys count
    active_api_keys = len(current_user.get_active_api_keys())
    
    return UserStats(
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
        active_api_keys=active_api_keys,
    )


# Admin endpoints
@router.get("/", response_model=List[UserResponse], summary="List all users (Admin)")
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> List[UserResponse]:
    """
    List all users (admin only).
    
    Args:
        skip: Number of users to skip
        limit: Maximum number of users to return
        current_user: Current superuser
        db: Database session
        
    Returns:
        List[UserResponse]: List of users
    """
    result = await db.execute(
        select(User).offset(skip).limit(limit)
    )
    users = result.scalars().all()
    
    return [UserResponse.model_validate(user) for user in users]


@router.get("/{user_id}", response_model=UserResponse, summary="Get user by ID (Admin)")
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> UserResponse:
    """
    Get user by ID (admin only).
    
    Args:
        user_id: User ID
        current_user: Current superuser
        db: Database session
        
    Returns:
        UserResponse: User data
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(user)


@router.post("/", response_model=UserResponse, summary="Create user (Admin)")
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> UserResponse:
    """
    Create new user (admin only).
    
    Args:
        user_data: User creation data
        current_user: Current superuser
        db: Database session
        
    Returns:
        UserResponse: Created user data
    """
    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    user = User(
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        bio=user_data.bio,
        avatar_url=user_data.avatar_url,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
        is_verified=True,  # Admin-created users are auto-verified
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserResponse.model_validate(user)


@router.delete("/{user_id}", summary="Delete user (Admin)")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> dict:
    """
    Delete user (admin only).
    
    Args:
        user_id: User ID
        current_user: Current superuser
        db: Database session
        
    Returns:
        dict: Success message
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Don't allow deleting superusers
    if user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete superuser"
        )
    
    await db.delete(user)
    await db.commit()
    
    return {"message": "User deleted successfully"}