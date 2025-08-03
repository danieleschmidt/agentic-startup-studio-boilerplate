"""
User service for business logic operations.
Handles user management, authentication, and profile operations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import User, APIKey, AgentTask

logger = logging.getLogger(__name__)


class UserService:
    """Service class for user-related operations."""
    
    def __init__(self, db: AsyncSession):
        """
        Initialize user service.
        
        Args:
            db: Database session
        """
        self.db = db
    
    async def create_user(self, user_data: Dict) -> User:
        """
        Create a new user.
        
        Args:
            user_data: User creation data
            
        Returns:
            User: Created user instance
        """
        try:
            user = User(**user_data)
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)
            
            logger.info(f"Created user: {user.username} ({user.email})")
            return user
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to create user: {e}")
            raise
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[User]: User instance if found
        """
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Failed to get user by ID {user_id}: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: Email address
            
        Returns:
            Optional[User]: User instance if found
        """
        try:
            result = await self.db.execute(
                select(User).where(User.email == email.lower())
            )
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: Username
            
        Returns:
            Optional[User]: User instance if found
        """
        try:
            result = await self.db.execute(
                select(User).where(User.username == username.lower())
            )
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    async def update_user(self, user_id: int, update_data: Dict) -> User:
        """
        Update user information.
        
        Args:
            user_id: User ID
            update_data: Data to update
            
        Returns:
            User: Updated user instance
        """
        try:
            # Update user
            await self.db.execute(
                update(User)
                .where(User.id == user_id)
                .values(**update_data)
            )
            
            await self.db.commit()
            
            # Fetch updated user
            updated_user = await self.get_user_by_id(user_id)
            
            logger.info(f"Updated user {user_id}: {list(update_data.keys())}")
            return updated_user
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update user {user_id}: {e}")
            raise
    
    async def update_last_login(self, user_id: int) -> None:
        """
        Update user's last login timestamp.
        
        Args:
            user_id: User ID
        """
        try:
            await self.db.execute(
                update(User)
                .where(User.id == user_id)
                .values(last_login=datetime.utcnow())
            )
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to update last login for user {user_id}: {e}")
    
    async def delete_user(self, user_id: int) -> bool:
        """
        Delete user (soft delete by deactivating).
        
        Args:
            user_id: User ID
            
        Returns:
            bool: Success status
        """
        try:
            await self.db.execute(
                update(User)
                .where(User.id == user_id)
                .values(is_active=False)
            )
            await self.db.commit()
            
            logger.info(f"Deleted (deactivated) user {user_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    async def list_users(
        self, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[User]:
        """
        List users with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            active_only: Whether to return only active users
            
        Returns:
            List[User]: List of user instances
        """
        try:
            query = select(User)
            
            if active_only:
                query = query.where(User.is_active == True)
            
            query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
            
            result = await self.db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    async def search_users(self, query: str, limit: int = 20) -> List[User]:
        """
        Search users by email, username, or full name.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List[User]: List of matching users
        """
        try:
            search_query = select(User).where(
                (User.email.contains(query.lower())) |
                (User.username.contains(query.lower())) |
                (User.full_name.contains(query))
            ).where(User.is_active == True).limit(limit)
            
            result = await self.db.execute(search_query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Failed to search users with query '{query}': {e}")
            return []
    
    async def get_user_with_tasks(self, user_id: int) -> Optional[User]:
        """
        Get user with their agent tasks.
        
        Args:
            user_id: User ID
            
        Returns:
            Optional[User]: User with loaded tasks
        """
        try:
            result = await self.db.execute(
                select(User)
                .options(selectinload(User.agent_tasks))
                .where(User.id == user_id)
            )
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Failed to get user with tasks {user_id}: {e}")
            return None
    
    async def get_user_statistics(self, user_id: int) -> Dict:
        """
        Get user statistics and metrics.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict: User statistics
        """
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return {}
            
            # Get task statistics
            tasks_query = select(AgentTask).where(AgentTask.user_id == user_id)
            tasks_result = await self.db.execute(tasks_query)
            tasks = tasks_result.scalars().all()
            
            # Calculate statistics
            total_tasks = len(tasks)
            completed_tasks = len([t for t in tasks if t.status == "completed"])
            failed_tasks = len([t for t in tasks if t.status == "failed"])
            running_tasks = len([t for t in tasks if t.status == "running"])
            
            # Calculate average execution time
            completed_with_time = [t for t in tasks if t.status == "completed" and t.execution_time]
            avg_execution_time = (
                sum(t.execution_time for t in completed_with_time) / len(completed_with_time)
                if completed_with_time else 0
            )
            
            # Get API key count
            api_keys_query = select(APIKey).where(APIKey.user_id == user_id, APIKey.is_active == True)
            api_keys_result = await self.db.execute(api_keys_query)
            active_api_keys = len(api_keys_result.scalars().all())
            
            return {
                "user_info": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "is_verified": user.is_verified,
                    "timezone": user.timezone,
                    "language": user.language,
                },
                "task_statistics": {
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "running_tasks": running_tasks,
                    "success_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
                    "average_execution_time": round(avg_execution_time, 2),
                },
                "api_usage": {
                    "active_api_keys": active_api_keys,
                },
                "activity": {
                    "days_since_registration": (
                        (datetime.utcnow() - user.created_at).days
                        if user.created_at else 0
                    ),
                    "last_activity": user.last_login.isoformat() if user.last_login else None,
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get user statistics for {user_id}: {e}")
            return {}
    
    async def verify_user_email(self, user_id: int) -> bool:
        """
        Mark user email as verified.
        
        Args:
            user_id: User ID
            
        Returns:
            bool: Success status
        """
        try:
            await self.db.execute(
                update(User)
                .where(User.id == user_id)
                .values(is_verified=True)
            )
            await self.db.commit()
            
            logger.info(f"Verified email for user {user_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to verify email for user {user_id}: {e}")
            return False
    
    async def check_user_permissions(self, user_id: int, permission: str) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            user_id: User ID
            permission: Permission to check
            
        Returns:
            bool: Whether user has permission
        """
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                return False
            
            # Basic permission check based on user status
            if permission == "admin":
                return user.is_superuser
            elif permission == "active":
                return user.is_active
            elif permission == "verified":
                return user.is_verified
            elif permission == "use_api":
                return user.is_active and user.is_verified
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check permissions for user {user_id}: {e}")
            return False
    
    async def update_user_preferences(self, user_id: int, preferences: Dict) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User ID
            preferences: Preferences dictionary
            
        Returns:
            bool: Success status
        """
        try:
            import json
            
            await self.db.execute(
                update(User)
                .where(User.id == user_id)
                .values(preferences=json.dumps(preferences))
            )
            await self.db.commit()
            
            logger.info(f"Updated preferences for user {user_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Failed to update preferences for user {user_id}: {e}")
            return False
    
    async def get_user_preferences(self, user_id: int) -> Dict:
        """
        Get user preferences.
        
        Args:
            user_id: User ID
            
        Returns:
            Dict: User preferences
        """
        try:
            user = await self.get_user_by_id(user_id)
            if not user or not user.preferences:
                return {}
            
            import json
            return json.loads(user.preferences)
            
        except Exception as e:
            logger.error(f"Failed to get preferences for user {user_id}: {e}")
            return {}