"""
Authentication utilities and dependencies for FastAPI.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_db_dependency
from app.core.security import verify_api_key, verify_token
from app.models.api_key import APIKey
from app.models.user import User

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Authorization credentials
        db: Database session
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not credentials:
        raise credentials_exception
    
    # Verify JWT token
    token = credentials.credentials
    subject = verify_token(token, token_type="access")
    
    if subject is None:
        raise credentials_exception
    
    # Get user from database
    try:
        user_id = int(subject)
        user = await db.get(User, user_id)
    except (ValueError, TypeError):
        # Subject might be email instead of ID
        from sqlalchemy import select
        result = await db.execute(select(User).where(User.email == subject))
        user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        User: Current active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current superuser.
    
    Args:
        current_user: Current user from JWT token
        
    Returns:
        User: Current superuser
        
    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def get_user_from_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> tuple[User, APIKey]:
    """
    Get user from API key authentication.
    
    Args:
        credentials: HTTP Authorization credentials
        db: Database session
        
    Returns:
        tuple: (User, APIKey) if authentication succeeds
        
    Raises:
        HTTPException: If API key authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not credentials:
        raise credentials_exception
    
    # Parse API key (format: "key_id:secret_key")
    api_key_parts = credentials.credentials.split(":", 1)
    if len(api_key_parts) != 2:
        raise credentials_exception
    
    key_id, secret_key = api_key_parts
    
    # Get API key from database
    from sqlalchemy import select
    result = await db.execute(
        select(APIKey).where(APIKey.key_id == key_id)
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise credentials_exception
    
    # Verify secret key
    if not verify_api_key(secret_key, api_key.key_hash):
        raise credentials_exception
    
    # Check if API key is valid
    if not api_key.is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is expired or inactive"
        )
    
    # Get associated user
    user = await db.get(User, api_key.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive"
        )
    
    # Record API key usage
    api_key.record_usage()
    await db.commit()
    
    return user, api_key


async def get_user_from_token_or_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_async_db_dependency),
) -> User:
    """
    Get user from either JWT token or API key authentication.
    
    Args:
        credentials: HTTP Authorization credentials
        db: Database session
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication credentials provided",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Try JWT token first
    subject = verify_token(token, token_type="access")
    if subject:
        try:
            user_id = int(subject)
            user = await db.get(User, user_id)
        except (ValueError, TypeError):
            # Subject might be email instead of ID
            from sqlalchemy import select
            result = await db.execute(select(User).where(User.email == subject))
            user = result.scalar_one_or_none()
        
        if user and user.is_active:
            return user
    
    # Try API key authentication
    try:
        user, _ = await get_user_from_api_key(credentials, db)
        return user
    except HTTPException:
        pass
    
    # Both authentication methods failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permissions(*required_permissions: str):
    """
    Decorator to require specific API key permissions.
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        Dependency function
    """
    async def permission_checker(
        user_and_key: tuple[User, APIKey] = Depends(get_user_from_api_key),
    ) -> tuple[User, APIKey]:
        user, api_key = user_and_key
        
        for permission in required_permissions:
            if not api_key.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key does not have required permission: {permission}"
                )
        
        return user, api_key
    
    return permission_checker