"""
Authentication API endpoints.
Handles user registration, login, token refresh, and profile management.
"""

import logging
from datetime import timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_user
from app.core.config import get_settings
from app.core.database import get_async_db_dependency
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.user import User
from app.services.user_service import UserService

logger = logging.getLogger(__name__)

router = APIRouter()


class UserRegistration(BaseModel):
    """User registration request model."""
    
    email: EmailStr
    username: str
    full_name: str
    password: str
    confirm_password: str
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username must be less than 50 characters')
        if not v.isalnum() and '_' not in v:
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if len(v) > 128:
            raise ValueError('Password must be less than 128 characters')
        
        # Check for basic complexity
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and digit characters')
        
        return v
    
    @validator('confirm_password')
    def validate_confirm_password(cls, v, values):
        """Validate password confirmation."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v


class UserLogin(BaseModel):
    """User login response model."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class TokenRefresh(BaseModel):
    """Token refresh request model."""
    
    refresh_token: str


class PasswordChange(BaseModel):
    """Password change request model."""
    
    current_password: str
    new_password: str
    confirm_new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        has_upper = any(c.isupper() for c in v)
        has_lower = any(c.islower() for c in v)
        has_digit = any(c.isdigit() for c in v)
        
        if not (has_upper and has_lower and has_digit):
            raise ValueError('Password must contain uppercase, lowercase, and digit characters')
        
        return v
    
    @validator('confirm_new_password')
    def validate_confirm_new_password(cls, v, values):
        """Validate new password confirmation."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v


class ProfileUpdate(BaseModel):
    """Profile update request model."""
    
    full_name: str = None
    bio: str = None
    timezone: str = None
    language: str = None
    
    @validator('timezone')
    def validate_timezone(cls, v):
        """Validate timezone format."""
        if v is not None:
            # Basic timezone validation
            import pytz
            try:
                pytz.timezone(v)
            except pytz.UnknownTimeZoneError:
                raise ValueError('Invalid timezone')
        return v
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code."""
        if v is not None:
            valid_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
            if v not in valid_languages:
                raise ValueError(f'Language must be one of: {", ".join(valid_languages)}')
        return v


@router.post("/register", response_model=Dict[str, str])
async def register_user(
    registration: UserRegistration,
    db: AsyncSession = Depends(get_async_db_dependency)
) -> Dict[str, str]:
    """
    Register a new user account.
    
    Args:
        registration: User registration data
        db: Database session
        
    Returns:
        Dict[str, str]: Registration success message
    """
    try:
        user_service = UserService(db)
        
        # Check if user already exists
        existing_user = await user_service.get_user_by_email(registration.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        existing_username = await user_service.get_user_by_username(registration.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Create new user
        user_data = {
            "email": registration.email,
            "username": registration.username,
            "full_name": registration.full_name,
            "hashed_password": get_password_hash(registration.password),
            "is_active": True,
            "is_verified": False,  # Require email verification
        }
        
        new_user = await user_service.create_user(user_data)
        
        logger.info(f"New user registered: {new_user.username} ({new_user.email})")
        
        # TODO: Send verification email
        
        return {
            "message": "User registered successfully. Please check your email for verification.",
            "user_id": str(new_user.id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=UserLogin)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_async_db_dependency)
) -> UserLogin:
    """
    Authenticate user and return access tokens.
    
    Args:
        form_data: Login form data (username/email and password)
        db: Database session
        
    Returns:
        UserLogin: Login response with tokens and user info
    """
    try:
        settings = get_settings()
        user_service = UserService(db)
        
        # Get user by email or username
        user = await user_service.get_user_by_email(form_data.username)
        if not user:
            user = await user_service.get_user_by_username(form_data.username)
        
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email/username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.jwt_expire_minutes)
        access_token = create_access_token(
            subject=str(user.id),
            expires_delta=access_token_expires,
            settings=settings
        )
        
        # Create refresh token
        refresh_token_expires = timedelta(days=settings.jwt_refresh_expire_days)
        refresh_token = create_access_token(
            subject=str(user.id),
            expires_delta=refresh_token_expires,
            settings=settings
        )
        
        # Update last login
        await user_service.update_last_login(user.id)
        
        logger.info(f"User logged in: {user.username}")
        
        return UserLogin(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_expire_minutes * 60,
            user=user.to_dict()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_async_db_dependency)
) -> Dict[str, Any]:
    """
    Refresh access token using refresh token.
    
    Args:
        token_data: Refresh token data
        db: Database session
        
    Returns:
        Dict[str, Any]: New access token
    """
    try:
        settings = get_settings()
        
        # TODO: Implement token refresh logic
        # This would involve validating the refresh token and creating a new access token
        
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Token refresh not implemented yet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user profile information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: User profile data
    """
    return current_user.to_dict()


@router.put("/me", response_model=Dict[str, Any])
async def update_user_profile(
    profile_update: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_dependency)
) -> Dict[str, Any]:
    """
    Update current user profile.
    
    Args:
        profile_update: Profile update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict[str, Any]: Updated user profile
    """
    try:
        user_service = UserService(db)
        
        # Prepare update data
        update_data = {}
        if profile_update.full_name is not None:
            update_data["full_name"] = profile_update.full_name
        if profile_update.bio is not None:
            update_data["bio"] = profile_update.bio
        if profile_update.timezone is not None:
            update_data["timezone"] = profile_update.timezone
        if profile_update.language is not None:
            update_data["language"] = profile_update.language
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields to update"
            )
        
        # Update user
        updated_user = await user_service.update_user(current_user.id, update_data)
        
        logger.info(f"User profile updated: {current_user.username}")
        
        return updated_user.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_dependency)
) -> Dict[str, str]:
    """
    Change user password.
    
    Args:
        password_change: Password change data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict[str, str]: Success message
    """
    try:
        # Verify current password
        if not verify_password(password_change.current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        user_service = UserService(db)
        new_password_hash = get_password_hash(password_change.new_password)
        
        await user_service.update_user(
            current_user.id,
            {"hashed_password": new_password_hash}
        )
        
        logger.info(f"Password changed for user: {current_user.username}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Logout user (invalidate token).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict[str, str]: Logout confirmation
    """
    # TODO: Implement token blacklisting for logout
    # For now, just return success (client should discard token)
    
    logger.info(f"User logged out: {current_user.username}")
    
    return {"message": "Logged out successfully"}


@router.delete("/me")
async def delete_user_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db_dependency)
) -> Dict[str, str]:
    """
    Delete current user account.
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict[str, str]: Deletion confirmation
    """
    try:
        user_service = UserService(db)
        
        # Soft delete - deactivate account
        await user_service.update_user(
            current_user.id,
            {"is_active": False}
        )
        
        logger.info(f"User account deleted: {current_user.username}")
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )