"""
Security utilities for password hashing, JWT tokens, and authentication.
"""

import secrets
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        subject: Token subject (usually user ID or email)
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT refresh token.
    
    Args:
        subject: Token subject (usually user ID or email)
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT refresh token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.jwt_refresh_expire_days)
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """
    Verify JWT token and return subject.
    
    Args:
        token: JWT token to verify
        token_type: Expected token type ("access" or "refresh")
        
    Returns:
        str: Token subject if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.jwt_algorithm]
        )
        
        subject: str = payload.get("sub")
        token_type_claim: str = payload.get("type")
        
        if subject is None or token_type_claim != token_type:
            return None
            
        return subject
        
    except JWTError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


def generate_password_reset_token(email: str) -> str:
    """
    Generate password reset token.
    
    Args:
        email: User email
        
    Returns:
        str: Password reset token
    """
    delta = timedelta(hours=1)  # Reset token expires in 1 hour
    expire = datetime.utcnow() + delta
    
    to_encode = {"exp": expire, "sub": email, "type": "password_reset"}
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify password reset token and return email.
    
    Args:
        token: Password reset token
        
    Returns:
        str: Email if token is valid, None otherwise
    """
    return verify_token(token, token_type="password_reset")


def generate_api_key() -> tuple[str, str]:
    """
    Generate API key pair (public key ID and secret key).
    
    Returns:
        tuple: (key_id, secret_key)
    """
    key_id = f"ak_{secrets.token_urlsafe(16)}"
    secret_key = secrets.token_urlsafe(32)
    
    return key_id, secret_key


def hash_api_key(secret_key: str) -> str:
    """
    Hash API secret key for storage.
    
    Args:
        secret_key: API secret key
        
    Returns:
        str: Hashed secret key
    """
    return get_password_hash(secret_key)


def verify_api_key(secret_key: str, hashed_key: str) -> bool:
    """
    Verify API secret key against its hash.
    
    Args:
        secret_key: Plain API secret key
        hashed_key: Hashed API secret key
        
    Returns:
        bool: True if key matches, False otherwise
    """
    return verify_password(secret_key, hashed_key)


def generate_verification_token(user_id: int) -> str:
    """
    Generate email verification token.
    
    Args:
        user_id: User ID
        
    Returns:
        str: Email verification token
    """
    delta = timedelta(days=1)  # Verification token expires in 1 day
    expire = datetime.utcnow() + delta
    
    to_encode = {"exp": expire, "sub": str(user_id), "type": "email_verification"}
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.secret_key, 
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_verification_token(token: str) -> Optional[str]:
    """
    Verify email verification token and return user ID.
    
    Args:
        token: Email verification token
        
    Returns:
        str: User ID if token is valid, None otherwise
    """
    return verify_token(token, token_type="email_verification")


def generate_secure_random_string(length: int = 32) -> str:
    """
    Generate a cryptographically secure random string.
    
    Args:
        length: Length of the string
        
    Returns:
        str: Random string
    """
    return secrets.token_urlsafe(length)