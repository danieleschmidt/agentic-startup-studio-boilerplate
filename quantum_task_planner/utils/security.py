"""
Quantum Task Planner Security

Comprehensive security utilities including authentication, authorization,
input sanitization, rate limiting, and quantum-safe cryptography.
"""

import asyncio
import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from functools import wraps
import re
import html
import json
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from .exceptions import (
    AuthenticationError, 
    AuthorizationError, 
    RateLimitError, 
    TaskValidationError,
    SecurityError
)


class SecurityError(Exception):
    """Base security exception"""
    pass


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 3600
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    encryption_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()


class PasswordValidator:
    """Password validation utilities"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password strength against policy"""
        if len(password) < self.config.password_min_length:
            raise TaskValidationError(
                "password", 
                "***", 
                f"Must be at least {self.config.password_min_length} characters"
            )
        
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            raise TaskValidationError("password", "***", "Must contain uppercase letters")
        
        if self.config.password_require_lowercase and not re.search(r'[a-z]', password):
            raise TaskValidationError("password", "***", "Must contain lowercase letters")
        
        if self.config.password_require_numbers and not re.search(r'\d', password):
            raise TaskValidationError("password", "***", "Must contain numbers")
        
        if self.config.password_require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise TaskValidationError("password", "***", "Must contain special symbols")
        
        # Check for common weak passwords
        weak_patterns = [
            r'password', r'123456', r'qwerty', r'admin', r'root',
            r'user', r'test', r'guest', r'demo', r'default'
        ]
        
        password_lower = password.lower()
        for pattern in weak_patterns:
            if re.search(pattern, password_lower):
                raise TaskValidationError("password", "***", "Password is too common or weak")
        
        return True
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        self.validate_password_strength(password)
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(password, hashed)


class InputSanitizer:
    """Input sanitization and validation"""
    
    @staticmethod
    def sanitize_html(input_text: str) -> str:
        """Sanitize HTML input to prevent XSS"""
        if not isinstance(input_text, str):
            return str(input_text)
        
        # HTML encode dangerous characters
        sanitized = html.escape(input_text, quote=True)
        
        # Remove potentially dangerous sequences
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'on\w+\s*=',  # Event handlers
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @staticmethod
    def sanitize_sql(input_text: str) -> str:
        """Sanitize SQL input to prevent injection"""
        if not isinstance(input_text, str):
            return str(input_text)
        
        # Remove SQL injection patterns
        sql_patterns = [
            r"';\s*drop\s+table",
            r"';\s*delete\s+from",
            r"';\s*update\s+",
            r"';\s*insert\s+into",
            r"union\s+select",
            r"';\s*exec\s*\(",
            r"';\s*execute\s*\(",
            r"--",
            r"/\*.*?\*/",
            r"'.*?or.*?'.*?=.*?'",
        ]
        
        sanitized = input_text
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @staticmethod
    def sanitize_command(input_text: str) -> str:
        """Sanitize command input to prevent injection"""
        if not isinstance(input_text, str):
            return str(input_text)
        
        # Remove command injection patterns
        cmd_patterns = [
            r'[;&|`$]',
            r'\$\(',
            r'`.*?`',
            r'\|\s*\w+',
            r'&&\s*\w+',
            r';;\s*\w+',
            r'>\s*/dev/',
            r'<\s*/dev/',
            r'/bin/',
            r'/usr/bin/',
            r'/sbin/',
            r'sudo\s+',
            r'su\s+',
        ]
        
        sanitized = input_text
        for pattern in cmd_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    @staticmethod
    def validate_file_path(file_path: str) -> str:
        """Validate and sanitize file paths"""
        if not isinstance(file_path, str):
            raise TaskValidationError("file_path", file_path, "Must be a string")
        
        # Remove directory traversal attempts
        sanitized = file_path.replace('..', '').replace('//', '/')
        
        # Check for suspicious patterns
        dangerous_paths = [
            '/etc/', '/bin/', '/usr/bin/', '/sbin/', '/usr/sbin/',
            '/proc/', '/sys/', '/dev/', '/tmp/', '/var/log/',
            'passwd', 'shadow', '.ssh/', '.config/'
        ]
        
        path_lower = sanitized.lower()
        for dangerous in dangerous_paths:
            if dangerous in path_lower:
                raise TaskValidationError("file_path", file_path, "Access to system paths not allowed")
        
        # Only allow alphanumeric, dash, underscore, dot, slash
        if not re.match(r'^[a-zA-Z0-9._/-]+$', sanitized):
            raise TaskValidationError("file_path", file_path, "Invalid characters in file path")
        
        return sanitized


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.requests: Dict[str, List[float]] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window_seconds
        
        # Clean up old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries()
            self.last_cleanup = current_time
        
        # Get request history for this identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        request_times = self.requests[identifier]
        
        # Remove requests outside the window
        self.requests[identifier] = [
            req_time for req_time in request_times 
            if req_time > window_start
        ]
        
        # Check if rate limit exceeded
        if len(self.requests[identifier]) >= self.config.rate_limit_requests:
            return True
        
        # Record this request
        self.requests[identifier].append(current_time)
        return False
    
    def _cleanup_old_entries(self):
        """Clean up old rate limit entries"""
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window_seconds
        
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
            
            # Remove empty entries
            if not self.requests[identifier]:
                del self.requests[identifier]
    
    def get_retry_after(self, identifier: str) -> int:
        """Get retry-after time in seconds"""
        if identifier not in self.requests or not self.requests[identifier]:
            return 0
        
        oldest_request = min(self.requests[identifier])
        retry_after = int(oldest_request + self.config.rate_limit_window_seconds - time.time())
        return max(0, retry_after)


class JWTManager:
    """JWT token management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blacklisted_tokens: set = set()
    
    def create_token(self, user_id: str, permissions: List[str] = None, 
                    custom_claims: Dict[str, Any] = None) -> str:
        """Create JWT token"""
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "iat": now,
            "exp": now + timedelta(minutes=self.config.jwt_expiration_minutes),
            "jti": secrets.token_urlsafe(16)  # Token ID for blacklisting
        }
        
        if custom_claims:
            payload.update(custom_claims)
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise AuthenticationError("Token has been revoked")
            
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    def blacklist_token(self, token: str):
        """Blacklist a token"""
        self.blacklisted_tokens.add(token)
    
    def refresh_token(self, token: str) -> str:
        """Refresh JWT token"""
        payload = self.verify_token(token)
        
        # Create new token with same claims
        return self.create_token(
            user_id=payload["user_id"],
            permissions=payload.get("permissions", []),
            custom_claims={k: v for k, v in payload.items() 
                          if k not in ["user_id", "permissions", "iat", "exp", "jti"]}
        )


class QuantumSafeCrypto:
    """Quantum-safe cryptography utilities"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet = Fernet(config.encryption_key.encode())
    
    def encrypt_data(self, data: Union[str, bytes]) -> str:
        """Encrypt data using quantum-safe encryption"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            raise SecurityError(f"Decryption failed: {str(e)}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_data(self, data: str, salt: str = None) -> Tuple[str, str]:
        """Hash data with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hash_obj.hex(), salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify hashed data"""
        computed_hash, _ = self.hash_data(data, salt)
        return hmac.compare_digest(hash_value, computed_hash)


class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_validator = PasswordValidator(config)
        self.rate_limiter = RateLimiter(config)
        self.jwt_manager = JWTManager(config)
        self.crypto = QuantumSafeCrypto(config)
        self.failed_logins: Dict[str, List[float]] = {}
    
    def authenticate_user(self, username: str, password: str, user_hash: str) -> Dict[str, Any]:
        """Authenticate user with rate limiting and lockout"""
        # Check if user is locked out
        if self._is_user_locked_out(username):
            lockout_time = self._get_lockout_remaining(username)
            raise AuthenticationError(f"Account locked. Try again in {lockout_time} minutes.")
        
        # Verify password
        if not self.password_validator.verify_password(password, user_hash):
            self._record_failed_login(username)
            raise AuthenticationError("Invalid credentials")
        
        # Clear failed login attempts on successful login
        if username in self.failed_logins:
            del self.failed_logins[username]
        
        # Create JWT token
        token = self.jwt_manager.create_token(username)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.config.jwt_expiration_minutes * 60
        }
    
    def _is_user_locked_out(self, username: str) -> bool:
        """Check if user is locked out due to failed attempts"""
        if username not in self.failed_logins:
            return False
        
        current_time = time.time()
        lockout_window = current_time - (self.config.lockout_duration_minutes * 60)
        
        # Clean up old failed attempts
        self.failed_logins[username] = [
            attempt_time for attempt_time in self.failed_logins[username]
            if attempt_time > lockout_window
        ]
        
        return len(self.failed_logins[username]) >= self.config.max_login_attempts
    
    def _record_failed_login(self, username: str):
        """Record failed login attempt"""
        current_time = time.time()
        
        if username not in self.failed_logins:
            self.failed_logins[username] = []
        
        self.failed_logins[username].append(current_time)
    
    def _get_lockout_remaining(self, username: str) -> int:
        """Get remaining lockout time in minutes"""
        if username not in self.failed_logins or not self.failed_logins[username]:
            return 0
        
        oldest_attempt = min(self.failed_logins[username])
        lockout_end = oldest_attempt + (self.config.lockout_duration_minutes * 60)
        remaining_seconds = max(0, lockout_end - time.time())
        
        return int(remaining_seconds / 60) + 1
    
    def sanitize_input(self, input_data: Any, input_type: str = "general") -> Any:
        """Sanitize input based on type"""
        if not isinstance(input_data, str):
            return input_data
        
        if input_type == "html":
            return InputSanitizer.sanitize_html(input_data)
        elif input_type == "sql":
            return InputSanitizer.sanitize_sql(input_data)
        elif input_type == "command":
            return InputSanitizer.sanitize_command(input_data)
        elif input_type == "file_path":
            return InputSanitizer.validate_file_path(input_data)
        else:
            # General sanitization
            return InputSanitizer.sanitize_html(input_data)


# Security decorators
def require_authentication(security_manager: SecurityManager):
    """Decorator to require JWT authentication"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract token from request headers (simplified)
            token = kwargs.get('authorization', '').replace('Bearer ', '')
            if not token:
                raise AuthenticationError("Authentication token required")
            
            try:
                payload = security_manager.jwt_manager.verify_token(token)
                kwargs['current_user'] = payload
                return await func(*args, **kwargs)
            except Exception as e:
                raise AuthenticationError(str(e))
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            token = kwargs.get('authorization', '').replace('Bearer ', '')
            if not token:
                raise AuthenticationError("Authentication token required")
            
            try:
                payload = security_manager.jwt_manager.verify_token(token)
                kwargs['current_user'] = payload
                return func(*args, **kwargs)
            except Exception as e:
                raise AuthenticationError(str(e))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def require_permission(permission: str, security_manager: SecurityManager):
    """Decorator to require specific permission"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user', {})
            user_permissions = current_user.get('permissions', [])
            
            if permission not in user_permissions and 'admin' not in user_permissions:
                raise AuthorizationError("Insufficient permissions", permission)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(identifier_func: Callable = None, security_manager: SecurityManager = None):
    """Decorator for rate limiting"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (IP, user ID, etc.)
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = kwargs.get('client_ip', 'unknown')
            
            if security_manager and security_manager.rate_limiter.is_rate_limited(identifier):
                retry_after = security_manager.rate_limiter.get_retry_after(identifier)
                raise RateLimitError(
                    security_manager.config.rate_limit_requests,
                    security_manager.config.rate_limit_window_seconds,
                    retry_after
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_inputs(input_types: Dict[str, str] = None):
    """Decorator to sanitize function inputs"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if input_types:
                for param_name, input_type in input_types.items():
                    if param_name in kwargs:
                        kwargs[param_name] = InputSanitizer.sanitize_html(kwargs[param_name])
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Default security configuration
def create_default_security_config() -> SecurityConfig:
    """Create default security configuration"""
    return SecurityConfig(
        jwt_secret_key=secrets.token_urlsafe(32),
        jwt_algorithm="HS256",
        jwt_expiration_minutes=30,
        password_min_length=8,
        password_require_uppercase=True,
        password_require_lowercase=True,
        password_require_numbers=True,
        password_require_symbols=True,
        rate_limit_requests=100,
        rate_limit_window_seconds=3600,
        max_login_attempts=5,
        lockout_duration_minutes=15,
        encryption_key=Fernet.generate_key().decode()
    )