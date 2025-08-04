"""
Rate limiting middleware using Redis for distributed rate limiting.
"""

import json
import time
from typing import Callable, Dict, Optional

import redis.asyncio as redis
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with Redis backend for distributed limiting.
    """
    
    def __init__(
        self,
        app,
        default_rate_limit: int = None,
        window_size: int = 60,
        redis_url: str = None,
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application instance
            default_rate_limit: Default requests per window
            window_size: Time window in seconds
            redis_url: Redis connection URL
        """
        super().__init__(app)
        self.default_rate_limit = default_rate_limit or settings.rate_limit_per_minute
        self.window_size = window_size
        self.redis_url = redis_url or settings.redis_url
        
        # Redis connection pool
        self.redis_pool = None
        
        # Path-specific rate limits
        self.path_limits: Dict[str, int] = {
            "/api/v1/agents/execute": 10,  # Lower limit for expensive operations
            "/api/v1/auth/login": 5,       # Lower limit for auth endpoints
            "/api/v1/auth/register": 3,    # Very low limit for registration
        }
        
        # Paths to skip rate limiting
        self.skip_paths = [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
    
    async def _get_redis(self) -> redis.Redis:
        """
        Get Redis connection with connection pooling.
        
        Returns:
            redis.Redis: Redis connection
        """
        if not self.redis_pool:
            try:
                self.redis_pool = redis.ConnectionPool.from_url(
                    self.redis_url,
                    decode_responses=True,
                    max_connections=20,
                    retry_on_timeout=True,
                )
                logger.info("Redis connection pool created for rate limiting")
            except Exception as e:
                logger.error(f"Failed to create Redis connection pool: {e}")
                # Fall back to in-memory rate limiting
                self.redis_pool = None
        
        if self.redis_pool:
            return redis.Redis(connection_pool=self.redis_pool)
        else:
            # Return None to indicate Redis is not available
            return None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: HTTP request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Skip rate limiting for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limit for this path
        rate_limit = self._get_rate_limit(request.url.path)
        
        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(
            client_id, request.url.path, rate_limit
        )
        
        if not is_allowed:
            # Rate limit exceeded
            logger.warning(
                f"Rate limit exceeded for client {client_id} "
                f"on path {request.url.path}"
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": rate_limit,
                    "window": self.window_size,
                    "reset_time": reset_time,
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(reset_time),
                },
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Args:
            request: HTTP request
            
        Returns:
            str: Client identifier
        """
        # Try to get user ID from request state (set by auth middleware)
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP-based identification
        client_ip = getattr(request.state, "client_ip", None)
        if not client_ip:
            # Extract IP if not set by auth middleware
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()
            elif hasattr(request.client, "host"):
                client_ip = request.client.host
            else:
                client_ip = "unknown"
        
        return f"ip:{client_ip}"
    
    def _get_rate_limit(self, path: str) -> int:
        """
        Get rate limit for specific path.
        
        Args:
            path: Request path
            
        Returns:
            int: Rate limit for the path
        """
        # Check for exact path match
        if path in self.path_limits:
            return self.path_limits[path]
        
        # Check for path prefix matches
        for path_pattern, limit in self.path_limits.items():
            if path.startswith(path_pattern):
                return limit
        
        return self.default_rate_limit
    
    async def _check_rate_limit(
        self, client_id: str, path: str, limit: int
    ) -> tuple[bool, int, int]:
        """
        Check rate limit for client and path.
        
        Args:
            client_id: Client identifier
            path: Request path
            limit: Rate limit
            
        Returns:
            tuple: (is_allowed, remaining_requests, reset_time)
        """
        redis_client = await self._get_redis()
        
        if redis_client:
            return await self._check_rate_limit_redis(
                redis_client, client_id, path, limit
            )
        else:
            # Fall back to in-memory rate limiting (less accurate but functional)
            return await self._check_rate_limit_memory(client_id, path, limit)
    
    async def _check_rate_limit_redis(
        self, redis_client: redis.Redis, client_id: str, path: str, limit: int
    ) -> tuple[bool, int, int]:
        """
        Check rate limit using Redis.
        
        Args:
            redis_client: Redis client
            client_id: Client identifier
            path: Request path
            limit: Rate limit
            
        Returns:
            tuple: (is_allowed, remaining_requests, reset_time)
        """
        current_time = int(time.time())
        window_start = current_time - (current_time % self.window_size)
        window_end = window_start + self.window_size
        
        key = f"rate_limit:{client_id}:{path}:{window_start}"
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_size)
            results = await pipe.execute()
            
            current_count = results[0]
            
            remaining = max(0, limit - current_count)
            is_allowed = current_count <= limit
            
            return is_allowed, remaining, window_end
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fall back to allowing the request
            return True, limit - 1, window_end
    
    # In-memory fallback (simplified implementation)
    _memory_cache: Dict[str, Dict] = {}
    
    async def _check_rate_limit_memory(
        self, client_id: str, path: str, limit: int
    ) -> tuple[bool, int, int]:
        """
        Check rate limit using in-memory storage (fallback).
        
        Args:
            client_id: Client identifier
            path: Request path
            limit: Rate limit
            
        Returns:
            tuple: (is_allowed, remaining_requests, reset_time)
        """
        current_time = int(time.time())
        window_start = current_time - (current_time % self.window_size)
        window_end = window_start + self.window_size
        
        key = f"{client_id}:{path}:{window_start}"
        
        # Clean old entries
        self._cleanup_memory_cache(current_time)
        
        # Get or create entry
        if key not in self._memory_cache:
            self._memory_cache[key] = {
                "count": 0,
                "expires": window_end,
            }
        
        entry = self._memory_cache[key]
        entry["count"] += 1
        
        remaining = max(0, limit - entry["count"])
        is_allowed = entry["count"] <= limit
        
        return is_allowed, remaining, window_end
    
    def _cleanup_memory_cache(self, current_time: int) -> None:
        """
        Clean up expired entries from memory cache.
        
        Args:
            current_time: Current timestamp
        """
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if entry["expires"] < current_time
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]