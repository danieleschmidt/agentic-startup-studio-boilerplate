"""
Authentication middleware for request processing.
"""

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for logging and request tracking.
    """
    
    def __init__(self, app, skip_paths: list[str] = None):
        """
        Initialize auth middleware.
        
        Args:
            app: FastAPI application instance
            skip_paths: List of paths to skip authentication logging
        """
        super().__init__(app)
        self.skip_paths = skip_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add authentication context.
        
        Args:
            request: HTTP request
            call_next: Next middleware or endpoint
            
        Returns:
            Response: HTTP response
        """
        start_time = time.time()
        
        # Skip processing for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Extract authentication info
        auth_header = request.headers.get("authorization")
        user_agent = request.headers.get("user-agent", "unknown")
        client_ip = self._get_client_ip(request)
        
        # Add request context
        request.state.start_time = start_time
        request.state.client_ip = client_ip
        request.state.user_agent = user_agent
        request.state.has_auth = bool(auth_header)
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {client_ip} with auth: {bool(auth_header)}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add response headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = str(id(request))
            
            # Log response
            logger.info(
                f"Response: {response.status_code} for {request.method} "
                f"{request.url.path} in {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"from {client_ip} after {process_time:.3f}s: {e}"
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.
        
        Args:
            request: HTTP request
            
        Returns:
            str: Client IP address
        """
        # Check for forwarded headers (common in reverse proxy setups)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Get the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection IP
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"