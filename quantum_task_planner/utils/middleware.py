"""
Quantum Task Planner Middleware

FastAPI middleware for security, rate limiting, request validation,
monitoring, and quantum-aware request processing.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
import uuid

from .security import SecurityManager, SecurityConfig, create_default_security_config
from .exceptions import RateLimitError, AuthenticationError, ValidationError
from .logging import get_logger
from .health_checks import get_health_manager, HealthStatus


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware with authentication, rate limiting, and input validation"""
    
    def __init__(self, app: FastAPI, security_config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.security_config = security_config or create_default_security_config()
        self.security_manager = SecurityManager(self.security_config)
        self.logger = get_logger()
        
        # Paths that don't require authentication
        self.public_paths = {
            "/api/v1/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Start timing
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request context
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Log incoming request
        self.logger.logger.info(
            "request_received",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", "")
        )
        
        try:
            # Rate limiting check
            await self._check_rate_limit(request)
            
            # Authentication check (if required)
            if request.url.path not in self.public_paths:
                await self._authenticate_request(request)
            
            # Input sanitization
            await self._sanitize_request(request)
            
            # Process request
            response = await call_next(request)
            
            # Log successful response
            duration_ms = (time.time() - start_time) * 1000
            self.logger.logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration_ms
            )
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except HTTPException as e:
            # Log HTTP exceptions
            duration_ms = (time.time() - start_time) * 1000
            self.logger.logger.warning(
                "request_failed",
                request_id=request_id,
                status_code=e.status_code,
                error=e.detail,
                duration_ms=duration_ms
            )
            raise
            
        except Exception as e:
            # Log unexpected errors
            duration_ms = (time.time() - start_time) * 1000
            self.logger.logger.error(
                "request_error",
                request_id=request_id,
                error_type=type(e).__name__,
                error=str(e),
                duration_ms=duration_ms
            )
            
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return str(request.client.host) if request.client else "unknown"
    
    async def _check_rate_limit(self, request: Request):
        """Check rate limiting"""
        client_ip = self._get_client_ip(request)
        
        if self.security_manager.rate_limiter.is_rate_limited(client_ip):
            retry_after = self.security_manager.rate_limiter.get_retry_after(client_ip)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)}
            )
    
    async def _authenticate_request(self, request: Request):
        """Authenticate request using JWT token"""
        auth_header = request.headers.get("authorization", "")
        
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Authentication token required"
            )
        
        token = auth_header.replace("Bearer ", "")
        
        try:
            payload = self.security_manager.jwt_manager.verify_token(token)
            request.state.current_user = payload
            
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail=str(e)
            )
    
    async def _sanitize_request(self, request: Request):
        """Sanitize request data"""
        # Sanitize query parameters
        if request.query_params:
            sanitized_params = {}
            for key, value in request.query_params.items():
                sanitized_params[key] = self.security_manager.sanitize_input(value)
            request.state.sanitized_params = sanitized_params
        
        # For POST/PUT requests, sanitize body data
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    body = await request.body()
                    if body:
                        data = json.loads(body)
                        sanitized_data = self._sanitize_json_data(data)
                        request.state.sanitized_data = sanitized_data
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON: {str(e)}"
                    )
    
    def _sanitize_json_data(self, data: Any) -> Any:
        """Recursively sanitize JSON data"""
        if isinstance(data, dict):
            return {
                key: self._sanitize_json_data(value)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize_json_data(item) for item in data]
        elif isinstance(data, str):
            return self.security_manager.sanitize_input(data)
        else:
            return data
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"


class QuantumStateMiddleware(BaseHTTPMiddleware):
    """Middleware for quantum state tracking and coherence management"""
    
    def __init__(self, app: FastAPI, coherence_threshold: float = 0.5):
        super().__init__(app)
        self.coherence_threshold = coherence_threshold
        self.request_quantum_states: Dict[str, float] = {}
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Initialize quantum state for request
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Calculate initial quantum coherence based on request complexity
        initial_coherence = self._calculate_initial_coherence(request)
        self.request_quantum_states[request_id] = initial_coherence
        
        request.state.quantum_coherence = initial_coherence
        
        try:
            # Process request with quantum state tracking
            response = await call_next(request)
            
            # Update quantum coherence based on processing
            final_coherence = self._update_coherence_on_success(request_id, response)
            
            # Add quantum state headers
            response.headers["X-Quantum-Coherence"] = str(final_coherence)
            response.headers["X-Request-Quantum-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Apply decoherence on error
            self._apply_error_decoherence(request_id)
            raise
            
        finally:
            # Cleanup quantum state
            self.request_quantum_states.pop(request_id, None)
    
    def _calculate_initial_coherence(self, request: Request) -> float:
        """Calculate initial quantum coherence for request"""
        # Base coherence
        coherence = 1.0
        
        # Reduce coherence for complex operations
        if request.method in ["POST", "PUT", "DELETE"]:
            coherence *= 0.9
        
        # Path complexity
        path_segments = len(request.url.path.split('/'))
        coherence *= max(0.5, 1.0 - (path_segments * 0.05))
        
        # Query parameter complexity
        if request.query_params:
            param_count = len(request.query_params)
            coherence *= max(0.7, 1.0 - (param_count * 0.02))
        
        return max(0.1, coherence)
    
    def _update_coherence_on_success(self, request_id: str, response: Response) -> float:
        """Update coherence based on successful response"""
        current_coherence = self.request_quantum_states.get(request_id, 0.5)
        
        # Maintain or slightly increase coherence on success
        if response.status_code < 300:
            # Successful response maintains coherence
            final_coherence = min(1.0, current_coherence * 1.02)
        elif response.status_code < 400:
            # Redirect - slight coherence loss
            final_coherence = current_coherence * 0.98
        else:
            # Error - coherence loss
            final_coherence = current_coherence * 0.85
        
        self.request_quantum_states[request_id] = final_coherence
        return final_coherence
    
    def _apply_error_decoherence(self, request_id: str):
        """Apply decoherence due to error"""
        current_coherence = self.request_quantum_states.get(request_id, 0.5)
        self.request_quantum_states[request_id] = current_coherence * 0.5


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware for health check integration"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.health_manager = get_health_manager()
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Check if system is healthy before processing non-health requests
        if request.url.path != "/api/v1/health":
            if not self.health_manager.is_healthy():
                overall_status = self.health_manager.overall_status
                
                if overall_status == HealthStatus.CRITICAL:
                    raise HTTPException(
                        status_code=503,
                        detail="System is in critical state"
                    )
                elif overall_status == HealthStatus.UNHEALTHY:
                    raise HTTPException(
                        status_code=503,
                        detail="System is unhealthy"
                    )
                # Allow degraded state to continue with warning header
                elif overall_status == HealthStatus.DEGRADED:
                    response = await call_next(request)
                    response.headers["X-System-Status"] = "degraded"
                    return response
        
        return await call_next(request)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request monitoring"""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.request_counts: Dict[str, int] = {}
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        self.logger = get_logger()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Track request
        endpoint = f"{request.method} {request.url.path}"
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Track response time
            duration = time.time() - start_time
            self.response_times.append(duration)
            
            # Keep only recent response times (last 1000)
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]
            
            # Log metrics
            self.logger.logger.info(
                "request_metrics",
                endpoint=endpoint,
                status_code=response.status_code,
                duration_ms=duration * 1000,
                request_count=self.request_counts[endpoint]
            )
            
            return response
            
        except HTTPException as e:
            # Track HTTP errors
            error_key = f"{endpoint}:{e.status_code}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            duration = time.time() - start_time
            self.logger.logger.warning(
                "request_error_metrics",
                endpoint=endpoint,
                status_code=e.status_code,
                duration_ms=duration * 1000,
                error_count=self.error_counts[error_key]
            )
            
            raise
            
        except Exception as e:
            # Track unexpected errors
            error_key = f"{endpoint}:500"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            duration = time.time() - start_time
            self.logger.logger.error(
                "request_exception_metrics",
                endpoint=endpoint,
                error_type=type(e).__name__,
                duration_ms=duration * 1000,
                error_count=self.error_counts[error_key]
            )
            
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "request_counts": dict(self.request_counts),
            "error_counts": dict(self.error_counts),
            "average_response_time_ms": avg_response_time * 1000,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "p95_response_time_ms": (
                sorted(self.response_times)[int(len(self.response_times) * 0.95)] * 1000
                if self.response_times else 0
            ),
            "p99_response_time_ms": (
                sorted(self.response_times)[int(len(self.response_times) * 0.99)] * 1000
                if self.response_times else 0
            )
        }


class CORSMiddleware:
    """Enhanced CORS middleware with security controls"""
    
    def __init__(self,
                 allowed_origins: List[str] = None,
                 allowed_methods: List[str] = None,
                 allowed_headers: List[str] = None,
                 allow_credentials: bool = True,
                 max_age: int = 600):
        
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or [
            "Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization"
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    def add_cors_headers(self, response: Response, origin: str = None):
        """Add CORS headers to response"""
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        return origin in self.allowed_origins or "*" in self.allowed_origins


def setup_middleware(app: FastAPI, 
                     security_config: Optional[SecurityConfig] = None,
                     enable_quantum_middleware: bool = True,
                     enable_monitoring: bool = True) -> Dict[str, Any]:
    """Setup all middleware components"""
    
    middleware_instances = {}
    
    # Security middleware (should be first)
    security_middleware = SecurityMiddleware(app, security_config)
    app.add_middleware(SecurityMiddleware, security_config=security_config)
    middleware_instances["security"] = security_middleware
    
    # Health check middleware
    health_middleware = HealthCheckMiddleware(app)
    app.add_middleware(HealthCheckMiddleware)
    middleware_instances["health"] = health_middleware
    
    # Quantum state middleware
    if enable_quantum_middleware:
        quantum_middleware = QuantumStateMiddleware(app)
        app.add_middleware(QuantumStateMiddleware)
        middleware_instances["quantum"] = quantum_middleware
    
    # Monitoring middleware
    if enable_monitoring:
        monitoring_middleware = MonitoringMiddleware(app)
        app.add_middleware(MonitoringMiddleware)
        middleware_instances["monitoring"] = monitoring_middleware
    
    return middleware_instances