"""
FastAPI main application entry point.
Configures the FastAPI application with all necessary middleware, routes, and startup logic.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    setup_logging(settings.log_level)
    
    # Initialize database
    await init_db()
    
    yield
    
    # Shutdown
    pass


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="An AI-powered startup built with CrewAI, FastAPI, and React",
        version="0.1.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)
    
    # Authentication middleware
    app.add_middleware(AuthMiddleware)
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unhandled errors."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.environment == "development" else "An unexpected error occurred"
            }
        )
    
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "service": settings.app_name}
    
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )