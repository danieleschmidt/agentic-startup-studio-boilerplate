"""
Health check endpoints for monitoring and status.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
import redis.asyncio as redis

from app.core.config import get_settings
from app.core.database import db_manager
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()


@router.get("/health", summary="Basic health check")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "{{cookiecutter.version}}",
        "environment": settings.environment,
    }


@router.get("/health/detailed", summary="Detailed health check")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with dependency status.
    
    Returns:
        dict: Detailed health status
    """
    start_time = time.time()
    
    # Check database health
    db_health = await db_manager.health_check()
    
    # Check Redis health
    redis_health = await _check_redis_health()
    
    # Check external services
    external_health = await _check_external_services()
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Determine overall status
    overall_status = "healthy"
    if (db_health.get("status") != "healthy" or 
        redis_health.get("status") != "healthy"):
        overall_status = "degraded"
    
    if any(service.get("status") == "unhealthy" 
           for service in external_health.values()):
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "{{cookiecutter.version}}",
        "environment": settings.environment,
        "response_time": f"{response_time:.3f}s",
        "dependencies": {
            "database": db_health,
            "redis": redis_health,
            "external_services": external_health,
        },
        "system_info": {
            "app_name": settings.app_name,
            "debug": settings.debug,
            "log_level": settings.log_level,
        },
    }


@router.get("/health/readiness", summary="Readiness probe")
async def readiness_check() -> Dict[str, Any]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns:
        dict: Readiness status
        
    Raises:
        HTTPException: If not ready to serve traffic
    """
    # Check critical dependencies
    db_health = await db_manager.health_check()
    
    if db_health.get("status") != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not available"
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/liveness", summary="Liveness probe")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes liveness probe endpoint.
    
    Returns:
        dict: Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time(),
    }


async def _check_redis_health() -> Dict[str, Any]:
    """
    Check Redis connection health.
    
    Returns:
        dict: Redis health status
    """
    try:
        redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=5,
        )
        
        # Test Redis connection
        await redis_client.ping()
        
        # Get Redis info
        info = await redis_client.info()
        
        await redis_client.close()
        
        return {
            "status": "healthy",
            "version": info.get("redis_version", "unknown"),
            "memory_usage": info.get("used_memory_human", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
        }
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def _check_external_services() -> Dict[str, Dict[str, Any]]:
    """
    Check external service health.
    
    Returns:
        dict: External service health status
    """
    services = {}
    
    # Check OpenAI API if configured
    if settings.openai_api_key:
        services["openai"] = await _check_openai_health()
    
    # Check Anthropic API if configured
    if settings.anthropic_api_key:
        services["anthropic"] = await _check_anthropic_health()
    
    return services


async def _check_openai_health() -> Dict[str, Any]:
    """
    Check OpenAI API health.
    
    Returns:
        dict: OpenAI API health status
    """
    try:
        import openai
        
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Make a simple API call with timeout
        response = await asyncio.wait_for(
            client.models.list(),
            timeout=10.0
        )
        
        return {
            "status": "healthy",
            "models_available": len(response.data) if response.data else 0,
        }
        
    except asyncio.TimeoutError:
        return {
            "status": "unhealthy",
            "error": "API request timeout",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def _check_anthropic_health() -> Dict[str, Any]:
    """
    Check Anthropic API health.
    
    Returns:
        dict: Anthropic API health status
    """
    try:
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        
        # Note: Anthropic doesn't have a simple health check endpoint
        # So we'll just verify the client can be created
        return {
            "status": "healthy",
            "note": "API key configured",
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/health/metrics", summary="Application metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    """
    Application metrics endpoint.
    
    Returns:
        dict: Application metrics
    """
    # Get database info
    db_info = await db_manager.get_db_info()
    
    # Get system metrics
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_info,
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            },
        },
        "application": {
            "environment": settings.environment,
            "debug": settings.debug,
            "log_level": settings.log_level,
        },
    }