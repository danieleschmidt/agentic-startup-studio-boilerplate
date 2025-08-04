"""
API package initialization.
Sets up the main API router and includes all endpoint modules.
"""

from fastapi import APIRouter

from app.api.routes import agents, auth, health, users

# Create main API router
router = APIRouter()

# Include all route modules
router.include_router(
    health.router,
    prefix="/health",
    tags=["health"]
)

router.include_router(
    auth.router,
    prefix="/auth",
    tags=["authentication"]
)

router.include_router(
    users.router,
    prefix="/users",
    tags=["users"]
)

router.include_router(
    agents.router,
    prefix="/agents",
    tags=["agents"]
)