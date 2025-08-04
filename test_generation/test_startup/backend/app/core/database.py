"""
Database configuration and session management.
Handles SQLAlchemy setup, connection pooling, and async operations.
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# SQLAlchemy metadata with naming convention for constraints
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

# Base class for all SQLAlchemy models
Base = declarative_base(metadata=metadata)

# Global database engines and sessions
engine: Optional[object] = None
async_engine: Optional[object] = None
SessionLocal: Optional[sessionmaker] = None
AsyncSessionLocal: Optional[async_sessionmaker] = None


def create_database_engines():
    """
    Create database engines with proper configuration.
    """
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    
    settings = get_settings()
    
    # Sync engine configuration
    engine_kwargs = {
        "echo": settings.database_echo,
        "pool_pre_ping": True,
        "pool_recycle": 300,  # Recycle connections every 5 minutes
    }
    
    # Configure connection pooling based on environment
    if settings.environment == "production":
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 30,
        })
    else:
        engine_kwargs.update({
            "poolclass": QueuePool,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
        })
    
    # Create synchronous engine
    sync_url = settings.database_url
    if sync_url.startswith("postgresql+asyncpg://"):
        sync_url = sync_url.replace("postgresql+asyncpg://", "postgresql://")
    
    engine = create_engine(sync_url, **engine_kwargs)
    
    # Create asynchronous engine
    async_url = settings.database_url
    if not async_url.startswith("postgresql+asyncpg://"):
        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://")
    
    async_engine_kwargs = engine_kwargs.copy()
    async_engine_kwargs.pop("poolclass", None)  # asyncpg handles pooling differently
    
    async_engine = create_async_engine(async_url, **async_engine_kwargs)
    
    # Create session factories
    SessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    
    # Add event listeners for connection management
    setup_event_listeners()
    
    logger.info("Database engines created successfully")


def setup_event_listeners():
    """
    Setup database event listeners for connection management and logging.
    """
    if not engine:
        return
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Set SQLite pragmas for better performance and reliability."""
        if "sqlite" in str(dbapi_connection):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.close()
    
    @event.listens_for(engine, "checkout")
    def log_connection_checkout(dbapi_connection, connection_record, connection_proxy):
        """Log connection checkout for monitoring."""
        logger.debug("Connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def log_connection_checkin(dbapi_connection, connection_record):
        """Log connection checkin for monitoring."""
        logger.debug("Connection checked in to pool")


async def init_db():
    """
    Initialize database with tables and initial data.
    """
    try:
        # Create engines if not already created
        if not engine or not async_engine:
            create_database_engines()
        
        # Create all tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
        # Run initial data migration if needed
        await create_initial_data()
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def create_initial_data():
    """
    Create initial data required for the application.
    """
    try:
        async with get_async_session() as session:
            # Import models here to avoid circular imports
            from app.models.user import User
            
            # Check if admin user exists
            admin_user = await session.get(User, 1)
            if not admin_user:
                # Create default admin user
                from app.core.security import get_password_hash
                
                admin_user = User(
                    id=1,
                    email="admin@test_startup.com",
                    username="admin",
                    full_name="System Administrator",
                    hashed_password=get_password_hash("admin123"),
                    is_active=True,
                    is_superuser=True,
                )
                
                session.add(admin_user)
                await session.commit()
                logger.info("Created default admin user")
        
    except Exception as e:
        logger.warning(f"Failed to create initial data: {e}")


def get_session() -> SessionLocal:
    """
    Get a synchronous database session.
    
    Returns:
        SessionLocal: Database session
    """
    if not SessionLocal:
        create_database_engines()
    
    return SessionLocal()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an asynchronous database session.
    
    Yields:
        AsyncSession: Async database session
    """
    if not AsyncSessionLocal:
        create_database_engines()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_db_dependency():
    """
    FastAPI dependency for getting database session.
    
    Yields:
        Session: Database session
    """
    db = get_session()
    try:
        yield db
    finally:
        db.close()


async def get_async_db_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting async database session.
    
    Yields:
        AsyncSession: Async database session
    """
    async for session in get_async_session():
        yield session


class DatabaseManager:
    """
    Database management utilities for migrations, backups, and maintenance.
    """
    
    @staticmethod
    async def health_check() -> dict:
        """
        Check database health and return status information.
        
        Returns:
            dict: Health status information
        """
        try:
            async with get_async_session() as session:
                result = await session.execute("SELECT 1")
                result.scalar()
                
                # Get connection pool status
                pool_status = {
                    "size": async_engine.pool.size(),
                    "checked_in": async_engine.pool.checkedin(),
                    "checked_out": async_engine.pool.checkedout(),
                    "invalid": async_engine.pool.invalid(),
                }
                
                return {
                    "status": "healthy",
                    "database": "connected",
                    "pool": pool_status,
                }
        
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
            }
    
    @staticmethod
    async def get_db_info() -> dict:
        """
        Get database information and statistics.
        
        Returns:
            dict: Database information
        """
        try:
            async with get_async_session() as session:
                # Get PostgreSQL version
                version_result = await session.execute("SELECT version()")
                version = version_result.scalar()
                
                # Get database size
                size_result = await session.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                size = size_result.scalar()
                
                # Get table count
                table_result = await session.execute(
                    """
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    """
                )
                table_count = table_result.scalar()
                
                return {
                    "version": version,
                    "size": size,
                    "table_count": table_count,
                    "url": get_settings().database_url.split("@")[1] if "@" in get_settings().database_url else "unknown",
                }
        
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def backup_database(backup_path: str) -> bool:
        """
        Create a database backup.
        
        Args:
            backup_path: Path where backup should be stored
            
        Returns:
            bool: Success status
        """
        try:
            import subprocess
            import os
            
            settings = get_settings()
            
            # Extract database connection info
            db_url = settings.database_url
            if "postgresql://" in db_url:
                # Parse PostgreSQL URL
                # Format: postgresql://user:password@host:port/database
                url_parts = db_url.replace("postgresql://", "").split("/")
                db_name = url_parts[1]
                user_host = url_parts[0].split("@")
                user_pass = user_host[0].split(":")
                host_port = user_host[1].split(":")
                
                user = user_pass[0]
                password = user_pass[1] if len(user_pass) > 1 else ""
                host = host_port[0]
                port = host_port[1] if len(host_port) > 1 else "5432"
                
                # Set environment for pg_dump
                env = os.environ.copy()
                if password:
                    env["PGPASSWORD"] = password
                
                # Run pg_dump
                cmd = [
                    "pg_dump",
                    "-h", host,
                    "-p", port,
                    "-U", user,
                    "-d", db_name,
                    "-f", backup_path,
                    "--verbose",
                ]
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"Database backup created successfully: {backup_path}")
                    return True
                else:
                    logger.error(f"Database backup failed: {result.stderr}")
                    return False
            
            else:
                logger.error("Unsupported database type for backup")
                return False
        
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()