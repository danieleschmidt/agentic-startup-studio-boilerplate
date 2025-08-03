"""
Pytest configuration and shared fixtures for testing.
Provides database setup, authentication, and common test utilities.
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings, get_settings
from app.core.database import Base, get_async_db_dependency, get_db_dependency
from app.main import app
from app.models.user import User
from app.core.security import get_password_hash


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"
TEST_DATABASE_URL_SYNC = "sqlite:///./test.db"


class TestSettings(Settings):
    """Test-specific settings override."""
    
    database_url: str = TEST_DATABASE_URL
    database_echo: bool = False
    secret_key: str = "test-secret-key-for-testing-only"
    environment: str = "testing"
    debug: bool = True
    
    class Config:
        env_file = None


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    """Test settings fixture."""
    return TestSettings()


@pytest.fixture(scope="session")
async def test_engine(test_settings):
    """Create test database engine."""
    engine = create_async_engine(
        test_settings.database_url,
        echo=test_settings.database_echo,
        future=True,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Drop all tables and close engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture(scope="session")
def sync_test_engine(test_settings):
    """Create synchronous test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL_SYNC,
        echo=test_settings.database_echo,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest_asyncio.fixture
async def async_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async_session_maker = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        autoflush=False,
        autocommit=False,
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def sync_session(sync_test_engine):
    """Create synchronous database session for testing."""
    Session = sessionmaker(bind=sync_test_engine)
    session = Session()
    
    yield session
    
    session.rollback()
    session.close()


@pytest.fixture
def override_get_settings(test_settings):
    """Override settings dependency for testing."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    app.dependency_overrides.clear()


@pytest.fixture
def override_get_db(sync_session):
    """Override database dependency for testing."""
    def _get_test_db():
        yield sync_session
    
    app.dependency_overrides[get_db_dependency] = _get_test_db
    yield sync_session
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def override_get_async_db(async_session):
    """Override async database dependency for testing."""
    async def _get_test_async_db():
        yield async_session
    
    app.dependency_overrides[get_async_db_dependency] = _get_test_async_db
    yield async_session
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_get_settings, override_get_db) -> Generator[TestClient, None, None]:
    """Create test client for synchronous testing."""
    with TestClient(app) as test_client:
        yield test_client


@pytest_asyncio.fixture
async def async_client(override_get_settings, override_get_async_db) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client for asynchronous testing."""
    async with AsyncClient(app=app, base_url="http://test") as test_client:
        yield test_client


@pytest_asyncio.fixture
async def test_user(async_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        full_name="Test User",
        hashed_password=get_password_hash("testpassword"),
        is_active=True,
        is_verified=True,
    )
    
    async_session.add(user)
    await async_session.commit()
    await async_session.refresh(user)
    
    return user


@pytest_asyncio.fixture
async def test_admin_user(async_session: AsyncSession) -> User:
    """Create a test admin user."""
    admin_user = User(
        email="admin@example.com",
        username="admin",
        full_name="Admin User",
        hashed_password=get_password_hash("adminpassword"),
        is_active=True,
        is_verified=True,
        is_superuser=True,
    )
    
    async_session.add(admin_user)
    await async_session.commit()
    await async_session.refresh(admin_user)
    
    return admin_user


@pytest.fixture
def test_user_token(test_user: User, test_settings: TestSettings) -> str:
    """Create JWT token for test user."""
    from app.core.security import create_access_token
    
    access_token = create_access_token(
        subject=str(test_user.id),
        settings=test_settings
    )
    return access_token


@pytest.fixture
def test_admin_token(test_admin_user: User, test_settings: TestSettings) -> str:
    """Create JWT token for test admin user."""
    from app.core.security import create_access_token
    
    access_token = create_access_token(
        subject=str(test_admin_user.id),
        settings=test_settings
    )
    return access_token


@pytest.fixture
def auth_headers(test_user_token: str) -> dict:
    """Create authorization headers for test user."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def admin_auth_headers(test_admin_token: str) -> dict:
    """Create authorization headers for test admin user."""
    return {"Authorization": f"Bearer {test_admin_token}"}


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"test content")
        tmp.flush()
        yield tmp.name
    
    # Clean up
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mock response from OpenAI API for testing purposes."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }


@pytest.fixture
def sample_agent_request():
    """Sample agent request data for testing."""
    return {
        "task_description": "Analyze the current market trends in artificial intelligence",
        "priority": "normal",
        "context": {
            "industry": "technology",
            "timeframe": "last 6 months"
        }
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "newuser@example.com",
        "username": "newuser",
        "full_name": "New User",
        "password": "newpassword123",
        "bio": "This is a test user",
    }


class TestDataFactory:
    """Factory class for creating test data."""
    
    @staticmethod
    def create_user_data(**kwargs):
        """Create user data with defaults."""
        default_data = {
            "email": "factory@example.com",
            "username": "factoryuser",
            "full_name": "Factory User",
            "password": "factorypassword123",
            "is_active": True,
            "is_verified": False,
        }
        default_data.update(kwargs)
        return default_data
    
    @staticmethod
    def create_agent_request_data(**kwargs):
        """Create agent request data with defaults."""
        default_data = {
            "task_description": "Default test task description",
            "priority": "normal",
            "context": {"test": True}
        }
        default_data.update(kwargs)
        return default_data


@pytest.fixture
def test_data_factory():
    """Test data factory fixture."""
    return TestDataFactory()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


@pytest.fixture(autouse=True)
def clean_db(async_session):
    """Clean database after each test."""
    yield
    # Cleanup happens automatically due to session rollback


# Mock external services
@pytest.fixture
def mock_external_services(monkeypatch):
    """Mock external services for testing."""
    
    # Mock OpenAI API
    def mock_openai_create(*args, **kwargs):
        return {
            "choices": [{"message": {"content": "Mocked OpenAI response"}}],
            "usage": {"total_tokens": 100}
        }
    
    # Mock email sending
    def mock_send_email(*args, **kwargs):
        return True
    
    # Mock Redis operations
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def get(self, key):
            return self.data.get(key)
        
        async def set(self, key, value, ex=None):
            self.data[key] = value
            return True
        
        async def delete(self, key):
            return self.data.pop(key, None) is not None
    
    monkeypatch.setattr("openai.ChatCompletion.create", mock_openai_create)
    monkeypatch.setattr("app.core.email.send_email", mock_send_email)
    
    yield {
        "openai": mock_openai_create,
        "email": mock_send_email,
        "redis": MockRedis(),
    }