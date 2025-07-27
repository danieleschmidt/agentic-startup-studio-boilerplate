"""
Pytest configuration and shared fixtures for Agentic Startup Studio Boilerplate
"""

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Test environment setup
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_env_vars() -> Generator:
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///./test.db",
        "REDIS_URL": "redis://localhost:6379/1",
        "API_SECRET_KEY": "test-secret-key",
        "JWT_SECRET_KEY": "test-jwt-secret",
        "OPENAI_API_KEY": "test-openai-key",
    }):
        yield


@pytest.fixture
def test_db():
    """Create a test database."""
    # This would be implemented with actual database setup
    # For now, using SQLite in-memory database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    # Base.metadata.create_all(bind=engine)
    
    yield TestingSessionLocal
    
    # Cleanup
    # Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_redis():
    """Mock Redis for testing."""
    mock_redis = MagicMock()
    with patch("redis.Redis", return_value=mock_redis):
        yield mock_redis


@pytest.fixture
def mock_openai():
    """Mock OpenAI API for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test AI response"
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_crewai():
    """Mock CrewAI for testing."""
    mock_crew = MagicMock()
    mock_crew.kickoff.return_value = "Test crew output"
    
    with patch("crewai.Crew", return_value=mock_crew):
        yield mock_crew


@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application."""
    # This would import the actual FastAPI app
    # from app.main import app
    # with TestClient(app) as client:
    #     yield client
    
    # For now, create a mock client
    mock_client = MagicMock(spec=TestClient)
    yield mock_client


@pytest.fixture
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async test client."""
    async with httpx.AsyncClient(
        base_url="http://test",
        timeout=10.0
    ) as client:
        yield client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "username": "testuser",
        "is_active": True,
        "is_verified": True,
    }


@pytest.fixture
def sample_project_data():
    """Sample project data for testing."""
    return {
        "id": "test-project-123",
        "name": "Test Project",
        "description": "A test project for unit testing",
        "tech_stack": ["fastapi", "react", "crewai"],
        "status": "active",
    }


@pytest.fixture
def sample_agent_data():
    """Sample agent data for testing."""
    return {
        "id": "test-agent-123",
        "name": "Test Agent",
        "role": "researcher",
        "goal": "Conduct research on given topics",
        "backstory": "An AI agent specialized in research tasks",
        "tools": ["web_search", "file_analysis"],
    }


@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "id": "test-task-123",
        "description": "Research the latest trends in AI",
        "expected_output": "A comprehensive report on AI trends",
        "agent_id": "test-agent-123",
        "status": "pending",
    }


@pytest.fixture
def mock_file_upload():
    """Mock file upload for testing."""
    from io import BytesIO
    
    file_content = b"Test file content"
    file = BytesIO(file_content)
    file.name = "test_file.txt"
    file.content_type = "text/plain"
    
    return file


@pytest.fixture
def mock_keycloak():
    """Mock Keycloak authentication for testing."""
    mock_keycloak = MagicMock()
    mock_keycloak.well_known.return_value = {
        "issuer": "http://localhost:8080/realms/test",
        "authorization_endpoint": "http://localhost:8080/auth",
        "token_endpoint": "http://localhost:8080/token",
    }
    
    with patch("keycloak.KeycloakOpenID", return_value=mock_keycloak):
        yield mock_keycloak


@pytest.fixture
def mock_jwt_token():
    """Mock JWT token for testing."""
    return {
        "access_token": "mock-access-token",
        "token_type": "bearer",
        "expires_in": 3600,
        "refresh_token": "mock-refresh-token",
    }


@pytest.fixture
def authorized_headers(mock_jwt_token):
    """Headers with authorization token for testing."""
    return {
        "Authorization": f"Bearer {mock_jwt_token['access_token']}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def performance_metrics():
    """Performance metrics tracking for tests."""
    metrics = {
        "response_times": [],
        "memory_usage": [],
        "cpu_usage": [],
    }
    return metrics


@pytest.fixture
def security_test_data():
    """Security test data including malicious inputs."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users--",
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        ],
        "command_injection": [
            "; cat /etc/passwd",
            "| whoami",
            "&& rm -rf /",
        ],
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (deselect with '-m \"not unit\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Async test utilities
@pytest.fixture
async def async_test_context():
    """Async context for testing async operations."""
    context = {
        "started_at": asyncio.get_event_loop().time(),
        "tasks": [],
    }
    
    yield context
    
    # Cleanup any remaining tasks
    for task in context["tasks"]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# Database test utilities
@pytest.fixture
def db_transaction():
    """Database transaction that rolls back after test."""
    # This would be implemented with actual database transactions
    # For now, just a placeholder
    transaction = MagicMock()
    yield transaction
    # transaction.rollback()


# Mock external services
@pytest.fixture
def mock_external_apis():
    """Mock all external API calls."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        yield mock_client


# Test data factories
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user(**kwargs):
        """Create a test user."""
        default_data = {
            "id": "test-user-123",
            "email": "test@example.com",
            "username": "testuser",
            "is_active": True,
        }
        default_data.update(kwargs)
        return default_data
    
    @staticmethod
    def create_project(**kwargs):
        """Create a test project."""
        default_data = {
            "id": "test-project-123",
            "name": "Test Project",
            "description": "Test project description",
            "owner_id": "test-user-123",
        }
        default_data.update(kwargs)
        return default_data


@pytest.fixture
def test_factory():
    """Test data factory fixture."""
    return TestDataFactory


# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup():
    """Automatic cleanup after each test."""
    yield
    
    # Cleanup temporary files
    temp_files = Path(".").glob("*.tmp")
    for file in temp_files:
        file.unlink(missing_ok=True)
    
    # Clear any test caches
    # cache.clear()


# Performance testing utilities
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "max_response_time": 0.2,  # 200ms
        "max_memory_usage": 100,   # 100MB
        "iterations": 100,
    }


# Load testing utilities
@pytest.fixture
def load_test_config():
    """Configuration for load testing."""
    return {
        "concurrent_users": 10,
        "ramp_up_time": 5,
        "test_duration": 30,
        "target_rps": 100,
    }