# Testing Guide

This guide covers testing strategies, frameworks, and best practices for the Agentic Startup Studio Boilerplate.

## Overview

Our testing strategy follows the testing pyramid approach:
- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Tests for component interactions and API endpoints
- **End-to-End Tests**: Full system tests simulating user workflows
- **Performance Tests**: Load testing and performance benchmarks
- **Security Tests**: Vulnerability scanning and security validation

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests
│   ├── test_backlog_manager.py
│   └── test_*.py
├── integration/             # Integration tests
│   ├── test_api_endpoints.py
│   └── test_*.py
├── e2e/                     # End-to-end tests
│   ├── test_user_workflows.py
│   ├── global-setup.js
│   └── global-teardown.js
├── performance/             # Performance tests
│   └── test_load_testing.py
└── security/                # Security tests
    └── test_security_scanning.py
```

## Running Tests

### Quick Commands

```bash
# All tests
npm test

# Specific test types
npm run test:unit
npm run test:integration
npm run test:e2e

# With coverage
npm run test:coverage

# Performance tests
npm run test:performance

# Security tests
npm run test:security
```

### Pytest Commands

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m security
pytest -m performance

# Run with coverage
pytest --cov --cov-report=html

# Run specific test file
pytest tests/unit/test_backlog_manager.py

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto

# Run only failed tests
pytest --lf
```

### Playwright E2E Tests

```bash
# Run E2E tests
npm run test:e2e

# Run with UI mode
npx playwright test --ui

# Run specific browser
npx playwright test --project=chromium

# Generate test report
npx playwright show-report
```

## Writing Tests

### Unit Tests

Unit tests should be fast, isolated, and test individual functions or classes.

```python
import pytest
from unittest.mock import Mock, patch
from myapp.services import UserService

class TestUserService:
    def test_create_user_success(self, mock_db):
        # Arrange
        user_data = {
            "email": "test@example.com",
            "username": "testuser"
        }
        service = UserService(mock_db)
        
        # Act
        result = service.create_user(user_data)
        
        # Assert
        assert result["status"] == "success"
        mock_db.save.assert_called_once()
    
    def test_create_user_validation_error(self):
        # Test validation errors
        service = UserService()
        
        with pytest.raises(ValueError, match="Invalid email"):
            service.create_user({"email": "invalid"})
```

### Integration Tests

Integration tests verify component interactions and API endpoints.

```python
import pytest
from fastapi.testclient import TestClient
from myapp.main import app

class TestUserAPI:
    def test_create_user_endpoint(self, api_client):
        # Arrange
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "securepass"
        }
        
        # Act
        response = api_client.post("/api/v1/users", json=user_data)
        
        # Assert
        assert response.status_code == 201
        assert response.json()["email"] == user_data["email"]
    
    def test_get_user_authenticated(self, api_client, authorized_headers):
        # Test authenticated endpoints
        response = api_client.get(
            "/api/v1/users/me", 
            headers=authorized_headers
        )
        
        assert response.status_code == 200
```

### End-to-End Tests

E2E tests use Playwright to simulate complete user workflows.

```javascript
const { test, expect } = require('@playwright/test');

test.describe('User Registration Flow', () => {
  test('should register new user successfully', async ({ page }) => {
    // Navigate to registration page
    await page.goto('/register');
    
    // Fill registration form
    await page.fill('[data-testid="email"]', 'test@example.com');
    await page.fill('[data-testid="username"]', 'testuser');
    await page.fill('[data-testid="password"]', 'securepass');
    
    // Submit form
    await page.click('[data-testid="submit"]');
    
    // Verify success
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page).toHaveURL('/dashboard');
  });
});
```

### Performance Tests

Performance tests use pytest-benchmark and Locust for load testing.

```python
import pytest
from locust import HttpUser, task, between

class TestAPIPerformance:
    def test_api_response_time(self, benchmark, api_client):
        # Benchmark API response time
        result = benchmark(api_client.get, "/api/v1/health")
        assert result.status_code == 200
    
    @pytest.mark.performance
    def test_concurrent_requests(self, api_client):
        # Test concurrent request handling
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(api_client.get, "/api/v1/health")
                for _ in range(100)
            ]
            
            results = [f.result() for f in futures]
            
        assert all(r.status_code == 200 for r in results)

class APILoadTest(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def test_health_endpoint(self):
        self.client.get("/api/v1/health")
    
    @task(1)
    def test_user_list(self):
        self.client.get("/api/v1/users")
```

### Security Tests

Security tests validate input sanitization and authentication.

```python
import pytest
from myapp.security import validate_input, sanitize_sql

class TestSecurityValidation:
    @pytest.mark.security
    def test_sql_injection_prevention(self, security_test_data):
        # Test SQL injection payloads
        for payload in security_test_data["sql_injection"]:
            with pytest.raises(ValueError):
                validate_input(payload)
    
    @pytest.mark.security
    def test_xss_prevention(self, security_test_data):
        # Test XSS payloads
        for payload in security_test_data["xss_payloads"]:
            sanitized = sanitize_sql(payload)
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized
    
    @pytest.mark.security
    def test_unauthorized_access(self, api_client):
        # Test unauthorized access to protected endpoints
        response = api_client.get("/api/v1/admin/users")
        assert response.status_code == 401
```

## Test Configuration

### Pytest Configuration

The `pytest.ini` file contains comprehensive test configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
addopts = 
    --verbose
    --cov=.
    --cov-report=html
    --cov-fail-under=80
    --junitxml=test-results/junit.xml

markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    security: Security tests
    performance: Performance tests
    slow: Slow tests
```

### Coverage Configuration

Coverage settings in `.coveragerc`:

```ini
[run]
source = .
omit = */tests/*, */venv/*, */node_modules/*
branch = true

[report]
fail_under = 80
show_missing = true
exclude_lines = pragma: no cover
```

## Fixtures and Utilities

### Common Fixtures

```python
@pytest.fixture
def api_client():
    """Test client for API requests."""
    from fastapi.testclient import TestClient
    from myapp.main import app
    
    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_database():
    """Mock database for testing."""
    with patch('myapp.database.get_db') as mock_db:
        yield mock_db

@pytest.fixture
def sample_user():
    """Sample user data."""
    return {
        "id": "test-user-123",
        "email": "test@example.com",
        "username": "testuser"
    }
```

### Test Data Factory

```python
class TestDataFactory:
    @staticmethod
    def create_user(**kwargs):
        """Create test user data."""
        default_data = {
            "id": "test-user-123",
            "email": "test@example.com",
            "username": "testuser",
            "is_active": True
        }
        default_data.update(kwargs)
        return default_data
```

## Testing AI Components

### Mock AI Services

```python
@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses."""
    with patch('openai.OpenAI') as mock_client:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test AI response"
        mock_client.return_value.chat.completions.create.return_value = mock_response
        yield mock_client

@pytest.fixture
def mock_crewai():
    """Mock CrewAI for testing."""
    with patch('crewai.Crew') as mock_crew:
        mock_crew.return_value.kickoff.return_value = "Test crew output"
        yield mock_crew
```

### Testing Agent Workflows

```python
class TestAgentWorkflow:
    def test_research_agent_task(self, mock_openai, mock_crewai):
        # Test agent task execution
        from myapp.agents import ResearchAgent
        
        agent = ResearchAgent()
        result = agent.execute_task("Research AI trends")
        
        assert "Test crew output" in result
        mock_crewai.return_value.kickoff.assert_called_once()
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Best Practices

### Test Organization

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **One assertion per test**: Focus on single behavior
3. **Descriptive test names**: Clearly describe what is being tested
4. **Use fixtures**: Share common setup between tests
5. **Mock external dependencies**: Keep tests isolated

### Performance Considerations

1. **Fast unit tests**: Keep under 100ms each
2. **Parallel execution**: Use pytest-xdist for speed
3. **Test data optimization**: Use minimal required data
4. **Database rollbacks**: Use transactions for database tests

### Maintenance

1. **Regular test reviews**: Remove obsolete tests
2. **Coverage monitoring**: Maintain 80%+ coverage
3. **Flaky test handling**: Identify and fix unstable tests
4. **Test documentation**: Document complex test scenarios

## Troubleshooting

### Common Issues

1. **Import errors**: Check PYTHONPATH and package structure
2. **Database connection**: Ensure test database is configured
3. **Async test failures**: Use proper async fixtures
4. **Mock not working**: Verify patch paths and import order

### Debug Tips

```bash
# Run with debug output
pytest -v -s

# Drop into debugger on failure
pytest --pdb

# Run single test with output
pytest -v -s tests/unit/test_specific.py::test_function
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Playwright Testing](https://playwright.dev/python/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)