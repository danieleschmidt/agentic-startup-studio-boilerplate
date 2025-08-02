# Testing Guide

This document provides comprehensive guidance on testing practices for the Agentic Startup Studio Boilerplate.

## Testing Philosophy

Our testing strategy follows the **testing pyramid** approach:
- **Unit Tests** (70%): Fast, isolated tests for individual components
- **Integration Tests** (20%): Test component interactions and APIs
- **End-to-End Tests** (10%): Full system tests simulating user workflows

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── unit/                # Unit tests (fast, isolated)
├── integration/         # Integration tests (API endpoints, database)
├── e2e/                 # End-to-end tests (full user workflows)
├── performance/         # Load and performance tests
└── security/            # Security and vulnerability tests
```

## Running Tests

### Quick Start
```bash
# Run all tests
npm test

# Run specific test types
npm run test:api          # Backend tests only
npm run test:frontend     # Frontend tests only
npm run test:e2e          # End-to-end tests

# Run with coverage
npm run test:coverage
```

### Test Commands
```bash
# Backend testing (Python)
cd backend
pytest                    # Run all Python tests
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Exclude slow tests
pytest --cov             # With coverage report

# Frontend testing (JavaScript/TypeScript)
cd frontend
npm test                  # Run all frontend tests
npm run test:watch       # Watch mode for development
npm run test:coverage    # Coverage report

# End-to-end testing (Playwright)
npx playwright test       # Run all E2E tests
npx playwright test --ui  # Interactive UI mode
npx playwright show-report  # View test results
```

## Test Categories

### Unit Tests
- **Purpose**: Test individual functions, classes, and components in isolation
- **Speed**: Fast (< 1 second each)
- **Scope**: Single unit of code
- **Mocking**: Heavily mock external dependencies

```python
# Example unit test
@pytest.mark.unit
def test_calculate_wsjf_score():
    """Test WSJF score calculation."""
    item = {
        "value": 8,
        "time_criticality": 5,
        "risk_reduction": 3,
        "effort": 2
    }
    expected_score = (8 + 5 + 3) / 2  # 8.0
    assert calculate_wsjf_score(item) == expected_score
```

### Integration Tests
- **Purpose**: Test component interactions, APIs, and data flow
- **Speed**: Medium (1-10 seconds each)
- **Scope**: Multiple components working together
- **Mocking**: Limited mocking, real database connections

```python
# Example integration test
@pytest.mark.integration
async def test_create_project_endpoint(api_client, test_db):
    """Test project creation through API."""
    project_data = {
        "name": "Test Project",
        "description": "Integration test project"
    }
    response = await api_client.post("/api/v1/projects", json=project_data)
    assert response.status_code == 201
    assert response.json()["name"] == project_data["name"]
```

### End-to-End Tests
- **Purpose**: Test complete user workflows and system behavior
- **Speed**: Slow (10+ seconds each)
- **Scope**: Full application stack
- **Mocking**: Minimal mocking, real external services in test mode

```javascript
// Example E2E test
test('user can create and manage a project', async ({ page }) => {
  await page.goto('/');
  await page.click('[data-testid="new-project-button"]');
  await page.fill('[data-testid="project-name"]', 'My Test Project');
  await page.click('[data-testid="create-project"]');
  await expect(page.locator('[data-testid="project-title"]')).toContainText('My Test Project');
});
```

### Performance Tests
- **Purpose**: Verify system performance under load
- **Tools**: pytest-benchmark, k6, artillery
- **Metrics**: Response time, throughput, resource usage

```python
# Example performance test
@pytest.mark.performance
def test_api_response_time(benchmark, api_client):
    """Test API response time under load."""
    def api_call():
        return api_client.get("/api/v1/health")
    
    result = benchmark(api_call)
    assert result.status_code == 200
```

### Security Tests
- **Purpose**: Identify security vulnerabilities
- **Tools**: bandit, safety, OWASP ZAP
- **Coverage**: Input validation, authentication, authorization

```python
# Example security test
@pytest.mark.security
def test_sql_injection_protection(api_client, security_test_data):
    """Test protection against SQL injection."""
    for payload in security_test_data["sql_injection"]:
        response = api_client.get(f"/api/v1/search?q={payload}")
        assert response.status_code != 500  # Should not crash
        assert "error" not in response.json().get("message", "").lower()
```

## Test Configuration

### Pytest Configuration
The main pytest configuration is in `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests
    e2e: End-to-end tests
    security: Security tests
    performance: Performance tests
addopts = --verbose --cov --cov-report=html
```

### Coverage Configuration
Coverage settings are in `.coveragerc`:

```ini
[run]
source = .
omit = */tests/*, */venv/*, */node_modules/*
branch = true

[report]
fail_under = 80
show_missing = true
```

### Playwright Configuration
E2E test configuration in `playwright.config.js`:

```javascript
module.exports = defineConfig({
  testDir: './tests/e2e',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
});
```

## Test Data Management

### Fixtures
Use pytest fixtures for reusable test data:

```python
@pytest.fixture
def sample_project():
    return {
        "name": "Test Project",
        "description": "A test project",
        "tech_stack": ["fastapi", "react"]
    }
```

### Factories
Use test data factories for complex objects:

```python
class ProjectFactory:
    @staticmethod
    def create(**kwargs):
        defaults = {
            "name": "Default Project",
            "status": "active"
        }
        defaults.update(kwargs)
        return defaults
```

### Database Tests
Use transactions that rollback for database tests:

```python
@pytest.fixture
def db_transaction():
    with database.transaction():
        yield
        # Automatic rollback
```

## Mocking Guidelines

### When to Mock
- External API calls
- File system operations
- Network requests
- Expensive computations
- Third-party services

### Mock Examples
```python
# Mock external API
@patch('httpx.AsyncClient.get')
def test_external_api_call(mock_get):
    mock_get.return_value.json.return_value = {"status": "success"}
    result = call_external_api()
    assert result["status"] == "success"

# Mock database
@patch('database.connection')
def test_database_operation(mock_db):
    mock_db.execute.return_value = [{"id": 1, "name": "test"}]
    result = get_users()
    assert len(result) == 1
```

## CI/CD Integration

### GitHub Actions
Tests run automatically on:
- Pull requests
- Pushes to main branch
- Scheduled daily runs

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    npm run test
    npm run test:e2e
    npm run security
```

### Test Reports
- Coverage reports uploaded to Codecov
- Test results published as GitHub check
- Performance regression detection

## Best Practices

### Writing Good Tests
1. **AAA Pattern**: Arrange, Act, Assert
2. **Single Responsibility**: One test per behavior
3. **Descriptive Names**: Test names should explain what they test
4. **Independent**: Tests should not depend on each other
5. **Fast**: Keep unit tests under 1 second

### Test Naming
```python
# Good: Describes what is being tested
def test_user_creation_with_valid_email_succeeds():
    pass

# Bad: Vague and unclear
def test_user():
    pass
```

### Assertions
```python
# Good: Specific and descriptive
assert response.status_code == 201
assert "error" not in response.json()
assert user.email == "test@example.com"

# Bad: Vague and unclear
assert response
assert user
```

## Debugging Tests

### Running Specific Tests
```bash
# Run single test file
pytest tests/unit/test_backlog_manager.py

# Run single test function
pytest tests/unit/test_backlog_manager.py::test_calculate_wsjf_score

# Run tests matching pattern
pytest -k "test_user"
```

### Debug Mode
```bash
# Run with debugger
pytest --pdb

# Run with verbose output
pytest -v -s

# Run with coverage and no capture
pytest --cov --no-cov-on-fail -s
```

### VS Code Integration
Configure VS Code for test debugging:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.testing.autoTestDiscoverOnSaveEnabled": true
}
```

## Performance Testing

### Load Testing
Use k6 for load testing:

```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '5m', target: 100 },
    { duration: '10m', target: 100 },
    { duration: '5m', target: 0 },
  ],
};

export default function() {
  let response = http.get('http://localhost:8000/api/v1/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
}
```

### Benchmarking
Use pytest-benchmark for Python performance tests:

```python
def test_algorithm_performance(benchmark):
    result = benchmark(expensive_algorithm, large_dataset)
    assert result is not None
```

## Security Testing

### Automated Security Scans
- **bandit**: Python security scanner
- **safety**: Python dependency scanner
- **npm audit**: Node.js dependency scanner
- **Trivy**: Container security scanner

### Manual Security Testing
- Input validation testing
- Authentication bypass attempts
- Authorization testing
- SQL injection testing
- XSS testing

## Troubleshooting

### Common Issues
1. **Flaky Tests**: Use proper waits and retries
2. **Slow Tests**: Optimize database operations and mocking
3. **Environment Issues**: Use containers for consistency
4. **Mock Problems**: Ensure mocks match real interfaces

### Getting Help
- Check test logs and error messages
- Run tests in isolation to identify issues
- Use debugger to step through failing tests
- Consult team documentation and knowledge base

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://testing.googleblog.com/)
- [Python Testing 101](https://realpython.com/pytest-python-testing/)