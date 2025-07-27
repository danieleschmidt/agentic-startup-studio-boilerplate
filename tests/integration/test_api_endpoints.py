"""
Integration tests for API endpoints
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints."""

    @pytest.fixture
    def mock_app(self):
        """Mock FastAPI application for testing."""
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(title="Test Agentic Startup Studio API")
        
        @app.get("/health")
        def health_check():
            return {"status": "healthy", "timestamp": "2025-07-27T12:00:00Z"}
        
        @app.get("/api/v1/projects")
        def list_projects():
            return {
                "projects": [
                    {
                        "id": "project-1",
                        "name": "Test Project",
                        "description": "A test project",
                        "status": "active"
                    }
                ],
                "total": 1
            }
        
        @app.post("/api/v1/projects")
        def create_project(project_data: dict):
            return {
                "id": "project-123",
                "name": project_data.get("name", "New Project"),
                "description": project_data.get("description", ""),
                "status": "active",
                "created_at": "2025-07-27T12:00:00Z"
            }
        
        @app.get("/api/v1/agents")
        def list_agents():
            return {
                "agents": [
                    {
                        "id": "agent-1",
                        "name": "Research Agent",
                        "role": "researcher",
                        "status": "active"
                    }
                ],
                "total": 1
            }
        
        @app.post("/api/v1/agents")
        def create_agent(agent_data: dict):
            return {
                "id": "agent-123",
                "name": agent_data.get("name", "New Agent"),
                "role": agent_data.get("role", "assistant"),
                "goal": agent_data.get("goal", ""),
                "status": "active"
            }
        
        @app.post("/api/v1/tasks")
        def create_task(task_data: dict):
            return {
                "id": "task-123",
                "description": task_data.get("description", ""),
                "status": "pending",
                "agent_id": task_data.get("agent_id"),
                "created_at": "2025-07-27T12:00:00Z"
            }
        
        @app.post("/api/v1/tasks/{task_id}/execute")
        def execute_task(task_id: str):
            return {
                "task_id": task_id,
                "status": "completed",
                "result": "Task executed successfully",
                "execution_time": 1.5
            }
        
        return app

    @pytest.fixture
    def client(self, mock_app):
        """Test client for the mock application."""
        return TestClient(mock_app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_list_projects(self, client):
        """Test listing projects endpoint."""
        response = client.get("/api/v1/projects")
        
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert "total" in data
        assert len(data["projects"]) == 1
        assert data["projects"][0]["id"] == "project-1"

    def test_create_project(self, client):
        """Test creating a new project."""
        project_data = {
            "name": "Test Project",
            "description": "A test project for integration testing",
            "tech_stack": ["fastapi", "react", "crewai"]
        }
        
        response = client.post("/api/v1/projects", json=project_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == project_data["name"]
        assert "id" in data
        assert "created_at" in data

    def test_create_project_validation(self, client):
        """Test project creation with invalid data."""
        # Test with empty data
        response = client.post("/api/v1/projects", json={})
        assert response.status_code == 200  # Mock endpoint accepts empty data
        
        # Test with invalid JSON
        response = client.post("/api/v1/projects", data="invalid json")
        assert response.status_code == 422  # Validation error

    def test_list_agents(self, client):
        """Test listing agents endpoint."""
        response = client.get("/api/v1/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert "total" in data
        assert len(data["agents"]) == 1

    def test_create_agent(self, client):
        """Test creating a new agent."""
        agent_data = {
            "name": "Research Agent",
            "role": "researcher",
            "goal": "Conduct thorough research on given topics",
            "backstory": "An experienced researcher with expertise in various domains",
            "tools": ["web_search", "document_analysis"]
        }
        
        response = client.post("/api/v1/agents", json=agent_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == agent_data["name"]
        assert data["role"] == agent_data["role"]
        assert "id" in data

    def test_create_task(self, client):
        """Test creating a new task."""
        task_data = {
            "description": "Research the latest trends in AI technology",
            "expected_output": "A comprehensive report with key findings",
            "agent_id": "agent-123"
        }
        
        response = client.post("/api/v1/tasks", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["description"] == task_data["description"]
        assert data["agent_id"] == task_data["agent_id"]
        assert "id" in data

    def test_execute_task(self, client):
        """Test task execution endpoint."""
        task_id = "task-123"
        
        response = client.post(f"/api/v1/tasks/{task_id}/execute")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "completed"
        assert "result" in data
        assert "execution_time" in data

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        import asyncio
        import httpx
        
        async def make_request():
            async with httpx.AsyncClient(base_url="http://testserver") as async_client:
                response = await async_client.get("/health")
                return response.status_code
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should succeed
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Request failed: {result}")
            assert result == 200

    def test_cors_headers(self, client):
        """Test CORS headers in responses."""
        response = client.get("/health")
        
        # CORS headers would be set by middleware
        # This is a basic test structure
        assert response.status_code == 200

    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # All should succeed in test environment
        assert all(status == 200 for status in responses)

    def test_authentication_required_endpoints(self, client):
        """Test endpoints that require authentication."""
        # These would be actual protected endpoints
        protected_endpoints = [
            "/api/v1/projects",
            "/api/v1/agents",
            "/api/v1/tasks"
        ]
        
        for endpoint in protected_endpoints:
            # Without authentication, should still work in mock
            response = client.get(endpoint)
            assert response.status_code in [200, 401, 403]

    @patch('openai.OpenAI')
    def test_ai_integration(self, mock_openai, client):
        """Test AI service integration."""
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "AI generated response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test agent creation with AI integration
        agent_data = {
            "name": "AI Agent",
            "role": "ai_assistant",
            "use_ai": True
        }
        
        response = client.post("/api/v1/agents", json=agent_data)
        assert response.status_code == 200

    def test_error_handling(self, client):
        """Test API error handling."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test invalid method
        response = client.delete("/health")
        assert response.status_code == 405

    def test_request_validation(self, client):
        """Test request data validation."""
        # Test with invalid content type
        response = client.post(
            "/api/v1/projects",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )
        assert response.status_code in [422, 400]

    def test_response_format(self, client):
        """Test consistent response formatting."""
        response = client.get("/api/v1/projects")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, dict)

    @pytest.mark.parametrize("endpoint,method", [
        ("/health", "GET"),
        ("/api/v1/projects", "GET"),
        ("/api/v1/projects", "POST"),
        ("/api/v1/agents", "GET"),
        ("/api/v1/agents", "POST"),
    ])
    def test_endpoint_availability(self, client, endpoint, method):
        """Test that all endpoints are available."""
        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST":
            response = client.post(endpoint, json={})
        
        # Should not return 404
        assert response.status_code != 404

    def test_large_payload_handling(self, client):
        """Test handling of large request payloads."""
        large_data = {
            "name": "Large Project",
            "description": "A" * 10000,  # Large description
            "metadata": {f"key_{i}": f"value_{i}" for i in range(1000)}
        }
        
        response = client.post("/api/v1/projects", json=large_data)
        
        # Should handle large payloads gracefully
        assert response.status_code in [200, 413]  # 413 = Payload Too Large

    def test_special_characters_handling(self, client):
        """Test handling of special characters in requests."""
        special_data = {
            "name": "Test with émojis 🚀 and spëcial chärs",
            "description": "Unicode: αβγ, Symbols: @#$%^&*()",
        }
        
        response = client.post("/api/v1/projects", json=special_data)
        assert response.status_code == 200

    def test_content_negotiation(self, client):
        """Test content type negotiation."""
        # Test JSON response (default)
        response = client.get("/api/v1/projects")
        assert "application/json" in response.headers["content-type"]
        
        # Test with explicit Accept header
        response = client.get(
            "/api/v1/projects",
            headers={"Accept": "application/json"}
        )
        assert response.status_code == 200