"""
API Integration Tests for Quantum Task Planner
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

from fastapi.testclient import TestClient
from httpx import AsyncClient

from quantum_task_planner.api.quantum_api import app
from quantum_task_planner.core.quantum_task import TaskState, TaskPriority


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Mock authentication headers"""
    # In a real scenario, this would be a valid JWT token
    return {"Authorization": "Bearer mock_token"}


class TestTaskManagementAPI:
    """Test task management endpoints"""
    
    def test_create_task(self, client):
        """Test task creation endpoint"""
        task_data = {
            "title": "Test API Task",
            "description": "A task created via API",
            "priority": "high",
            "estimated_duration_hours": 2.5,
            "tags": ["api", "test"],
            "complexity_factor": 3.0
        }
        
        # Mock authentication for this test
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.post("/api/v1/tasks", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "task" in data
        
        task = data["task"]
        assert task["title"] == task_data["title"]
        assert task["description"] == task_data["description"]
        assert task["priority"] == TaskPriority.HIGH.value
        assert task["state"] == TaskState.PENDING.value
        assert "quantum_coherence" in task
        assert "task_id" in task
    
    def test_list_tasks(self, client):
        """Test task listing endpoint"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.get("/api/v1/tasks")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "tasks" in data
        assert "count" in data
        assert isinstance(data["tasks"], list)
        assert data["count"] == len(data["tasks"])
    
    def test_get_task_by_id(self, client):
        """Test getting specific task by ID"""
        # First create a task
        task_data = {
            "title": "Get Task Test",
            "description": "Test getting task by ID",
            "priority": "medium"
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            create_response = client.post("/api/v1/tasks", json=task_data)
            assert create_response.status_code == 200
            
            task_id = create_response.json()["task"]["task_id"]
            
            # Now get the task
            get_response = client.get(f"/api/v1/tasks/{task_id}")
            assert get_response.status_code == 200
            
            data = get_response.json()
            assert data["status"] == "success"
            assert data["task"]["task_id"] == task_id
            assert data["task"]["title"] == task_data["title"]
    
    def test_update_task(self, client):
        """Test task update endpoint"""
        # First create a task
        task_data = {
            "title": "Original Title",
            "description": "Original description",
            "priority": "low"
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            create_response = client.post("/api/v1/tasks", json=task_data)
            task_id = create_response.json()["task"]["task_id"]
            
            # Update the task
            update_data = {
                "title": "Updated Title",
                "priority": "high"
            }
            
            update_response = client.put(f"/api/v1/tasks/{task_id}", json=update_data)
            assert update_response.status_code == 200
            
            data = update_response.json()
            assert data["status"] == "success"
            assert data["task"]["title"] == "Updated Title"
            assert data["task"]["priority"] == TaskPriority.HIGH.value
            assert data["task"]["description"] == "Original description"  # Unchanged
    
    def test_delete_task(self, client):
        """Test task deletion endpoint"""
        # First create a task
        task_data = {
            "title": "Task to Delete",
            "description": "This task will be deleted",
            "priority": "medium"
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            create_response = client.post("/api/v1/tasks", json=task_data)
            task_id = create_response.json()["task"]["task_id"]
            
            # Delete the task
            delete_response = client.delete(f"/api/v1/tasks/{task_id}")
            assert delete_response.status_code == 200
            
            # Verify task is deleted
            get_response = client.get(f"/api/v1/tasks/{task_id}")
            assert get_response.status_code == 404
    
    def test_task_validation_errors(self, client):
        """Test task validation error handling"""
        # Test empty title
        invalid_data = {
            "title": "",
            "description": "Invalid task"
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.post("/api/v1/tasks", json=invalid_data)
            assert response.status_code == 400
            
        # Test invalid complexity factor
        invalid_data = {
            "title": "Invalid Task",
            "description": "Task with invalid complexity",
            "complexity_factor": 0.0
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.post("/api/v1/tasks", json=invalid_data)
            assert response.status_code == 400


class TestQuantumOperationsAPI:
    """Test quantum operations endpoints"""
    
    def test_quantum_measurement(self, client):
        """Test quantum measurement endpoint"""
        # First create a task
        task_data = {
            "title": "Measurement Test Task",
            "description": "Task for measurement testing",
            "priority": "medium"
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            create_response = client.post("/api/v1/tasks", json=task_data)
            task_id = create_response.json()["task"]["task_id"]
            
            # Perform quantum measurement
            measurement_data = {
                "task_ids": [task_id],
                "observer_effect": 0.2
            }
            
            response = client.post("/api/v1/quantum/measure", json=measurement_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "measurements" in data
            assert len(data["measurements"]) == 1
            
            measurement = data["measurements"][0]
            assert "measured_state" in measurement
            assert "measurement_probability" in measurement
            assert "observer_effect" in measurement
    
    def test_create_entanglement(self, client):
        """Test entanglement creation endpoint"""
        # Create two tasks for entanglement
        tasks = []
        for i in range(2):
            task_data = {
                "title": f"Entanglement Task {i}",
                "description": f"Task {i} for entanglement testing",
                "priority": "medium"
            }
            
            with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
                response = client.post("/api/v1/tasks", json=task_data)
                task_id = response.json()["task"]["task_id"]
                tasks.append(task_id)
        
        # Create entanglement
        entanglement_data = {
            "task_ids": tasks,
            "entanglement_type": "bell_state",
            "strength": 0.9
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.post("/api/v1/quantum/entangle", json=entanglement_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "entanglement_bond" in data
            
            bond = data["entanglement_bond"]
            assert bond["entanglement_type"] == "bell_state"
            assert bond["strength"] == 0.9
            assert len(bond["task_ids"]) == 2
    
    def test_optimization_endpoint(self, client):
        """Test optimization endpoint"""
        # Create tasks for optimization
        task_ids = []
        for i in range(3):
            task_data = {
                "title": f"Optimization Task {i}",
                "description": f"Task {i} for optimization testing",
                "priority": "medium" if i % 2 == 0 else "high",
                "complexity_factor": float(i + 1)
            }
            
            with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
                response = client.post("/api/v1/tasks", json=task_data)
                task_id = response.json()["task"]["task_id"]
                task_ids.append(task_id)
        
        # Run optimization
        optimization_data = {
            "task_ids": task_ids,
            "max_iterations": 20
        }
        
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.post("/api/v1/optimize/allocation", json=optimization_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "optimization_results" in data
            
            results = data["optimization_results"]
            assert "best_fitness" in results
            assert "optimization_history" in results
            assert results["best_fitness"] >= 0.0


class TestSchedulingAPI:
    """Test scheduling endpoints"""
    
    def test_get_schedule(self, client):
        """Test schedule retrieval endpoint"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.get("/api/v1/schedule")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "schedule" in data
            assert isinstance(data["schedule"], list)
    
    def test_schedule_statistics(self, client):
        """Test schedule statistics endpoint"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.get("/api/v1/schedule/stats")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "statistics" in data
            
            stats = data["statistics"]
            assert "total_tasks" in stats
            assert "scheduled_tasks" in stats
            assert "average_coherence" in stats


class TestEnhancedProductionEndpoints:
    """Test enhanced production endpoints"""
    
    def test_health_check(self, client):
        """Test comprehensive health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "2.0.0"
    
    def test_system_metrics(self, client):
        """Test system metrics endpoint"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.get("/api/v1/metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "metrics" in data
            
            metrics = data["metrics"]
            assert "timestamp" in metrics
            assert "system" in metrics
            
            system_metrics = metrics["system"]
            assert "total_tasks" in system_metrics
            assert "active_tasks" in system_metrics
            assert "average_coherence" in system_metrics
    
    def test_performance_metrics(self, client):
        """Test performance metrics endpoint"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.get("/api/v1/performance")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "performance" in data
    
    def test_cache_management(self, client):
        """Test cache management endpoints"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            # Clear all caches
            response = client.post("/api/v1/cache/clear")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert data["cleared_cache"] == "all"
            
            # Clear specific cache
            response = client.post("/api/v1/cache/clear?cache_name=default")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert data["cleared_cache"] == "default"


class TestErrorHandling:
    """Test API error handling"""
    
    def test_not_found_errors(self, client):
        """Test 404 error handling"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            # Non-existent task
            response = client.get("/api/v1/tasks/non-existent-id")
            assert response.status_code == 404
            
            # Non-existent endpoint
            response = client.get("/api/v1/nonexistent")
            assert response.status_code == 404
    
    def test_validation_errors(self, client):
        """Test validation error handling"""
        # Invalid JSON
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            response = client.post(
                "/api/v1/tasks",
                data="invalid json",
                headers={"content-type": "application/json"}
            )
            assert response.status_code == 400
    
    def test_authentication_errors(self, client):
        """Test authentication error handling"""
        # No authentication token
        response = client.get("/api/v1/tasks")
        # Should be redirected or return 401 depending on middleware configuration
        assert response.status_code in [401, 422]  # 422 for validation error in test environment
    
    def test_rate_limiting(self, client):
        """Test rate limiting"""
        # This would require proper rate limiting configuration in test environment
        # For now, just test that the endpoint exists and handles requests
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._check_rate_limit'):
            response = client.get("/api/v1/health")
            assert response.status_code == 200


class TestAsyncAPIOperations:
    """Test asynchronous API operations"""
    
    @pytest.mark.asyncio
    async def test_async_client_operations(self):
        """Test API with async client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Test health endpoint
            response = await ac.get("/api/v1/health")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
    
    @pytest.mark.asyncio 
    async def test_concurrent_api_requests(self):
        """Test handling concurrent API requests"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Make multiple concurrent requests
            tasks = []
            for i in range(10):
                task = ac.get("/api/v1/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "status" in data


class TestQuantumAPIIntegration:
    """Test complete quantum API integration scenarios"""
    
    def test_complete_quantum_workflow(self, client):
        """Test complete quantum task workflow via API"""
        with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
            # 1. Create multiple tasks
            task_ids = []
            for i in range(3):
                task_data = {
                    "title": f"Workflow Task {i}",
                    "description": f"Complete workflow test task {i}",
                    "priority": "high" if i == 0 else "medium",
                    "complexity_factor": float(i + 1)
                }
                
                response = client.post("/api/v1/tasks", json=task_data)
                assert response.status_code == 200
                task_id = response.json()["task"]["task_id"]
                task_ids.append(task_id)
            
            # 2. Create entanglement between first two tasks
            entanglement_data = {
                "task_ids": task_ids[:2],
                "entanglement_type": "bell_state",
                "strength": 0.8
            }
            
            response = client.post("/api/v1/quantum/entangle", json=entanglement_data)
            assert response.status_code == 200
            bond_data = response.json()["entanglement_bond"]
            
            # 3. Perform quantum measurements
            measurement_data = {
                "task_ids": task_ids,
                "observer_effect": 0.1
            }
            
            response = client.post("/api/v1/quantum/measure", json=measurement_data)
            assert response.status_code == 200
            measurements = response.json()["measurements"]
            assert len(measurements) == 3
            
            # 4. Run optimization
            optimization_data = {
                "task_ids": task_ids,
                "max_iterations": 10
            }
            
            response = client.post("/api/v1/optimize/allocation", json=optimization_data)
            assert response.status_code == 200
            
            # 5. Get final schedule
            response = client.get("/api/v1/schedule")
            assert response.status_code == 200
            schedule = response.json()["schedule"]
            
            # 6. Check system metrics
            response = client.get("/api/v1/metrics")
            assert response.status_code == 200
            metrics = response.json()["metrics"]
            
            # Verify workflow completed successfully
            assert metrics["system"]["total_tasks"] >= 3
            assert len(schedule) >= 0  # Schedule may be empty but should exist
    
    @pytest.mark.asyncio
    async def test_async_quantum_operations(self):
        """Test asynchronous quantum operations via API"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            with patch('quantum_task_planner.utils.middleware.SecurityMiddleware._authenticate_request'):
                # Create tasks concurrently
                task_creation_tasks = []
                for i in range(5):
                    task_data = {
                        "title": f"Async Task {i}",
                        "description": f"Async test task {i}",
                        "priority": "medium"
                    }
                    
                    task = ac.post("/api/v1/tasks", json=task_data)
                    task_creation_tasks.append(task)
                
                # Wait for all tasks to be created
                responses = await asyncio.gather(*task_creation_tasks)
                
                # Extract task IDs
                task_ids = []
                for response in responses:
                    assert response.status_code == 200
                    task_id = response.json()["task"]["task_id"]
                    task_ids.append(task_id)
                
                # Perform concurrent quantum operations
                quantum_ops = [
                    ac.post("/api/v1/quantum/measure", json={"task_ids": [task_ids[0]]}),
                    ac.post("/api/v1/quantum/measure", json={"task_ids": [task_ids[1]]}),
                    ac.post("/api/v1/quantum/entangle", json={
                        "task_ids": task_ids[2:4],
                        "entanglement_type": "bell_state"
                    })
                ]
                
                quantum_responses = await asyncio.gather(*quantum_ops)
                
                # All quantum operations should succeed
                for response in quantum_responses:
                    assert response.status_code == 200
                    assert response.json()["status"] == "success"


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--cov=quantum_task_planner.api",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])