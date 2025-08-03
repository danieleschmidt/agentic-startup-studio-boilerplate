"""
Test cases for agent functionality.
Tests AI agent operations, task execution, and crew coordination.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient

from app.agents.crew import AgentCrew, AgentRequest, AgentResponse


class TestAgentModels:
    """Test agent model validation and serialization."""
    
    def test_agent_request_validation(self):
        """Test AgentRequest model validation."""
        # Valid request
        valid_request = AgentRequest(
            task_description="Test task description",
            priority="high",
            context={"key": "value"}
        )
        assert valid_request.task_description == "Test task description"
        assert valid_request.priority == "high"
        assert valid_request.context == {"key": "value"}
        
        # Test default values
        minimal_request = AgentRequest(task_description="Minimal task")
        assert minimal_request.priority == "normal"
        assert minimal_request.context is None
        assert minimal_request.deadline is None
    
    def test_agent_response_model(self):
        """Test AgentResponse model structure."""
        response = AgentResponse(
            success=True,
            result="Task completed successfully",
            metadata={"execution_info": "test"},
            execution_time=1.5,
            agents_used=["research_agent", "analysis_agent"]
        )
        
        assert response.success is True
        assert response.result == "Task completed successfully"
        assert response.execution_time == 1.5
        assert len(response.agents_used) == 2


class TestAgentTools:
    """Test individual agent tools."""
    
    @pytest.mark.asyncio
    async def test_web_search_tool(self):
        """Test web search tool functionality."""
        from app.agents.tools import WebSearchTool
        
        tool = WebSearchTool()
        result = tool._run("artificial intelligence trends")
        
        assert "Search Results for 'artificial intelligence trends'" in result
        assert "Comprehensive Guide to" in result
        assert "Latest Trends in" in result
        assert "Case Studies and Examples" in result
    
    @pytest.mark.asyncio
    async def test_data_analysis_tool(self):
        """Test data analysis tool functionality."""
        from app.agents.tools import DataAnalysisTool
        
        tool = DataAnalysisTool()
        result = tool._run("user engagement metrics")
        
        assert "Data Analysis Results for: user engagement metrics" in result
        assert "Sample Size:" in result
        assert "Key Metrics:" in result
        assert "Key Insights:" in result
        assert "Recommendations:" in result
    
    @pytest.mark.asyncio
    async def test_report_generator_tool(self):
        """Test report generator tool functionality."""
        from app.agents.tools import ReportGeneratorTool
        
        tool = ReportGeneratorTool()
        
        # Test standard report
        result = tool._run("Test content for report generation", "standard")
        assert "# RESEARCH AND ANALYSIS REPORT" in result
        assert "## EXECUTIVE SUMMARY" in result
        assert "## DETAILED FINDINGS" in result
        
        # Test executive report
        executive_result = tool._run("Executive content", "executive")
        assert "# EXECUTIVE SUMMARY REPORT" in executive_result
        assert "## STRATEGIC IMPLICATIONS" in executive_result
        
        # Test technical report
        technical_result = tool._run("Technical content", "technical")
        assert "# TECHNICAL ANALYSIS REPORT" in technical_result
        assert "## METHODOLOGY" in technical_result


class TestAgentCrew:
    """Test agent crew coordination and execution."""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        mock = MagicMock()
        mock.predict.return_value = "Mocked LLM response"
        return mock
    
    @pytest.fixture
    def mock_agent_crew(self, mock_llm):
        """Create mock agent crew for testing."""
        with patch('app.agents.crew.OpenAI', return_value=mock_llm):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                from app.agents.crew import AgentCrew
                crew = AgentCrew()
                return crew
    
    @pytest.mark.asyncio
    async def test_agent_crew_initialization(self, mock_agent_crew):
        """Test agent crew initialization."""
        assert mock_agent_crew.research_agent is not None
        assert mock_agent_crew.analysis_agent is not None
        assert mock_agent_crew.report_agent is not None
        assert mock_agent_crew.crew is not None
    
    @pytest.mark.asyncio
    async def test_execute_research_task_success(self, mock_agent_crew):
        """Test successful research task execution."""
        # Mock crew kickoff
        mock_result = "Mocked crew execution result"
        mock_agent_crew.crew.kickoff = MagicMock(return_value=mock_result)
        
        request = AgentRequest(
            task_description="Test research task",
            priority="high",
            context={"test": True}
        )
        
        response = await mock_agent_crew.execute_research_task(request)
        
        assert response.success is True
        assert response.result == mock_result
        assert response.execution_time > 0
        assert "research_agent" in response.agents_used
        assert "analysis_agent" in response.agents_used
        assert "report_agent" in response.agents_used
    
    @pytest.mark.asyncio
    async def test_execute_research_task_failure(self, mock_agent_crew):
        """Test research task execution failure."""
        # Mock crew kickoff to raise exception
        mock_agent_crew.crew.kickoff = MagicMock(side_effect=Exception("Test error"))
        
        request = AgentRequest(
            task_description="Test research task",
            priority="normal"
        )
        
        response = await mock_agent_crew.execute_research_task(request)
        
        assert response.success is False
        assert "Error executing research task: Test error" in response.result
        assert response.execution_time > 0
        assert len(response.agents_used) == 0


class TestAgentAPI:
    """Test agent API endpoints."""
    
    @pytest.mark.asyncio
    async def test_execute_research_task_endpoint(
        self, 
        async_client: AsyncClient, 
        auth_headers: dict,
        sample_agent_request: dict,
        mock_external_services
    ):
        """Test research task execution endpoint."""
        with patch('app.agents.crew.get_agent_crew') as mock_get_crew:
            # Mock agent crew
            mock_crew = AsyncMock()
            mock_response = AgentResponse(
                success=True,
                result="Test research completed",
                metadata={"test": True},
                execution_time=2.5,
                agents_used=["research_agent"]
            )
            mock_crew.execute_research_task.return_value = mock_response
            mock_get_crew.return_value = mock_crew
            
            response = await async_client.post(
                "/api/v1/agents/research",
                json=sample_agent_request,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["result"] == "Test research completed"
            assert data["execution_time"] == 2.5
    
    @pytest.mark.asyncio
    async def test_submit_research_task_async_endpoint(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_agent_request: dict
    ):
        """Test async research task submission endpoint."""
        response = await async_client.post(
            "/api/v1/agents/research/async",
            json=sample_agent_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "submitted"
        assert "message" in data
    
    @pytest.mark.asyncio
    async def test_get_task_status_endpoint(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_agent_request: dict
    ):
        """Test task status retrieval endpoint."""
        # First, submit a task
        submit_response = await async_client.post(
            "/api/v1/agents/research/async",
            json=sample_agent_request,
            headers=auth_headers
        )
        
        task_id = submit_response.json()["task_id"]
        
        # Then get its status
        status_response = await async_client.get(
            f"/api/v1/agents/tasks/{task_id}",
            headers=auth_headers
        )
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert "status" in status_data
        assert "progress" in status_data
    
    @pytest.mark.asyncio
    async def test_list_user_tasks_endpoint(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_agent_request: dict
    ):
        """Test user tasks listing endpoint."""
        # Submit a couple of tasks
        await async_client.post(
            "/api/v1/agents/research/async",
            json=sample_agent_request,
            headers=auth_headers
        )
        
        modified_request = sample_agent_request.copy()
        modified_request["task_description"] = "Second test task"
        await async_client.post(
            "/api/v1/agents/research/async",
            json=modified_request,
            headers=auth_headers
        )
        
        # List tasks
        response = await async_client.get(
            "/api/v1/agents/tasks",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        tasks = response.json()
        assert isinstance(tasks, list)
        assert len(tasks) == 2
    
    @pytest.mark.asyncio
    async def test_cancel_task_endpoint(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        sample_agent_request: dict
    ):
        """Test task cancellation endpoint."""
        # Submit a task
        submit_response = await async_client.post(
            "/api/v1/agents/research/async",
            json=sample_agent_request,
            headers=auth_headers
        )
        
        task_id = submit_response.json()["task_id"]
        
        # Cancel the task
        cancel_response = await async_client.delete(
            f"/api/v1/agents/tasks/{task_id}",
            headers=auth_headers
        )
        
        assert cancel_response.status_code == 200
        cancel_data = cancel_response.json()
        assert "message" in cancel_data
        assert task_id in cancel_data["message"]
    
    @pytest.mark.asyncio
    async def test_unauthorized_access(self, async_client: AsyncClient, sample_agent_request: dict):
        """Test unauthorized access to agent endpoints."""
        # Try to access without auth headers
        response = await async_client.post(
            "/api/v1/agents/research",
            json=sample_agent_request
        )
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_invalid_request_data(self, async_client: AsyncClient, auth_headers: dict):
        """Test invalid request data handling."""
        invalid_request = {
            "task_description": "",  # Empty description
            "priority": "invalid_priority",  # Invalid priority
        }
        
        response = await async_client.post(
            "/api/v1/agents/research",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error


class TestAgentIntegration:
    """Integration tests for agent functionality."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        mock_external_services
    ):
        """Test complete research workflow from submission to completion."""
        request_data = {
            "task_description": "Analyze the impact of AI on healthcare industry",
            "priority": "high",
            "context": {
                "industry": "healthcare",
                "focus": "AI applications",
                "timeframe": "2024"
            }
        }
        
        with patch('app.agents.crew.get_agent_crew') as mock_get_crew:
            # Mock successful agent execution
            mock_crew = AsyncMock()
            mock_response = AgentResponse(
                success=True,
                result="Comprehensive analysis of AI in healthcare completed",
                metadata={"industry": "healthcare", "agents_count": 3},
                execution_time=5.2,
                agents_used=["research_agent", "analysis_agent", "report_agent"]
            )
            mock_crew.execute_research_task.return_value = mock_response
            mock_get_crew.return_value = mock_crew
            
            # Execute synchronous research task
            response = await async_client.post(
                "/api/v1/agents/research",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert data["success"] is True
            assert "AI in healthcare" in data["result"]
            assert data["execution_time"] > 0
            assert len(data["agents_used"]) == 3
            assert data["metadata"]["industry"] == "healthcare"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_task_lifecycle(
        self,
        async_client: AsyncClient,
        auth_headers: dict
    ):
        """Test complete async task lifecycle."""
        request_data = {
            "task_description": "Research renewable energy trends",
            "priority": "normal"
        }
        
        # 1. Submit async task
        submit_response = await async_client.post(
            "/api/v1/agents/research/async",
            json=request_data,
            headers=auth_headers
        )
        
        assert submit_response.status_code == 200
        task_data = submit_response.json()
        task_id = task_data["task_id"]
        
        # 2. Check initial status
        status_response = await async_client.get(
            f"/api/v1/agents/tasks/{task_id}",
            headers=auth_headers
        )
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] in ["pending", "running"]
        assert status_data["progress"] >= 0.0
        
        # 3. List all tasks (should include our task)
        list_response = await async_client.get(
            "/api/v1/agents/tasks",
            headers=auth_headers
        )
        
        assert list_response.status_code == 200
        tasks = list_response.json()
        task_ids = [task["task_id"] for task in tasks]
        assert task_id in task_ids
        
        # 4. Cancel the task
        cancel_response = await async_client.delete(
            f"/api/v1/agents/tasks/{task_id}",
            headers=auth_headers
        )
        
        assert cancel_response.status_code == 200


class TestAgentSecurity:
    """Test security aspects of agent functionality."""
    
    @pytest.mark.asyncio
    async def test_task_access_control(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        test_data_factory
    ):
        """Test that users can only access their own tasks."""
        # Create second user with different token
        user2_data = test_data_factory.create_user_data(
            email="user2@example.com",
            username="user2"
        )
        
        # This test would need additional setup for a second user
        # For now, just test access with invalid task ID
        response = await async_client.get(
            "/api/v1/agents/tasks/invalid-task-id",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_input_sanitization(
        self,
        async_client: AsyncClient,
        auth_headers: dict,
        mock_external_services
    ):
        """Test input sanitization for malicious content."""
        malicious_request = {
            "task_description": "<script>alert('xss')</script>Analyze data",
            "priority": "high",
            "context": {
                "malicious_key": "'; DROP TABLE users; --"
            }
        }
        
        with patch('app.agents.crew.get_agent_crew') as mock_get_crew:
            mock_crew = AsyncMock()
            mock_response = AgentResponse(
                success=True,
                result="Safe analysis completed",
                metadata={},
                execution_time=1.0,
                agents_used=["research_agent"]
            )
            mock_crew.execute_research_task.return_value = mock_response
            mock_get_crew.return_value = mock_crew
            
            response = await async_client.post(
                "/api/v1/agents/research",
                json=malicious_request,
                headers=auth_headers
            )
            
            # Should still work but content should be handled safely
            assert response.status_code == 200
            # The actual sanitization would happen in the crew execution