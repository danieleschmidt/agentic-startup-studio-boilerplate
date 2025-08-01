"""
Test data fixtures and factories for Agentic Startup Studio Boilerplate
"""

import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4

import pytest


class TestDataFactory:
    """Factory class for generating test data."""
    
    @staticmethod
    def generate_id(prefix: str = "test") -> str:
        """Generate a unique test ID."""
        return f"{prefix}-{uuid4().hex[:8]}"
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_email(domain: str = "test.com") -> str:
        """Generate a random email address."""
        username = TestDataFactory.random_string(8).lower()
        return f"{username}@{domain}"


class UserDataFactory(TestDataFactory):
    """Factory for user-related test data."""
    
    @classmethod
    def create_user(cls, **kwargs) -> Dict[str, Any]:
        """Create a test user."""
        defaults = {
            "id": cls.generate_id("user"),
            "email": cls.random_email(),
            "username": cls.random_string(8).lower(),
            "first_name": cls.random_string(6).title(),
            "last_name": cls.random_string(8).title(),
            "is_active": True,
            "is_verified": True,
            "is_superuser": False,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "preferences": {
                "theme": "light",
                "notifications": True,
                "language": "en"
            }
        }
        defaults.update(kwargs)
        return defaults
    
    @classmethod
    def create_admin_user(cls, **kwargs) -> Dict[str, Any]:
        """Create a test admin user."""
        defaults = {
            "is_superuser": True,
            "email": "admin@test.com",
            "username": "admin"
        }
        defaults.update(kwargs)
        return cls.create_user(**defaults)
    
    @classmethod
    def create_inactive_user(cls, **kwargs) -> Dict[str, Any]:
        """Create an inactive test user."""
        defaults = {
            "is_active": False,
            "is_verified": False
        }
        defaults.update(kwargs)
        return cls.create_user(**defaults)


class ProjectDataFactory(TestDataFactory):
    """Factory for project-related test data."""
    
    TECH_STACKS = [
        ["fastapi", "react", "crewai", "postgresql"],
        ["django", "vue", "langchain", "mysql"],
        ["flask", "angular", "autogen", "sqlite"],
        ["nextjs", "typescript", "openai", "mongodb"]
    ]
    
    PROJECT_TYPES = ["web_app", "api", "mobile_app", "desktop_app", "cli_tool"]
    STATUSES = ["active", "inactive", "archived", "draft"]
    
    @classmethod
    def create_project(cls, **kwargs) -> Dict[str, Any]:
        """Create a test project."""
        defaults = {
            "id": cls.generate_id("project"),
            "name": f"Test Project {cls.random_string(4)}",
            "description": f"A test project for {cls.random_string(6)} functionality",
            "tech_stack": random.choice(cls.TECH_STACKS),
            "project_type": random.choice(cls.PROJECT_TYPES),
            "status": random.choice(cls.STATUSES),
            "owner_id": cls.generate_id("user"),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "repository_url": f"https://github.com/test/{cls.random_string(8).lower()}",
            "deployment_url": None,
            "settings": {
                "auto_deploy": True,
                "notifications": True,
                "public": False
            },
            "tags": ["test", "automation", cls.random_string(5).lower()]
        }
        defaults.update(kwargs)
        return defaults
    
    @classmethod
    def create_active_project(cls, **kwargs) -> Dict[str, Any]:
        """Create an active test project."""
        defaults = {"status": "active"}
        defaults.update(kwargs)
        return cls.create_project(**defaults)


class AgentDataFactory(TestDataFactory):
    """Factory for agent-related test data."""
    
    ROLES = [
        "researcher", "writer", "analyst", "developer", 
        "designer", "tester", "reviewer", "coordinator"
    ]
    
    TOOLS = [
        ["web_search", "file_analysis", "data_extraction"],
        ["content_generation", "grammar_check", "formatting"],
        ["code_generation", "code_review", "testing"],
        ["image_generation", "image_analysis", "ui_design"],
        ["email_sender", "notification", "webhook"]
    ]
    
    @classmethod
    def create_agent(cls, **kwargs) -> Dict[str, Any]:
        """Create a test agent."""
        role = kwargs.get("role", random.choice(cls.ROLES))
        defaults = {
            "id": cls.generate_id("agent"),
            "name": f"Test {role.title()} Agent",
            "role": role,
            "goal": f"Perform {role} tasks efficiently and accurately",
            "backstory": f"An AI agent specialized in {role} activities with extensive experience",
            "tools": random.choice(cls.TOOLS),
            "llm_model": "gpt-4",
            "max_tokens": 2000,
            "temperature": 0.7,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "owner_id": cls.generate_id("user"),
            "project_id": cls.generate_id("project"),
            "configuration": {
                "memory_enabled": True,
                "verbose": False,
                "allow_delegation": True
            },
            "performance_metrics": {
                "tasks_completed": random.randint(0, 100),
                "success_rate": round(random.uniform(0.8, 1.0), 2),
                "avg_response_time": round(random.uniform(1.0, 5.0), 2)
            }
        }
        defaults.update(kwargs)
        return defaults
    
    @classmethod
    def create_researcher_agent(cls, **kwargs) -> Dict[str, Any]:
        """Create a researcher agent."""
        defaults = {
            "role": "researcher",
            "tools": ["web_search", "file_analysis", "data_extraction", "summarization"]
        }
        defaults.update(kwargs)
        return cls.create_agent(**defaults)


class TaskDataFactory(TestDataFactory):
    """Factory for task-related test data."""
    
    STATUSES = ["pending", "in_progress", "completed", "failed", "cancelled"]
    PRIORITIES = ["low", "medium", "high", "urgent"]
    
    @classmethod
    def create_task(cls, **kwargs) -> Dict[str, Any]:
        """Create a test task."""
        defaults = {
            "id": cls.generate_id("task"),
            "description": f"Research and analyze {cls.random_string(8)} trends in the market",
            "expected_output": "A comprehensive report with findings and recommendations",
            "agent_id": cls.generate_id("agent"),
            "project_id": cls.generate_id("project"),
            "status": random.choice(cls.STATUSES),
            "priority": random.choice(cls.PRIORITIES),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "estimated_duration": random.randint(10, 120),  # minutes
            "actual_duration": None,
            "context": {
                "source": "user_request",
                "parent_task_id": None,
                "dependencies": []
            },
            "output": None,
            "error_message": None,
            "retry_count": 0,
            "max_retries": 3
        }
        defaults.update(kwargs)
        
        # Set timestamps based on status
        if defaults["status"] in ["in_progress", "completed", "failed"]:
            defaults["started_at"] = (datetime.utcnow() - timedelta(minutes=random.randint(1, 60))).isoformat()
        
        if defaults["status"] in ["completed", "failed"]:
            start_time = datetime.fromisoformat(defaults["started_at"])
            duration = random.randint(5, defaults["estimated_duration"])
            defaults["completed_at"] = (start_time + timedelta(minutes=duration)).isoformat()
            defaults["actual_duration"] = duration
        
        return defaults
    
    @classmethod
    def create_completed_task(cls, **kwargs) -> Dict[str, Any]:
        """Create a completed test task."""
        defaults = {
            "status": "completed",
            "output": "Task completed successfully with comprehensive results."
        }
        defaults.update(kwargs)
        return cls.create_task(**defaults)


class CrewDataFactory(TestDataFactory):
    """Factory for crew/team-related test data."""
    
    @classmethod
    def create_crew(cls, **kwargs) -> Dict[str, Any]:
        """Create a test crew."""
        defaults = {
            "id": cls.generate_id("crew"),
            "name": f"Test Crew {cls.random_string(4)}",
            "description": f"A test crew for {cls.random_string(6)} operations",
            "agents": [
                AgentDataFactory.create_agent()["id"] for _ in range(random.randint(2, 5))
            ],
            "project_id": cls.generate_id("project"),
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "configuration": {
                "process": "sequential",  # or "hierarchical"
                "verbose": False,
                "memory": True
            },
            "performance_metrics": {
                "tasks_completed": random.randint(0, 50),
                "success_rate": round(random.uniform(0.7, 1.0), 2),
                "avg_completion_time": round(random.uniform(10.0, 60.0), 2)
            }
        }
        defaults.update(kwargs)
        return defaults


# Pytest fixtures
@pytest.fixture
def user_factory():
    """User data factory fixture."""
    return UserDataFactory


@pytest.fixture
def project_factory():
    """Project data factory fixture."""
    return ProjectDataFactory


@pytest.fixture
def agent_factory():
    """Agent data factory fixture."""
    return AgentDataFactory


@pytest.fixture
def task_factory():
    """Task data factory fixture."""
    return TaskDataFactory


@pytest.fixture
def crew_factory():
    """Crew data factory fixture."""
    return CrewDataFactory


@pytest.fixture
def sample_user(user_factory):
    """Sample user data."""
    return user_factory.create_user()


@pytest.fixture
def sample_admin_user(user_factory):
    """Sample admin user data."""
    return user_factory.create_admin_user()


@pytest.fixture
def sample_project(project_factory):
    """Sample project data."""
    return project_factory.create_project()


@pytest.fixture
def sample_agent(agent_factory):
    """Sample agent data."""
    return agent_factory.create_agent()


@pytest.fixture
def sample_task(task_factory):
    """Sample task data."""
    return task_factory.create_task()


@pytest.fixture
def sample_crew(crew_factory):
    """Sample crew data."""
    return crew_factory.create_crew()


# Bulk data fixtures
@pytest.fixture
def bulk_users(user_factory):
    """Create multiple test users."""
    return [user_factory.create_user() for _ in range(10)]


@pytest.fixture
def bulk_projects(project_factory):
    """Create multiple test projects."""
    return [project_factory.create_project() for _ in range(5)]


@pytest.fixture
def bulk_agents(agent_factory):
    """Create multiple test agents."""
    return [agent_factory.create_agent() for _ in range(8)]


@pytest.fixture
def bulk_tasks(task_factory):
    """Create multiple test tasks."""
    return [task_factory.create_task() for _ in range(15)]


# File-based fixtures
@pytest.fixture
def load_test_data():
    """Load test data from JSON files."""
    def _load_data(filename: str) -> Dict[str, Any]:
        test_data_dir = Path(__file__).parent / "data"
        file_path = test_data_dir / f"{filename}.json"
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    
    return _load_data


@pytest.fixture
def performance_test_data():
    """Performance test scenario data."""
    return {
        "light_load": {
            "concurrent_users": 10,
            "duration": 30,
            "requests_per_second": 5
        },
        "medium_load": {
            "concurrent_users": 50,
            "duration": 60,
            "requests_per_second": 25
        },
        "heavy_load": {
            "concurrent_users": 100,
            "duration": 120,
            "requests_per_second": 50
        },
        "stress_test": {
            "concurrent_users": 200,
            "duration": 180,
            "requests_per_second": 100
        }
    }


@pytest.fixture
def security_test_payloads():
    """Security testing payloads."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users--",
            "'; INSERT INTO users VALUES ('hacker','password'); --"
        ],
        "xss_payloads": [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')></svg>"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ],
        "command_injection": [
            "; cat /etc/passwd",
            "| whoami",
            "&& rm -rf /",
            "`id`"
        ],
        "nosql_injection": [
            "'; return db.users.find(); var dummy='",
            "{\"$ne\": null}",
            "{\"$regex\": \".*\"}",
            "{\"$where\": \"function() { return true; }\"}"
        ]
    }