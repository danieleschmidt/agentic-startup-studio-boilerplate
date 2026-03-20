"""Tests for Agentic Startup Studio Boilerplate."""

import pytest
import tempfile
import json
from pathlib import Path

from studio.cookiecutter_template import CookiecutterTemplate
from studio.agent_scaffold import AgentWorkflowScaffold
from studio.fastapi_scaffold import FastAPIScaffold
from studio.react_scaffold import ReactScaffold
from studio.cli import cmd_new


# ── CookiecutterTemplate ───────────────────────────────────────────────────────

class TestCookiecutterTemplate:
    def test_render_string_basic(self):
        tmpl = CookiecutterTemplate({"name": "Alice"})
        assert tmpl.render_string("Hello {{name}}!") == "Hello Alice!"

    def test_render_path(self):
        tmpl = CookiecutterTemplate({"project_slug": "my-app"})
        assert tmpl.render_path("{{project_slug}}/main.py") == "my-app/main.py"

    def test_unknown_placeholder_preserved(self):
        tmpl = CookiecutterTemplate({})
        result = tmpl.render_string("{{unknown}}")
        assert result == "{{unknown}}"

    def test_generate_files(self, tmp_path):
        tmpl = CookiecutterTemplate({"name": "Test"})
        templates = {
            "README.md": "# {{name}}",
            "src/main.py": "# {{name}} main",
        }
        created = tmpl.generate_files(templates, str(tmp_path))
        assert len(created) == 2
        assert (tmp_path / "README.md").read_text() == "# Test"
        assert (tmp_path / "src/main.py").read_text() == "# Test main"

    def test_add_context(self):
        tmpl = CookiecutterTemplate({"a": "1"})
        tmpl.add_context("b", "2")
        assert tmpl.get_context()["b"] == "2"

    def test_get_context_copy(self):
        tmpl = CookiecutterTemplate({"a": "1"})
        ctx = tmpl.get_context()
        ctx["a"] = "modified"
        assert tmpl.get_context()["a"] == "1"


# ── AgentWorkflowScaffold ──────────────────────────────────────────────────────

class TestAgentWorkflowScaffold:
    def _config(self):
        return {
            "agents": [
                {
                    "name": "researcher",
                    "role": "Research Analyst",
                    "goal": "Research topics",
                    "backstory": "Expert researcher",
                },
                {
                    "name": "writer",
                    "role": "Content Writer",
                    "goal": "Write content",
                    "backstory": "Skilled writer",
                },
            ]
        }

    def test_generate_agent_file(self):
        scaffold = AgentWorkflowScaffold(self._config())
        code = scaffold.generate_agent_file(self._config()["agents"][0])
        assert "researcher" in code
        assert "Agent(" in code
        assert "Research Analyst" in code

    def test_generate_crew_file(self):
        scaffold = AgentWorkflowScaffold(self._config())
        code = scaffold.generate_crew_file()
        assert "Crew(" in code
        assert "researcher" in code
        assert "writer" in code

    def test_generate_all_returns_expected_keys(self):
        scaffold = AgentWorkflowScaffold(self._config())
        files = scaffold.generate_all()
        assert "crew.py" in files
        assert any("researcher" in k for k in files)

    def test_generate_task_file(self):
        scaffold = AgentWorkflowScaffold(self._config())
        agent = self._config()["agents"][0]
        task = {"name": "research_task", "description": "Do research", "expected_output": "Report"}
        code = scaffold.generate_task_file(agent, task)
        assert "Task(" in code
        assert "Do research" in code


# ── FastAPIScaffold ────────────────────────────────────────────────────────────

class TestFastAPIScaffold:
    def _config(self):
        return {"project_name": "TestApp", "resources": ["User", "Product"]}

    def test_generate_router_has_crud(self):
        scaffold = FastAPIScaffold(self._config())
        code = scaffold.generate_router("User")
        assert "list_user" in code
        assert "get_user" in code
        assert "create_user" in code
        assert "update_user" in code
        assert "delete_user" in code

    def test_generate_main(self):
        scaffold = FastAPIScaffold(self._config())
        code = scaffold.generate_main()
        assert "FastAPI(" in code
        assert "TestApp" in code

    def test_generate_all_keys(self):
        scaffold = FastAPIScaffold(self._config())
        files = scaffold.generate_all()
        assert "main.py" in files
        assert "routers/user_router.py" in files
        assert "routers/product_router.py" in files


# ── ReactScaffold ──────────────────────────────────────────────────────────────

class TestReactScaffold:
    def _config(self):
        return {"project_name": "TestApp", "components": ["Dashboard", "UserList"]}

    def test_generate_component(self):
        scaffold = ReactScaffold(self._config())
        code = scaffold.generate_component("Dashboard")
        assert "Dashboard" in code
        assert "useState" in code
        assert "useEffect" in code

    def test_generate_app(self):
        scaffold = ReactScaffold(self._config())
        code = scaffold.generate_app()
        assert "Dashboard" in code
        assert "UserList" in code
        assert "TestApp" in code

    def test_generate_package_json(self):
        scaffold = ReactScaffold(self._config())
        pkg = json.loads(scaffold.generate_package_json())
        assert "react" in pkg["dependencies"]
        assert "start" in pkg["scripts"]

    def test_generate_all_keys(self):
        scaffold = ReactScaffold(self._config())
        files = scaffold.generate_all()
        assert "src/App.tsx" in files
        assert "package.json" in files
        assert "src/components/Dashboard.tsx" in files


# ── CLI: cmd_new ───────────────────────────────────────────────────────────────

class TestCmdNew:
    def test_creates_project_directory(self, tmp_path):
        project_dir, files = cmd_new("My Startup", str(tmp_path))
        assert project_dir.exists()
        assert project_dir.is_dir()

    def test_creates_readme(self, tmp_path):
        project_dir, files = cmd_new("My Startup", str(tmp_path))
        readme = project_dir / "README.md"
        assert readme.exists()
        assert "My Startup" in readme.read_text()

    def test_creates_backend_files(self, tmp_path):
        project_dir, files = cmd_new("My Startup", str(tmp_path))
        assert (project_dir / "backend" / "main.py").exists()
        assert (project_dir / "backend" / "requirements.txt").exists()

    def test_creates_agent_files(self, tmp_path):
        project_dir, files = cmd_new("My Startup", str(tmp_path))
        assert (project_dir / "backend" / "crew.py").exists()

    def test_creates_frontend_files(self, tmp_path):
        project_dir, files = cmd_new("My Startup", str(tmp_path))
        assert (project_dir / "frontend" / "package.json").exists()
        assert (project_dir / "frontend" / "src" / "App.tsx").exists()

    def test_project_slug(self, tmp_path):
        project_dir, files = cmd_new("My Cool App", str(tmp_path))
        assert project_dir.name == "my-cool-app"
