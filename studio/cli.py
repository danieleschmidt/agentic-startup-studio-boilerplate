"""Studio CLI: `studio new <name>` → writes complete project skeleton."""

import argparse
import sys
from pathlib import Path

from .cookiecutter_template import CookiecutterTemplate
from .agent_scaffold import AgentWorkflowScaffold
from .fastapi_scaffold import FastAPIScaffold
from .react_scaffold import ReactScaffold


DEFAULT_AGENTS = [
    {
        "name": "researcher",
        "role": "Research Analyst",
        "goal": "Gather and analyze information",
        "backstory": "Expert at finding insights from data",
        "tasks": [{"name": "research", "description": "Research the topic", "expected_output": "Research report"}],
    },
    {
        "name": "writer",
        "role": "Content Writer",
        "goal": "Produce high-quality written content",
        "backstory": "Skilled writer with domain expertise",
        "tasks": [{"name": "write", "description": "Write the content", "expected_output": "Written document"}],
    },
]

DEFAULT_RESOURCES = ["User", "Task", "Project"]
DEFAULT_COMPONENTS = ["Dashboard", "TaskList", "UserProfile"]

PROJECT_TEMPLATES = {
    "README.md": """\
# {{project_name}}

An agentic startup built with CrewAI, FastAPI, and React.

## Stack
- **Agents**: CrewAI multi-agent workflow
- **Backend**: FastAPI REST API
- **Frontend**: React TypeScript

## Setup

```bash
# Backend
cd backend && pip install -r requirements.txt && uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm start
```

## Architecture
- `backend/agents/` — CrewAI agent definitions
- `backend/tasks/` — CrewAI task definitions
- `backend/crew.py` — Crew assembly
- `backend/routers/` — FastAPI route handlers
- `frontend/src/components/` — React components
""",
    "backend/requirements.txt": """\
crewai>=0.28.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
""",
    ".gitignore": """\
__pycache__/
*.pyc
.env
node_modules/
dist/
build/
*.egg-info/
""",
}


def cmd_new(name: str, output_dir: str = "."):
    """Create a new agentic startup project skeleton."""
    project_slug = name.lower().replace(" ", "-").replace("_", "-")
    project_dir = Path(output_dir) / project_slug

    context = {
        "project_name": name,
        "project_slug": project_slug,
    }

    # 1. Render base templates
    tmpl = CookiecutterTemplate(context)
    tmpl.generate_files(PROJECT_TEMPLATES, str(project_dir))

    # 2. Agent scaffold
    agent_config = {"agents": DEFAULT_AGENTS}
    agent_scaffold = AgentWorkflowScaffold(agent_config)
    agent_files = agent_scaffold.generate_all()
    for rel_path, content in agent_files.items():
        out = project_dir / "backend" / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)

    # 3. FastAPI scaffold
    api_config = {"project_name": name, "resources": DEFAULT_RESOURCES}
    api_scaffold = FastAPIScaffold(api_config)
    api_files = api_scaffold.generate_all()
    for rel_path, content in api_files.items():
        out = project_dir / "backend" / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)

    # 4. React scaffold
    react_config = {"project_name": name, "components": DEFAULT_COMPONENTS}
    react_scaffold = ReactScaffold(react_config)
    react_files = react_scaffold.generate_all()
    for rel_path, content in react_files.items():
        out = project_dir / "frontend" / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)

    all_files = (
        list(PROJECT_TEMPLATES.keys())
        + [f"backend/{p}" for p in agent_files]
        + [f"backend/{p}" for p in api_files]
        + [f"frontend/{p}" for p in react_files]
    )
    return project_dir, all_files


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="studio",
        description="Agentic Startup Studio — scaffold a new agentic startup",
    )
    subparsers = parser.add_subparsers(dest="command")

    new_parser = subparsers.add_parser("new", help="Create a new project")
    new_parser.add_argument("name", help="Project name")
    new_parser.add_argument("--output", "-o", default=".", help="Output directory")

    args = parser.parse_args(argv)

    if args.command == "new":
        project_dir, files = cmd_new(args.name, args.output)
        print(f"✓ Created project: {project_dir}")
        print(f"  {len(files)} files generated")
        for f in files[:8]:
            print(f"  + {f}")
        if len(files) > 8:
            print(f"  ... and {len(files) - 8} more")
        return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
