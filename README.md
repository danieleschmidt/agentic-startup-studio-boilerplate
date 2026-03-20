# agentic-startup-studio-boilerplate

Cookiecutter-style project generator for agentic startups using CrewAI, FastAPI, and React.

## Features

- **CookiecutterTemplate** – `{{variable}}` file generation
- **AgentWorkflowScaffold** – CrewAI agent/task/crew code from config
- **FastAPIScaffold** – Full CRUD router stubs
- **ReactScaffold** – TypeScript component stubs
- **CLI** – `studio new <name>` creates complete project skeleton

## Usage

```bash
pip install -e .
studio new "My AI Startup"
```

This generates:
```
my-ai-startup/
  README.md
  backend/
    main.py
    requirements.txt
    agents/researcher.py
    agents/writer.py
    tasks/
    crew.py
    routers/
  frontend/
    package.json
    src/App.tsx
    src/components/
```

## Development

```bash
pytest tests/
```
