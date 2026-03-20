"""FastAPIScaffold: generate route stubs for a FastAPI application."""

from typing import Any, Dict, List, Optional


MAIN_TEMPLATE = '''\
"""FastAPI application for {{project_name}}."""

from fastapi import FastAPI
from .routers import {router_imports}

app = FastAPI(title="{{project_name}}", version="0.1.0")

{include_routers}

@app.get("/health")
async def health():
    return {{"status": "ok"}}
'''

ROUTER_TEMPLATE = '''\
"""Router for {resource} endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional

router = APIRouter(prefix="/{resource_lower}", tags=["{resource}"])


@router.get("/", response_model=List[dict])
async def list_{resource_lower}():
    """List all {resource} records."""
    return []


@router.get("/{{item_id}}", response_model=dict)
async def get_{resource_lower}(item_id: int):
    """Get a single {resource} record."""
    return {{"id": item_id}}


@router.post("/", response_model=dict, status_code=201)
async def create_{resource_lower}(data: dict):
    """Create a new {resource} record."""
    return {{"id": 1, **data}}


@router.put("/{{item_id}}", response_model=dict)
async def update_{resource_lower}(item_id: int, data: dict):
    """Update a {resource} record."""
    return {{"id": item_id, **data}}


@router.delete("/{{item_id}}", status_code=204)
async def delete_{resource_lower}(item_id: int):
    """Delete a {resource} record."""
    return None
'''


class FastAPIScaffold:
    """Generate FastAPI route stubs from a config."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: dict with 'resources' list (e.g. ['User', 'Product'])
        """
        self.config = config

    def generate_router(self, resource: str) -> str:
        """Generate a router file for a resource."""
        return ROUTER_TEMPLATE.format(
            resource=resource,
            resource_lower=resource.lower(),
        )

    def generate_main(self) -> str:
        """Generate the main FastAPI app file."""
        resources = self.config.get("resources", [])
        project_name = self.config.get("project_name", "MyApp")
        router_imports = ", ".join([f"{r.lower()}_router" for r in resources])
        include_lines = "\n".join(
            [f'app.include_router({r.lower()}_router.router)' for r in resources]
        )
        # Use a safe placeholder to avoid .format() consuming {project_name}
        raw = MAIN_TEMPLATE.replace("{{project_name}}", "__PROJECT_NAME__")
        rendered = raw.format(router_imports=router_imports, include_routers=include_lines)
        return rendered.replace("__PROJECT_NAME__", project_name)

    def generate_all(self) -> Dict[str, str]:
        """Generate all FastAPI files. Returns {filename: content}."""
        files = {}
        resources = self.config.get("resources", [])
        for resource in resources:
            files[f"routers/{resource.lower()}_router.py"] = self.generate_router(resource)
        files["main.py"] = self.generate_main()
        return files
