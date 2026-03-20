"""CookiecutterTemplate: parameterized file generation for project skeletons."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class CookiecutterTemplate:
    """Generate project files from parameterized templates."""

    PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

    def __init__(self, context: Dict[str, Any]):
        """
        Args:
            context: template variables, e.g. {'project_name': 'my-app', 'author': 'Alice'}
        """
        self.context = context

    def render_string(self, template: str) -> str:
        """Substitute {{var}} placeholders in a string."""
        def replace(m):
            key = m.group(1)
            return str(self.context.get(key, m.group(0)))
        return self.PLACEHOLDER_RE.sub(replace, template)

    def render_path(self, path: str) -> str:
        """Substitute placeholders in a file path."""
        return self.render_string(path)

    def generate_files(self, templates: Dict[str, str], output_dir: str) -> List[str]:
        """
        Generate files from a dict of {relative_path: content_template}.
        Returns list of created file paths.
        """
        output_path = Path(output_dir)
        created = []
        for rel_path, content_template in templates.items():
            out_rel = self.render_path(rel_path)
            out_file = output_path / out_rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            content = self.render_string(content_template)
            out_file.write_text(content)
            created.append(str(out_file))
        return created

    def add_context(self, key: str, value: Any):
        self.context[key] = value

    def get_context(self) -> Dict[str, Any]:
        return dict(self.context)
