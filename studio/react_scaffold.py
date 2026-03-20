"""ReactScaffold: generate React component stubs."""

from typing import Any, Dict, List, Optional


COMPONENT_TEMPLATE = '''\
import React, {{ useState, useEffect }} from 'react';

interface {name}Props {{
  // Add props here
}}

const {name}: React.FC<{name}Props> = (props) => {{
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {{
    setLoading(true);
    fetch('/api/{resource_lower}/')
      .then(r => r.json())
      .then(d => {{ setData(d); setLoading(false); }})
      .catch(e => {{ setError(e.message); setLoading(false); }});
  }}, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {{error}}</div>;

  return (
    <div className="{name_lower}-container">
      <h2>{name}</h2>
      <ul>
        {{data.map((item, i) => <li key={{i}}>{{JSON.stringify(item)}}</li>)}}
      </ul>
    </div>
  );
}};

export default {name};
'''

APP_TEMPLATE = '''\
import React from 'react';
{imports}

function App() {{
  return (
    <div className="app">
      <h1>{{project_name}}</h1>
{component_uses}
    </div>
  );
}}

export default App;
'''


class ReactScaffold:
    """Generate React component stubs from a config."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: dict with 'components' list and 'project_name'
        """
        self.config = config

    def generate_component(self, name: str) -> str:
        """Generate a React component for a named resource."""
        return COMPONENT_TEMPLATE.format(
            name=name,
            name_lower=name.lower(),
            resource_lower=name.lower(),
        )

    def generate_app(self) -> str:
        """Generate the root App.tsx."""
        components = self.config.get("components", [])
        project_name = self.config.get("project_name", "MyApp")
        imports = "\n".join([f"import {c} from './components/{c}';" for c in components])
        uses = "\n".join([f"      <{c} />" for c in components])
        # Use str.replace for all substitutions to avoid .format() KeyErrors with arbitrary values
        return (
            APP_TEMPLATE
            .replace("{project_name}", project_name)
            .replace("{imports}", imports)
            .replace("{component_uses}", uses)
        )

    def generate_package_json(self) -> str:
        """Generate a minimal package.json."""
        import json
        name = self.config.get("project_name", "my-app").lower().replace(" ", "-")
        return json.dumps({
            "name": name,
            "version": "0.1.0",
            "private": True,
            "dependencies": {
                "react": "^18.0.0",
                "react-dom": "^18.0.0",
                "typescript": "^5.0.0",
            },
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
            },
        }, indent=2)

    def generate_all(self) -> Dict[str, str]:
        """Generate all React files. Returns {filename: content}."""
        files = {}
        for component in self.config.get("components", []):
            files[f"src/components/{component}.tsx"] = self.generate_component(component)
        files["src/App.tsx"] = self.generate_app()
        files["package.json"] = self.generate_package_json()
        return files
