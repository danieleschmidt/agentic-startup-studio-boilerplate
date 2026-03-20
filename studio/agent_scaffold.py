"""AgentWorkflowScaffold: generate agent roles from config."""

from typing import Any, Dict, List, Optional


AGENT_TEMPLATE = '''\
# {name} Agent
from crewai import Agent

{variable_name} = Agent(
    role="{role}",
    goal="{goal}",
    backstory="{backstory}",
    verbose=True,
)
'''

TASK_TEMPLATE = '''\
from crewai import Task

{variable_name}_task = Task(
    description="{description}",
    agent={agent_var},
    expected_output="{expected_output}",
)
'''

CREW_TEMPLATE = '''\
from crewai import Crew, Process

crew = Crew(
    agents=[{agents}],
    tasks=[{tasks}],
    process=Process.sequential,
    verbose=True,
)
'''


class AgentWorkflowScaffold:
    """Generate CrewAI agent workflow code from a config."""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: dict with 'agents' list, each having name/role/goal/backstory
        """
        self.config = config

    def generate_agent_file(self, agent: Dict[str, str]) -> str:
        """Generate code for a single agent."""
        var = agent["name"].lower().replace(" ", "_").replace("-", "_")
        return AGENT_TEMPLATE.format(
            name=agent["name"],
            variable_name=var,
            role=agent.get("role", agent["name"]),
            goal=agent.get("goal", f"Complete {agent['name']} tasks"),
            backstory=agent.get("backstory", f"Expert {agent['name']}"),
        )

    def generate_task_file(self, agent: Dict[str, str], task: Dict[str, str]) -> str:
        """Generate code for a task."""
        agent_var = agent["name"].lower().replace(" ", "_").replace("-", "_")
        task_var = task["name"].lower().replace(" ", "_").replace("-", "_")
        return TASK_TEMPLATE.format(
            variable_name=task_var,
            description=task.get("description", f"Execute {task['name']}"),
            agent_var=agent_var,
            expected_output=task.get("expected_output", "Completed output"),
        )

    def generate_crew_file(self) -> str:
        """Generate the Crew assembly file."""
        agents = self.config.get("agents", [])
        agent_vars = [a["name"].lower().replace(" ", "_").replace("-", "_") for a in agents]
        tasks_vars = [v + "_task" for v in agent_vars]
        return CREW_TEMPLATE.format(
            agents=", ".join(agent_vars),
            tasks=", ".join(tasks_vars),
        )

    def generate_all(self) -> Dict[str, str]:
        """Generate all agent/task/crew files. Returns {filename: content}."""
        files = {}
        for agent in self.config.get("agents", []):
            var = agent["name"].lower().replace(" ", "_").replace("-", "_")
            files[f"agents/{var}.py"] = self.generate_agent_file(agent)
            tasks = agent.get("tasks", [{"name": var, "description": f"Execute {var} workflow"}])
            for task in tasks:
                files[f"tasks/{var}_task.py"] = self.generate_task_file(agent, task)
        files["crew.py"] = self.generate_crew_file()
        return files
