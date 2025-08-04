"""
CrewAI orchestration and agent definitions.
Implements the core AI agent workflow for the application.
"""

from typing import Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.agents.tools import (
    WebSearchTool,
    DataAnalysisTool,
    ReportGeneratorTool,
)


class AgentRequest(BaseModel):
    """Request model for agent operations."""
    
    task_description: str = Field(..., description="Description of the task to perform")
    context: Optional[Dict] = Field(default=None, description="Additional context for the task")
    priority: str = Field(default="normal", description="Task priority: low, normal, high")
    deadline: Optional[str] = Field(default=None, description="Task deadline in ISO format")


class AgentResponse(BaseModel):
    """Response model for agent operations."""
    
    success: bool = Field(..., description="Whether the task was completed successfully")
    result: str = Field(..., description="The result of the task execution")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    execution_time: float = Field(..., description="Time taken to execute the task")
    agents_used: List[str] = Field(..., description="List of agents that participated")


class ResearchAgent:
    """Research agent for gathering and analyzing information."""
    
    def __init__(self, llm):
        self.agent = Agent(
            role='Research Analyst',
            goal='Gather comprehensive information and insights on given topics',
            backstory="""You are an expert research analyst with years of experience in 
            information gathering, data analysis, and insight generation. You excel at 
            finding relevant information from multiple sources and synthesizing it into 
            actionable insights.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[WebSearchTool(), DataAnalysisTool()]
        )


class AnalysisAgent:
    """Analysis agent for processing and interpreting data."""
    
    def __init__(self, llm):
        self.agent = Agent(
            role='Data Analyst',
            goal='Process and analyze data to extract meaningful patterns and insights',
            backstory="""You are a skilled data analyst with expertise in statistical analysis, 
            pattern recognition, and data interpretation. You can transform raw data into 
            valuable business insights and recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[DataAnalysisTool()]
        )


class ReportAgent:
    """Report agent for generating comprehensive reports."""
    
    def __init__(self, llm):
        self.agent = Agent(
            role='Report Specialist',
            goal='Create detailed, professional reports based on research and analysis',
            backstory="""You are an expert report writer with extensive experience in 
            creating clear, concise, and actionable reports. You excel at organizing 
            complex information into well-structured documents.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[ReportGeneratorTool()]
        )


class AgentCrew:
    """Main crew orchestrator for coordinating AI agents."""
    
    def __init__(self):
        """Initialize the agent crew with configured LLM."""
        settings = get_settings()
        
        # Initialize LLM based on available API keys
        if settings.openai_api_key:
            self.llm = OpenAI(openai_api_key=settings.openai_api_key)
        elif settings.anthropic_api_key:
            # For production, you might want to use Anthropic's LLM
            self.llm = OpenAI(openai_api_key=settings.openai_api_key)  # Fallback
        else:
            raise ValueError("No LLM API key configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        # Initialize agents
        self.research_agent = ResearchAgent(self.llm)
        self.analysis_agent = AnalysisAgent(self.llm)
        self.report_agent = ReportAgent(self.llm)
        
        # Create crew
        self.crew = Crew(
            agents=[
                self.research_agent.agent,
                self.analysis_agent.agent,
                self.report_agent.agent
            ],
            process=Process.sequential,
            verbose=True
        )
    
    async def execute_research_task(self, request: AgentRequest) -> AgentResponse:
        """
        Execute a research task using the agent crew.
        
        Args:
            request: The research request
            
        Returns:
            AgentResponse: The result of the research task
        """
        import time
        start_time = time.time()
        
        try:
            # Create tasks based on the request
            tasks = self._create_research_tasks(request)
            
            # Execute the crew with tasks
            result = self.crew.kickoff(tasks=tasks)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                success=True,
                result=str(result),
                metadata={
                    "task_type": "research",
                    "priority": request.priority,
                    "context": request.context or {}
                },
                execution_time=execution_time,
                agents_used=["research_agent", "analysis_agent", "report_agent"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResponse(
                success=False,
                result=f"Error executing research task: {str(e)}",
                metadata={"error_type": type(e).__name__},
                execution_time=execution_time,
                agents_used=[]
            )
    
    def _create_research_tasks(self, request: AgentRequest) -> List[Task]:
        """
        Create tasks for the research workflow.
        
        Args:
            request: The research request
            
        Returns:
            List[Task]: List of tasks for the crew
        """
        tasks = []
        
        # Research task
        research_task = Task(
            description=f"""
            Conduct comprehensive research on: {request.task_description}
            
            Context: {request.context or 'No additional context provided'}
            Priority: {request.priority}
            
            Your research should include:
            1. Key facts and data points
            2. Current trends and developments
            3. Relevant case studies or examples
            4. Potential challenges and opportunities
            
            Provide a detailed research summary with sources and insights.
            """,
            agent=self.research_agent.agent,
            expected_output="Comprehensive research summary with key findings and sources"
        )
        tasks.append(research_task)
        
        # Analysis task
        analysis_task = Task(
            description="""
            Analyze the research findings and extract actionable insights.
            
            Your analysis should include:
            1. Pattern identification and trend analysis
            2. Risk assessment and opportunity evaluation
            3. Strategic recommendations
            4. Quantitative insights where applicable
            
            Build upon the research findings to provide strategic insights.
            """,
            agent=self.analysis_agent.agent,
            expected_output="Strategic analysis with actionable insights and recommendations"
        )
        tasks.append(analysis_task)
        
        # Report task
        report_task = Task(
            description="""
            Create a professional, comprehensive report based on the research and analysis.
            
            The report should include:
            1. Executive summary
            2. Key findings and insights
            3. Detailed analysis and recommendations
            4. Conclusion and next steps
            
            Format the report in a clear, professional manner suitable for stakeholders.
            """,
            agent=self.report_agent.agent,
            expected_output="Professional report with executive summary, findings, and recommendations"
        )
        tasks.append(report_task)
        
        return tasks


# Global crew instance
_crew_instance: Optional[AgentCrew] = None


def get_agent_crew() -> AgentCrew:
    """
    Get the global agent crew instance.
    
    Returns:
        AgentCrew: The initialized agent crew
    """
    global _crew_instance
    if _crew_instance is None:
        _crew_instance = AgentCrew()
    return _crew_instance