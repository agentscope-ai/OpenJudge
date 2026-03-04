# -*- coding: utf-8 -*-
"""
OpenJudge Agentic Infrastructure Package.

This package provides the foundational infrastructure for agentic capabilities
in OpenJudge, including tool systems and agent implementations. It enables
tool-augmented evaluation where LLMs can autonomously call tools (web search,
code execution, etc.) to gather information for making judgments.

Architecture:
    The agentic system is organized into three layers:

    1. Tool Layer (tools.py):
       - BaseTool: Abstract base class defining tool interface
       - ToolResult: Standardized result format for tool execution
       - Defines WHAT capabilities the agent has

    2. Agent Layer (agents.py):
       - BaseAgent: Abstract base class defining agent interface
       - AgentResult: Standardized result format for agent execution
       - ReActAgent: Built-in ReAct implementation
       - Defines HOW the agent thinks and reasons

    3. Adapter Layer (adapters/):
       - FunctionToolAdapter: Wrap Python functions as tools
       - LangChainToolAdapter, LangChainAgentAdapter: LangChain integration
       - AgentScopeToolAdapter, AgentScopeAgentAdapter: AgentScope integration
       - Enables integration with external frameworks

Design Principle:
    AgenticGrader follows "unified interface" design - it only accepts a
    pre-built agent parameter. Whether using built-in ReActAgent or external
    framework adapters, the agent must be constructed externally first.

Core Components (exported from this module):
    BaseTool: Abstract base class for all tools.
    ToolResult: Standardized result format for tool execution.
    BaseAgent: Abstract base class for all agents.
    AgentResult: Standardized result format for agent execution.
    ReActAgent: Built-in ReAct agent implementation.

Adapters (import from their modules):
    >>> # Core adapter (always available)
    >>> from openjudge.agentic.adapters.function import FunctionToolAdapter
    >>>
    >>> # External framework adapters (see cookbooks/agentic_grader/adapters/)
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainToolAdapter
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainAgentAdapter
    >>> from cookbooks.agentic_grader.adapters.agentscope import AgentScopeToolAdapter
    >>> from cookbooks.agentic_grader.adapters.agentscope import AgentScopeAgentAdapter

Example - Creating a Custom Tool:
    >>> from openjudge.agentic import BaseTool, ToolResult
    >>>
    >>> class WebSearchTool(BaseTool):
    ...     schema = {
    ...         "type": "function",
    ...         "function": {
    ...             "name": "web_search",
    ...             "description": "Search the web for information",
    ...             "parameters": {
    ...                 "type": "object",
    ...                 "properties": {
    ...                     "query": {"type": "string", "description": "Search query"}
    ...                 },
    ...                 "required": ["query"]
    ...             }
    ...         }
    ...     }
    ...
    ...     async def aexecute(self, query: str, **kwargs) -> ToolResult:
    ...         # Perform search
    ...         return ToolResult(success=True, output="search results...")

Example - Using Built-in ReActAgent:
    >>> from openjudge.agentic import ReActAgent
    >>>
    >>> agent = ReActAgent(
    ...     model=my_model,
    ...     tools=[WebSearchTool()],
    ...     max_iterations=10,
    ... )
    >>> result = await agent.arun(messages)
    >>> print(result.content)

Example - Wrapping a Function as a Tool:
    >>> from openjudge.agentic.adapters.function import FunctionToolAdapter
    >>>
    >>> def calculate(expression: str) -> float:
    ...     return eval(expression)
    >>>
    >>> tool = FunctionToolAdapter(
    ...     func=calculate,
    ...     name="calculator",
    ...     description="Evaluate a mathematical expression",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {
    ...             "expression": {"type": "string", "description": "Math expression"}
    ...         },
    ...         "required": ["expression"]
    ...     }
    ... )

Example - Using LangChain Tools:
    >>> from langchain_community.tools import DuckDuckGoSearchRun
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainToolAdapter
    >>> from openjudge.agentic import ReActAgent
    >>>
    >>> lc_tool = DuckDuckGoSearchRun()
    >>> oj_tool = LangChainToolAdapter(lc_tool)
    >>> agent = ReActAgent(model=my_model, tools=[oj_tool])

Example - Using External Agent:
    >>> from langchain.agents import create_agent
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainAgentAdapter
    >>>
    >>> lc_agent = create_agent(llm, tools)
    >>> oj_agent = LangChainAgentAdapter(lc_agent)
    >>> # Use oj_agent with AgenticGrader
"""

# Agent system
from .agents import AgentResult, BaseAgent, ReActAgent

# Tool system
from .tools import BaseTool, ToolResult

__all__ = [
    # Tool system
    "BaseTool",
    "ToolResult",
    # Agent system
    "BaseAgent",
    "AgentResult",
    "ReActAgent",
]
