# -*- coding: utf-8 -*-
"""
Adapters for integrating third-party frameworks with OpenJudge.

This package provides adapter classes that bridge external agent frameworks
with OpenJudge's tool and agent abstractions. The design is fully pluggable -
you can add or remove adapters without affecting the core functionality.

Core Adapters (in this package):
    FunctionToolAdapter: Wrap simple Python functions as OpenJudge tools.

External Framework Adapters (in cookbooks/agentic_grader/adapters/):
    LangChainToolAdapter: Wrap LangChain tools for use with OpenJudge agents.
    LangChainAgentAdapter: Use LangChain agents as OpenJudge agents.
    AgentScopeToolAdapter: Wrap AgentScope tools for use with OpenJudge agents.
    AgentScopeAgentAdapter: Use AgentScope agents as OpenJudge agents.

Usage:
    >>> # Core adapter
    >>> from openjudge.agentic.adapters.function import FunctionToolAdapter
    >>>
    >>> # External framework adapters (see cookbooks for examples)
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainToolAdapter
    >>> from cookbooks.agentic_grader.adapters.agentscope import AgentScopeAgentAdapter

Example:
    >>> # Wrap a simple function as a tool
    >>> from openjudge.agentic.adapters.function import FunctionToolAdapter
    >>>
    >>> def search_web(query: str) -> str:
    ...     return "search results..."
    >>>
    >>> tool = FunctionToolAdapter(
    ...     func=search_web,
    ...     name="web_search",
    ...     description="Search the web for information",
    ...     parameters={
    ...         "type": "object",
    ...         "properties": {"query": {"type": "string"}},
    ...         "required": ["query"]
    ...     }
    ... )
"""

from .function import FunctionToolAdapter

__all__ = [
    "FunctionToolAdapter",
]
