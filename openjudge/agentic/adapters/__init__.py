# -*- coding: utf-8 -*-
"""
Adapters for integrating third-party frameworks with OpenJudge.

This package provides adapter classes that bridge external agent frameworks
(LangChain, AgentScope, etc.) with OpenJudge's tool and agent abstractions.
The design is fully pluggable - you can add or remove adapters without
affecting the core functionality.

Available Adapters:
    FunctionToolAdapter: Wrap simple Python functions as OpenJudge tools.
    LangChainToolAdapter: Wrap LangChain tools for use with OpenJudge agents.
    LangChainAgentAdapter: Use LangChain agents as OpenJudge agents.
    AgentScopeToolAdapter: Wrap AgentScope tools for use with OpenJudge agents.
    AgentScopeAgentAdapter: Use AgentScope agents as OpenJudge agents.

Usage:
    Import adapters directly from their respective modules:

    >>> from openjudge.agentic.adapters.function import FunctionToolAdapter
    >>> from openjudge.agentic.adapters.langchain import LangChainToolAdapter
    >>> from openjudge.agentic.adapters.langchain import LangChainAgentAdapter
    >>> from openjudge.agentic.adapters.agentscope import AgentScopeToolAdapter
    >>> from openjudge.agentic.adapters.agentscope import AgentScopeAgentAdapter

Pluggable Design:
    - Delete langchain.py if you don't need LangChain/LangGraph support
    - Delete agentscope.py if you don't need AgentScope support
    - Add new adapters by creating a new file following the same pattern

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
