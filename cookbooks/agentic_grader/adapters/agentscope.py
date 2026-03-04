# -*- coding: utf-8 -*-
"""
AgentScope adapters for tools and agents.

This module provides adapters for integrating the AgentScope framework with OpenJudge.
It allows you to use AgentScope's tool functions with OpenJudge's built-in agents,
or use existing AgentScope agents directly in OpenJudge evaluations.

Classes:
    AgentScopeToolAdapter: Adapt AgentScope tool functions to OpenJudge BaseTool.
    AgentScopeAgentAdapter: Adapt AgentScope agents to OpenJudge BaseAgent.

Dependencies:
    This module requires agentscope to be installed:
    pip install agentscope

Example:
    >>> # Use AgentScope tools with OpenJudge's ReActAgent
    >>> from agentscope.tool import Toolkit
    >>> from cookbooks.agentic_grader.adapters.agentscope import AgentScopeToolAdapter
    >>> from openjudge.agentic import ReActAgent
    >>>
    >>> toolkit = Toolkit()
    >>> @toolkit.register
    >>> def my_tool(query: str) -> str:
    ...     return "result"
    >>>
    >>> oj_tool = AgentScopeToolAdapter(my_tool)
    >>> agent = ReActAgent(model=my_model, tools=[oj_tool])
    >>>
    >>> # Use AgentScope agent directly in OpenJudge
    >>> from agentscope.agent import ReActAgent as ASReActAgent
    >>> from cookbooks.agentic_grader.adapters.agentscope import AgentScopeAgentAdapter
    >>>
    >>> as_agent = ASReActAgent(name="evaluator", ...)
    >>> oj_agent = AgentScopeAgentAdapter(as_agent)
    >>> grader = AgenticGrader(agent=oj_agent, template="...")
"""

import asyncio
from typing import Any, Callable, Dict, List, Union

from loguru import logger

from openjudge.agentic.agents import AgentResult, BaseAgent
from openjudge.agentic.tools import BaseTool, ToolResult

__all__ = [
    "AgentScopeToolAdapter",
    "AgentScopeAgentAdapter",
]


class AgentScopeToolAdapter(BaseTool):
    """Adapter for wrapping an AgentScope tool function as an OpenJudge BaseTool.

    Use this when you want to use AgentScope's tool functions with OpenJudge's
    built-in ReActAgent. This adapter handles the conversion between AgentScope's
    tool interface and OpenJudge's BaseTool interface.

    Note:
        This adapter wraps individual tool functions, not Toolkit instances.
        If you have a Toolkit, extract the individual functions first.

    Attributes:
        schema: OpenAI function calling schema extracted from the tool function.

    Example:
        >>> from agentscope.tool import Toolkit
        >>> from openjudge.agentic.adapters.agentscope import AgentScopeToolAdapter
        >>>
        >>> toolkit = Toolkit()
        >>> @toolkit.register
        >>> def search_web(query: str) -> str:
        ...     '''Search the web for information.'''
        ...     return "search results..."
        >>>
        >>> # Wrap the tool function (not the toolkit)
        >>> oj_tool = AgentScopeToolAdapter(search_web)
        >>>
        >>> # Use with OpenJudge's ReActAgent
        >>> from openjudge.agentic import ReActAgent
        >>> agent = ReActAgent(model=model, tools=[oj_tool])
    """

    def __init__(self, as_tool: Union[Callable[..., Any], Any]):
        """Initialize the AgentScopeToolAdapter.

        Args:
            as_tool: An AgentScope tool function. Should be a callable that
                    was registered with an AgentScope Toolkit.

        Raises:
            ImportError: If agentscope is not installed.
            ValueError: If a Toolkit instance is passed instead of a function.
        """
        try:
            from agentscope.tool import Toolkit
        except ImportError as exc:
            raise ImportError(
                "AgentScope is required for this adapter. " + "Install it with: pip install agentscope"
            ) from exc

        if isinstance(as_tool, Toolkit):
            raise ValueError(
                "Cannot convert a Toolkit directly. "
                "Please extract individual tool functions first. "
                "Example: AgentScopeToolAdapter(toolkit.tools['my_tool'])"
            )

        self._as_tool = as_tool

        # Extract schema from function metadata
        tool_name = getattr(as_tool, "__name__", "unknown_tool")
        tool_description = getattr(as_tool, "__doc__", "") or ""

        self.schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description.strip(),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def aexecute(self, **kwargs: Any) -> ToolResult:
        """Execute the AgentScope tool function asynchronously.

        Args:
            **kwargs: Tool parameters. Should match the function signature.

        Returns:
            ToolResult: The execution result. On success, output contains the
                       function's return value as a string.
        """
        try:
            if asyncio.iscoroutinefunction(self._as_tool):
                result = await self._as_tool(**kwargs)
            else:
                result = self._as_tool(**kwargs)
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            logger.warning(f"AgentScope tool '{self.name}' execution failed: {e}")
            return ToolResult(success=False, error=str(e))


class AgentScopeAgentAdapter(BaseAgent):
    """Adapter for using AgentScope agents as an OpenJudge agent.

    Use this when you have an existing AgentScope agent and want to use it
    for evaluation in OpenJudge. The adapter handles message format conversion
    and result extraction.

    Note:
        AgentScope agents come with their own tools and configuration.
        You don't need to convert tools separately - just pass the whole agent.

    Attributes:
        model: Always None (managed by AgentScope agent internally).
        tools: Always empty dict (managed by AgentScope agent internally).
        max_iterations: Extracted from AgentScope agent's max_iters attribute.
        callback: Always None (use AgentScope's callback system instead).

    Example:
        >>> from agentscope.agent import ReActAgent as ASReActAgent
        >>> from openjudge.agentic.adapters.agentscope import AgentScopeAgentAdapter
        >>>
        >>> # Create AgentScope agent
        >>> as_agent = ASReActAgent(
        ...     name="evaluator",
        ...     model_config_name="gpt-4",
        ...     tools=[...],
        ... )
        >>>
        >>> # Wrap for OpenJudge
        >>> oj_agent = AgentScopeAgentAdapter(as_agent)
        >>> grader = AgenticGrader(agent=oj_agent, template="...")
    """

    def __init__(self, agentscope_agent: Any):
        """Initialize the AgentScopeAgentAdapter.

        Args:
            agentscope_agent: An AgentScope agent instance. Can be any agent
                             class from agentscope.agent (e.g., ReActAgent,
                             DialogAgent, etc.).
        """
        self._as_agent = agentscope_agent
        # These are not used but required by BaseAgent interface
        self.model = None
        self.tools: Dict[str, BaseTool] = {}
        self.max_iterations = getattr(agentscope_agent, "max_iters", 10)
        self.callback = None

    async def arun(
        self,
        messages: List[Dict[str, Any]],
    ) -> AgentResult:
        """Run the AgentScope agent asynchronously.

        Extracts the user message from OpenJudge format, creates an AgentScope
        message, runs the agent, and extracts the result.

        Args:
            messages: Initial messages in OpenAI format. The last user message
                     will be extracted and sent to the AgentScope agent.

        Returns:
            AgentResult containing the agent's response and execution metadata.

        Raises:
            ImportError: If agentscope is not installed.
        """
        try:
            from agentscope.message import Msg
        except ImportError as exc:
            raise ImportError(
                "AgentScope is required for this adapter. " + "Install it with: pip install agentscope"
            ) from exc

        # Extract the last user message as input
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_input = msg.get("content", "")
                break

        # Create AgentScope message
        input_msg = Msg(name="user", content=user_input, role="user")

        try:
            # Call the agent - try reply method first, then __call__
            if hasattr(self._as_agent, "reply"):
                result_msg = await self._as_agent.reply(input_msg)
            else:
                result_msg = await self._as_agent(input_msg)

            # Extract content from result
            output = self._extract_output(result_msg)

            # Try to extract iteration count from AgentScope agent
            iterations = getattr(self._as_agent, "_current_iter", 0)

            return AgentResult(
                content=output,
                messages=messages + [{"role": "assistant", "content": output}],
                iterations=max(iterations, 1),  # At least 1 iteration
                metadata={"source": "agentscope"},
            )

        except Exception as e:
            logger.error(f"AgentScope agent failed: {e}")
            return AgentResult(
                content=f"Error: {str(e)}",
                messages=messages,
                metadata={"error": str(e)},
            )

    def _extract_output(self, result_msg: Any) -> str:
        """Extract the output string from AgentScope message.

        Args:
            result_msg: The result message from AgentScope agent.

        Returns:
            The extracted output string.
        """
        if hasattr(result_msg, "content"):
            output = result_msg.content
            # Handle list content (e.g., multimodal responses)
            if isinstance(output, list):
                output = " ".join(str(block.get("text", block)) for block in output if isinstance(block, dict))
        else:
            output = str(result_msg)

        return output
