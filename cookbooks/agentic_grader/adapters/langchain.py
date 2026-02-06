# -*- coding: utf-8 -*-
"""
LangChain adapters for tools and agents.

This module provides adapters for integrating the LangChain ecosystem with OpenJudge.
It allows you to use LangChain's rich tool library with OpenJudge's built-in agents,
or use existing LangChain agents directly in OpenJudge evaluations.

Classes:
    LangChainToolAdapter: Adapt LangChain tools to OpenJudge BaseTool.
    LangChainAgentAdapter: Adapt LangChain agents to OpenJudge BaseAgent.

Dependencies:
    This module requires langchain-core to be installed:
    pip install langchain-core

Example:
    >>> # Use LangChain tools with OpenJudge's ReActAgent
    >>> from langchain_community.tools import DuckDuckGoSearchRun
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainToolAdapter
    >>> from openjudge.agentic import ReActAgent
    >>>
    >>> lc_tool = DuckDuckGoSearchRun()
    >>> oj_tool = LangChainToolAdapter(lc_tool)
    >>> agent = ReActAgent(model=my_model, tools=[oj_tool])
    >>>
    >>> # Use LangChain agent directly in OpenJudge
    >>> from langchain.agents import create_agent
    >>> from cookbooks.agentic_grader.adapters.langchain import LangChainAgentAdapter
    >>>
    >>> lc_agent = create_agent(llm, tools)
    >>> oj_agent = LangChainAgentAdapter(lc_agent)
    >>> grader = AgenticGrader(agent=oj_agent, template="...")
"""

from typing import Any, Dict, List

from loguru import logger

from openjudge.agentic.agents import AgentResult, BaseAgent
from openjudge.agentic.tools import BaseTool, ToolResult

__all__ = [
    "LangChainToolAdapter",
    "LangChainAgentAdapter",
]


class LangChainToolAdapter(BaseTool):
    """Adapter for wrapping a LangChain tool as an OpenJudge BaseTool.

    Use this when you want to use LangChain's rich tool ecosystem with
    OpenJudge's built-in ReActAgent. This adapter handles the conversion
    between LangChain's tool interface and OpenJudge's BaseTool interface.

    Attributes:
        schema: OpenAI function calling schema extracted from the LangChain tool.

    Example:
        >>> from langchain_community.tools import DuckDuckGoSearchRun
        >>> from openjudge.agentic.adapters.langchain import LangChainToolAdapter
        >>> from openjudge.agentic import ReActAgent
        >>> from openjudge.graders.agentic_grader import AgenticGrader
        >>>
        >>> # Wrap a LangChain tool
        >>> lc_tool = DuckDuckGoSearchRun()
        >>> oj_tool = LangChainToolAdapter(lc_tool)
        >>>
        >>> # Build agent first, then create grader (unified interface)
        >>> agent = ReActAgent(model=model, tools=[oj_tool])
        >>> grader = AgenticGrader(agent=agent, template="...")
    """

    def __init__(self, lc_tool: Any):
        """Initialize the LangChainToolAdapter.

        Args:
            lc_tool: A LangChain tool instance (BaseTool or StructuredTool).
                    Must be an instance of langchain_core.tools.BaseTool.

        Raises:
            ImportError: If langchain-core is not installed.
            ValueError: If the input is not a valid LangChain tool.
        """
        try:
            from langchain_core.tools import BaseTool as LCBaseTool
        except ImportError as exc:
            raise ImportError(
                "LangChain is required for this adapter. " + "Install it with: pip install langchain-core"
            ) from exc

        if not isinstance(lc_tool, LCBaseTool):
            raise ValueError(f"Expected a LangChain BaseTool instance, got {type(lc_tool).__name__}")

        self._lc_tool = lc_tool

        # Extract schema from LangChain tool
        tool_name = lc_tool.name
        tool_description = lc_tool.description or ""

        # Build parameters schema from args_schema if available
        parameters: Dict[str, Any]
        if hasattr(lc_tool, "args_schema") and lc_tool.args_schema:
            try:
                parameters = lc_tool.args_schema.model_json_schema()
                # Remove unnecessary fields from schema
                parameters.pop("title", None)
                parameters.pop("description", None)
            except Exception:
                parameters = {"type": "object", "properties": {}, "required": []}
        else:
            parameters = {"type": "object", "properties": {}, "required": []}

        self.schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": parameters,
            },
        }

    async def aexecute(self, **kwargs: Any) -> ToolResult:
        """Execute the LangChain tool asynchronously.

        This method tries async methods first (ainvoke, arun), then falls back
        to sync methods (invoke, run) if async is not available.

        Args:
            **kwargs: Tool parameters. Should match the tool's args_schema.

        Returns:
            ToolResult: The execution result. On success, output contains the
                       tool's return value as a string.
        """
        try:
            # Try async methods first for better performance
            if hasattr(self._lc_tool, "ainvoke"):
                result = await self._lc_tool.ainvoke(kwargs)
            elif hasattr(self._lc_tool, "arun"):
                # arun expects a single argument for simple tools
                if len(kwargs) == 1:
                    result = await self._lc_tool.arun(list(kwargs.values())[0])
                else:
                    result = await self._lc_tool.arun(kwargs)
            else:
                # Fall back to sync methods
                if hasattr(self._lc_tool, "invoke"):
                    result = self._lc_tool.invoke(kwargs)
                elif hasattr(self._lc_tool, "run"):
                    if len(kwargs) == 1:
                        result = self._lc_tool.run(list(kwargs.values())[0])
                    else:
                        result = self._lc_tool.run(kwargs)
                else:
                    # Last resort: call the tool directly
                    result = self._lc_tool(kwargs)

            return ToolResult(success=True, output=str(result))
        except Exception as e:
            logger.warning(f"LangChain tool '{self.name}' execution failed: {e}")
            return ToolResult(success=False, error=str(e))


class LangChainAgentAdapter(BaseAgent):
    """Adapter for using LangChain agents as an OpenJudge agent.

    Use this when you have an existing LangChain agent and want to use it
    for evaluation in OpenJudge. The adapter handles message format conversion
    and result extraction.

    Note:
        LangChain agents come with their own tools. You don't need to convert
        tools separately - just pass the whole agent.

    Attributes:
        model: Always None (managed by LangChain agent internally).
        tools: Always empty dict (managed by LangChain agent internally).
        max_iterations: Default iteration limit (actual limit is in LangChain agent).
        callback: Always None (use LangChain's callback system instead).

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain.agents import create_agent
        >>> from langchain_tavily import TavilySearch
        >>> from openjudge.agentic.adapters.langchain import LangChainAgentAdapter
        >>>
        >>> # Create LangChain agent with tools
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> tools = [TavilySearch()]
        >>> lc_agent = create_agent(llm, tools)
        >>>
        >>> # Wrap for OpenJudge
        >>> oj_agent = LangChainAgentAdapter(lc_agent)
        >>> grader = AgenticGrader(agent=oj_agent, template="...")
    """

    def __init__(self, langchain_agent: Any):
        """Initialize the LangChainAgentAdapter.

        Args:
            langchain_agent: A LangChain agent instance. Can be created using
                            langchain.agents.create_agent or similar factory
                            functions.
        """
        self._lc_agent = langchain_agent
        # These are not used but required by BaseAgent interface
        self.model = None
        self.tools: Dict[str, BaseTool] = {}
        self.max_iterations = 10
        self.callback = None

    async def arun(
        self,
        messages: List[Dict[str, Any]],
    ) -> AgentResult:
        """Run the LangChain agent asynchronously.

        Converts OpenJudge messages to LangChain format, runs the agent,
        and extracts the result.

        Args:
            messages: Initial messages in OpenAI format. Will be converted
                     to LangChain message format.

        Returns:
            AgentResult containing the agent's response and execution metadata.

        Raises:
            ImportError: If langchain-core is not installed.
        """
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError as exc:
            raise ImportError(
                "LangChain is required for this adapter. " + "Install it with: pip install langchain-core"
            ) from exc

        # Convert OpenJudge messages to LangChain format
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        try:
            # LangChain agents expect {"messages": [...]} as input
            if hasattr(self._lc_agent, "ainvoke"):
                result = await self._lc_agent.ainvoke({"messages": lc_messages})
            else:
                result = self._lc_agent.invoke({"messages": lc_messages})

            # Extract the final message from the result
            output = self._extract_output(result)

            # Count tool calls and iterations from output messages
            tool_calls_count, iterations = self._count_tool_calls(result)

            return AgentResult(
                content=output,
                messages=messages + [{"role": "assistant", "content": output}],
                tool_calls_count=tool_calls_count,
                iterations=iterations,
                metadata={"source": "langchain"},
            )

        except Exception as e:
            logger.error(f"LangChain agent failed: {e}")
            return AgentResult(
                content=f"Error: {str(e)}",
                messages=messages,
                metadata={"error": str(e)},
            )

    def _extract_output(self, result: Any) -> str:
        """Extract the final output from LangChain agent result.

        Args:
            result: The result from LangChain agent invocation.

        Returns:
            The extracted output string.
        """
        output_messages = result.get("messages", []) if isinstance(result, dict) else []
        output = ""

        if output_messages:
            # Find the last AI message with content
            for msg in reversed(output_messages):
                msg_content = getattr(msg, "content", None)
                msg_type = type(msg).__name__

                # Skip ToolMessage, find AIMessage
                if msg_type == "AIMessage" and msg_content:
                    output = msg_content
                    break
                if msg_type == "HumanMessage":
                    # Stop if we hit a human message
                    break

            # If still empty, try the last message
            if not output and output_messages:
                last_msg = output_messages[-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    output = last_msg.content
                else:
                    output = str(last_msg)

        if not output:
            logger.warning(
                f"LangChain agent returned empty output. "
                f"Result keys: {result.keys() if isinstance(result, dict) else type(result)}"
            )
            output = f"Error: LangChain agent returned empty response. " f"Raw result: {str(result)[:500]}"

        return output

    def _count_tool_calls(self, result: Any) -> tuple[int, int]:
        """Count tool calls and iterations from LangChain result.

        Args:
            result: The result from LangChain agent invocation.

        Returns:
            Tuple of (tool_calls_count, iterations).
        """
        output_messages = result.get("messages", []) if isinstance(result, dict) else []
        tool_calls_count = 0
        iterations = 0

        for msg in output_messages:
            msg_type = type(msg).__name__
            if msg_type == "AIMessage":
                iterations += 1
                # Count tool calls in this AI message
                tool_calls = getattr(msg, "tool_calls", None) or []
                tool_calls_count += len(tool_calls)

        return tool_calls_count, iterations
