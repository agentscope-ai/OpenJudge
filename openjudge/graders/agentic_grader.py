# -*- coding: utf-8 -*-
"""
Agentic grader implementation for tool-augmented evaluation.

This module provides the AgenticGrader class, which uses autonomous tool calling
to evaluate model responses. The grader supports multi-turn interactions where
the LLM decides which tools to call and when to produce final judgment.

Architecture:
    AgenticGrader extends BaseGrader with agentic capabilities:

    - Tool Layer (openjudge.agentic.tools):
      BaseTool defines what capabilities the agent has (search, code execution, etc.)

    - Agent Layer (openjudge.agentic.agents):
      BaseAgent defines how the agent thinks (ReAct, CoT, etc.)
      ReActAgent is the built-in implementation (default)

    - Adapter Layer (openjudge.agentic.adapters):
      ToolAdapter and AgentAdapter for integrating external frameworks

Classes:
    AgenticGrader: Main class for tool-augmented evaluation.
"""

import json
import os
import re
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Union

from openjudge.agentic import BaseAgent, BaseTool, ReActAgent
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderRank, GraderScore
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

__all__ = [
    "AgenticGrader",
]


class AgenticGrader(BaseGrader):
    """Agentic grader using tool-augmented LLM evaluation.

    This grader extends BaseGrader with agentic capabilities:
    - Tool Layer: Defines what capabilities the agent has (search, code execution, etc.)
    - Agent Layer: Defines how the agent thinks (ReAct, CoT, Multi-Agent, etc.)

    The default configuration uses OpenJudge's built-in ReActAgent, which
    has zero external dependencies. You can swap in external frameworks
    (LangChain, AgentScope, etc.) using the adapter classes.

    Attributes:
        agent (BaseAgent): The agent responsible for reasoning and tool calling.
        template (PromptTemplate): Template for generating evaluation prompts.
        model (BaseChatModel): The LLM model (managed by agent for built-in agent).
        language (LanguageEnum): Language for prompts.
        max_iterations (int): Maximum number of tool-calling iterations.

    Example (Default - Built-in ReActAgent):
        >>> grader = AgenticGrader(
        ...     model=model,
        ...     tools=[WebSearchTool(), CodeExecutionTool()],
        ...     template="Evaluate the response: {response}",
        ... )
        >>> result = await grader.aevaluate(query="...", response="...")

    Example (With LangChain Tools):
        >>> from langchain_tavily import TavilySearch
        >>> from openjudge.agentic.adapters.langchain import LangChainToolAdapter
        >>> lc_tool = TavilySearch()
        >>> oj_tool = LangChainToolAdapter(lc_tool)
        >>> grader = AgenticGrader(model=model, tools=[oj_tool], template="...")

    Example (With LangChain Agent):
        >>> from langchain.agents import create_agent
        >>> from openjudge.agentic.adapters.langchain import LangChainAgentAdapter
        >>> lc_agent = create_agent(llm, tools)
        >>> oj_agent = LangChainAgentAdapter(lc_agent)
        >>> grader = AgenticGrader(agent=oj_agent, template="...")

    Example (With AgentScope Agent):
        >>> from agentscope.agent import ReActAgent as ASReActAgent
        >>> from openjudge.agentic.adapters.agentscope import AgentScopeAgentAdapter
        >>> as_agent = ASReActAgent(...)
        >>> oj_agent = AgentScopeAgentAdapter(as_agent)
        >>> grader = AgenticGrader(agent=oj_agent, template="...")
    """

    def __init__(
        self,
        model: Optional[Union[BaseChatModel, dict]] = None,
        tools: Optional[List[BaseTool]] = None,
        agent: Optional[BaseAgent] = None,
        template: Optional[Union[str, dict, list, PromptTemplate]] = None,
        name: str = "agentic_grader",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "Tool-augmented agentic grader",
        language: Optional[Union[LanguageEnum, str]] = None,
        max_iterations: int = 10,
        callback: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Initialize AgenticGrader.

        You can initialize in two ways:
        1. Provide model + tools + template: Uses built-in ReActAgent (recommended)
        2. Provide agent + template: Uses the provided agent directly

        Args:
            model: LLM for reasoning and tool calling. Can be either a BaseChatModel
                   instance or a dictionary configuration. Required if agent is None.
            tools: List of available tools. Only used with built-in agent.
            agent: Pre-configured agent (e.g., from LangChain or AgentScope).
                   If provided, model and tools are ignored.
            template: Template for generating prompts (required). Can be a str, list,
                     dict or PromptTemplate object. Defines how query/response are
                     formatted and passed to the agent.
            name: Grader name.
            mode: POINTWISE or LISTWISE.
            description: Grader description.
            language: Language for prompts. Can be LanguageEnum, string, or None.
                     If None, defaults to environment variable LANGUAGE or "en".
            max_iterations: Max tool-calling iterations (only for built-in agent).
            callback: Callback function for processing model response metadata.
            **kwargs: Additional keyword arguments passed to template rendering.

        Raises:
            ValueError: If neither model nor agent is provided.
            ValueError: If template is not provided.
        """
        super().__init__(name=name, mode=mode, description=description, **kwargs)

        # Validate inputs
        if agent is None and model is None:
            raise ValueError("Either 'model' or 'agent' must be provided")

        if template is None:
            raise ValueError(
                "Template is required for AgenticGrader. "
                "Please provide a template that defines the evaluation criteria "
                "and output format."
            )

        # Handle language parameter
        if not language:
            language = os.environ.get("LANGUAGE", "en")

        if isinstance(language, str):
            self.language = (
                LanguageEnum(language) if language in [item.value for item in LanguageEnum] else LanguageEnum.EN
            )
        else:
            self.language = language

        # Handle template parameter
        if isinstance(template, str):
            self.template = PromptTemplate(
                messages={
                    LanguageEnum.EN: [
                        ChatMessage(
                            role="system",
                            content="You are a professional evaluation assistant. "
                            "Please evaluate according to the user's requirements.",
                        ),
                        ChatMessage(
                            role="user",
                            content=textwrap.dedent(template),
                        ),
                    ],
                    LanguageEnum.ZH: [
                        ChatMessage(
                            role="system",
                            content="你是个专业的评估助手，请你根据用户要求进行评估。",
                        ),
                        ChatMessage(
                            role="user",
                            content=textwrap.dedent(template),
                        ),
                    ],
                },
            )
        elif isinstance(template, PromptTemplate):
            self.template = template
        elif isinstance(template, list):
            self.template = PromptTemplate.from_prompt(template)
        elif isinstance(template, dict):
            self.template = PromptTemplate(**template)
        else:
            raise ValueError("Template must be a str, list, dict or PromptTemplate object")

        # Initialize model
        if isinstance(model, dict):
            self._model_config = model
            self.model = OpenAIChatModel(**model)
        elif model is not None:
            self._model_config = None
            self.model = model
        else:
            self._model_config = None
            self.model = None

        # Store callback
        self.callback = callback
        self.max_iterations = max_iterations

        # Create or use the agent
        if agent is not None:
            self.agent = agent
        else:
            self.agent = ReActAgent(
                model=self.model,
                tools=tools,
                max_iterations=max_iterations,
                callback=callback,
            )

    async def aevaluate(
        self,
        query: str = "",
        response: str = "",
        **kwargs: Any,
    ) -> Union[GraderScore, GraderRank]:
        """Evaluate using tool-augmented LLM.

        The agent (ReActAgent or external agent) autonomously decides which
        tools to call and when to produce the final judgment.

        Args:
            query: The original query/task.
            response: The response to evaluate.
            **kwargs: Additional context passed to the template.

        Returns:
            GraderScore: In POINTWISE mode, contains score, reason, and metadata.
            GraderRank: In LISTWISE mode, contains rank list, reason, and metadata.

        Raises:
            ValueError: If required fields cannot be extracted from agent output.

        Example:
            >>> grader = AgenticGrader(
            ...     model=OpenAIChatModel(model="gpt-4"),
            ...     tools=[WebSearchTool()],
            ...     template="Evaluate: {response}",
            ... )
            >>> result = await grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     response="The capital of France is Paris."
            ... )
            >>> print(result.score, result.reason)
        """
        start_time = time.time()

        # Build initial messages from template
        params = {**self.kwargs}
        params.update(kwargs)
        params["query"] = query
        params["response"] = response
        messages = self.template.format(language=self.language, **params)
        messages = [msg.to_dict() if hasattr(msg, "to_dict") else msg for msg in messages]

        # Run agent
        agent_result = await self.agent.arun(messages)

        # Parse result
        parsed = self._parse_agent_output(agent_result.content)

        # Build result based on mode
        if self.mode == GraderMode.LISTWISE:
            rank = parsed.pop("rank")
            reason = parsed.pop("reason", agent_result.content)
            result = GraderRank(
                name=self.name,
                rank=rank,
                reason=reason,
                metadata=parsed,
            )
        else:
            score = parsed.pop("score")
            reason = parsed.pop("reason", agent_result.content)
            result = GraderScore(
                name=self.name,
                score=float(score),
                reason=reason,
                metadata=parsed,
            )

        # Add execution metadata
        result.metadata.update(
            {
                "total_time": time.time() - start_time,
                "tool_calls": agent_result.tool_calls_count,
                "iterations": agent_result.iterations,
                "messages": agent_result.messages,
                **agent_result.metadata,
            }
        )

        return result

    def _parse_agent_output(self, content: str) -> Dict[str, Any]:
        """Parse agent output to extract structured data.

        Attempts to extract structured data from agent output using multiple strategies:
        1. Try to parse JSON (supports nested JSON with recursive matching)
        2. Fall back to regex extraction for score/rank patterns

        Args:
            content: Raw agent output string.

        Returns:
            Dictionary containing parsed fields (score/rank, reason, etc.)

        Raises:
            ValueError: If required fields cannot be extracted.
        """
        data: Dict[str, Any] = {}

        # Strategy 1: Try to extract JSON from text (supports nested JSON)
        json_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
        json_matches = re.findall(json_pattern, content, re.DOTALL)

        for match in json_matches:
            try:
                parsed = json.loads(match)
                if "score" in parsed or "rank" in parsed:
                    data = parsed
                    break
            except json.JSONDecodeError:
                continue

        # Strategy 2: Fall back to regex extraction if no valid JSON found
        if not data:
            if self.mode == GraderMode.POINTWISE:
                score_patterns = [
                    r"(?:score|rating|分数)[:\s]*(\d+(?:\.\d+)?)",
                    r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+",
                    r"(?:give|assign|rate)[^0-9]*(\d+(?:\.\d+)?)",
                ]
                for pattern in score_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        data["score"] = float(match.group(1))
                        break
            else:
                rank_patterns = [
                    r"rank[:\s]*\[([^\]]+)\]",
                    r"\[(\d+(?:\s*,\s*\d+)+)\]",
                    r"rank[:\s]*([\d,\s]+)",
                ]
                for pattern in rank_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        try:
                            rank_str = match.group(1)
                            data["rank"] = [int(x.strip()) for x in rank_str.split(",")]
                            break
                        except ValueError:
                            continue

        # Validate required fields
        if self.mode == GraderMode.POINTWISE and "score" not in data:
            raise ValueError(f"Failed to extract 'score' from agent output: {content}")
        if self.mode == GraderMode.LISTWISE and "rank" not in data:
            raise ValueError(f"Failed to extract 'rank' from agent output: {content}")

        # Use content as reason if not provided
        if "reason" not in data:
            data["reason"] = content

        return data

    @property
    def tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools from the agent."""
        return self.agent.tools

    def to_dict(self) -> dict:
        """Convert the AgenticGrader to a dictionary representation.

        Returns:
            A dictionary containing the serialized AgenticGrader information.
        """
        d = {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "template": (self.template.model_dump() if isinstance(self.template, PromptTemplate) else self.template),
            "tools": list(self.tools.keys()),
            "max_iterations": self.max_iterations,
            "agent_type": type(self.agent).__name__,
            **self.kwargs,
        }

        if hasattr(self, "_model_config") and self._model_config:
            d["model"] = self._model_config

        return d

    @classmethod
    def from_config(cls, config: dict) -> "AgenticGrader":
        """Create an AgenticGrader from a configuration dictionary.

        Args:
            config: A dictionary containing the AgenticGrader configuration.

        Returns:
            A new AgenticGrader instance.
        """
        config = config.copy()

        name = config.pop("name", "agentic_grader")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "Tool-augmented agentic grader")
        template = config.pop("template", None)
        model = config.pop("model", {})
        tools = config.pop("tools", None)
        max_iterations = config.pop("max_iterations", 10)

        return cls(
            name=name,
            mode=mode,
            description=description,
            template=template,
            model=model,
            tools=tools,
            max_iterations=max_iterations,
            **config,
        )

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return metadata about the AgenticGrader."""
        return {"aevaluate": AgenticGrader.aevaluate.__doc__, "prompt": {}}
