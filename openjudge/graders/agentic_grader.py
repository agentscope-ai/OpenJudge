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
      ReActAgent is the built-in implementation

    - Adapter Layer (openjudge.agentic.adapters):
      FunctionToolAdapter for wrapping Python functions as tools.
      External framework adapters (LangChain, AgentScope) are provided as
      examples in tutorials/integrations/ to avoid circular dependencies.

Design Principle:
    AgenticGrader follows "unified interface" design - it only accepts a
    pre-built agent parameter. Whether using built-in ReActAgent or external
    framework adapters, the agent must be constructed externally first.

Classes:
    AgenticGrader: Main class for tool-augmented evaluation.
"""

import json
import os
import re
import textwrap
import time
from typing import Any, Dict, Optional, Union

from openjudge.agentic import BaseAgent, BaseTool, ReActAgent
from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderRank, GraderScore
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

    Design Principle (Unified Interface):
        AgenticGrader only accepts a pre-built `agent` parameter. Whether using
        the built-in ReActAgent or external framework adapters, the agent must
        be constructed externally first. This design:
        - Keeps the interface simple and consistent
        - Separates concerns: Grader handles evaluation, Agent handles reasoning
        - Avoids parameter conflicts (model/tools vs agent)

    Attributes:
        agent (BaseAgent): The agent responsible for reasoning and tool calling.
        template (PromptTemplate): Template for generating evaluation prompts.
        language (LanguageEnum): Language for prompts.

    Example (Built-in ReActAgent - Recommended):
        >>> from openjudge.agentic import ReActAgent
        >>> # Step 1: Build the agent first
        >>> agent = ReActAgent(
        ...     model={"model": "gpt-4", "api_key": "..."},
        ...     tools=[WebSearchTool()],
        ...     max_iterations=10,
        ... )
        >>> # Step 2: Create grader with the agent
        >>> grader = AgenticGrader(
        ...     agent=agent,
        ...     template="Evaluate the response: {response}",
        ... )
        >>> result = await grader.aevaluate(query="...", response="...")

    Example (With External Agent - LangChain):
        >>> # Adapter code: see tutorials/integrations/langchain_adapter.py
        >>> from langchain.agents import create_react_agent
        >>> lc_agent = create_react_agent(llm, tools)
        >>> # Wrap with adapter (implement BaseAgent.arun)
        >>> class LangChainAgentAdapter(BaseAgent):
        ...     def __init__(self, lc_agent):
        ...         self._lc_agent = lc_agent
        ...     async def arun(self, messages):
        ...         result = await self._lc_agent.ainvoke({"messages": messages})
        ...         return AgentResult(content=result.get("output", ""))
        >>> agent = LangChainAgentAdapter(lc_agent)
        >>> grader = AgenticGrader(agent=agent, template="...")
    """

    def __init__(
        self,
        agent: BaseAgent,
        template: Optional[Union[str, dict, list, PromptTemplate]] = None,
        name: str = "agentic_grader",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "Tool-augmented agentic grader",
        language: Optional[Union[LanguageEnum, str]] = None,
        **kwargs: Any,
    ):
        """Initialize AgenticGrader.

        Args:
            agent: Pre-configured agent (required). Can be built-in ReActAgent or
                   external framework adapter (LangChain, AgentScope, etc.).
                   The agent must be constructed externally first.
            template: Template for generating prompts (required). Can be a str, list,
                     dict or PromptTemplate object. Defines how query/response are
                     formatted and passed to the agent.
            name: Grader name.
            mode: POINTWISE or LISTWISE.
            description: Grader description.
            language: Language for prompts. Can be LanguageEnum, string, or None.
                     If None, defaults to environment variable LANGUAGE or "en".
            **kwargs: Additional keyword arguments passed to template rendering.

        Raises:
            ValueError: If agent is not provided.
            ValueError: If template is not provided.

        Example:
            >>> from openjudge.agentic import ReActAgent
            >>> agent = ReActAgent(
            ...     model={"model": "gpt-4", "api_key": "..."},
            ...     tools=[WebSearchTool()],
            ... )
            >>> grader = AgenticGrader(agent=agent, template="Evaluate: {response}")
        """
        super().__init__(name=name, mode=mode, description=description, **kwargs)

        # Validate inputs
        if agent is None:
            raise ValueError(
                "Agent is required for AgenticGrader. "
                "Please construct an agent first (e.g., ReActAgent) and pass it in. "
                "Example: agent = ReActAgent(model=..., tools=[...])"
            )

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

        # Use the provided agent directly
        self.agent = agent

    async def _aevaluate(
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
            >>> from openjudge.agentic import ReActAgent
            >>> agent = ReActAgent(
            ...     model={"model": "gpt-4", "api_key": "..."},
            ...     tools=[WebSearchTool()],
            ... )
            >>> grader = AgenticGrader(agent=agent, template="Evaluate: {response}")
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
            "agent_type": type(self.agent).__name__,
            **self.kwargs,
        }

        # Include agent's max_iterations if available
        if hasattr(self.agent, "max_iterations"):
            d["max_iterations"] = self.agent.max_iterations

        return d

    @classmethod
    def from_config(cls, config: dict) -> "AgenticGrader":
        """Create an AgenticGrader from a configuration dictionary.

        This is a convenience method for creating AgenticGrader from serialized
        config (e.g., from YAML/JSON files). It internally builds a ReActAgent
        from the model/tools config.

        Note:
            This method is provided for convenience when loading from config files.
            For programmatic usage, prefer constructing the agent explicitly:

            >>> agent = ReActAgent(model=..., tools=[...])
            >>> grader = AgenticGrader(agent=agent, template=...)

        Args:
            config: A dictionary containing the AgenticGrader configuration.
                Required keys:
                - template: The evaluation template
                - model: Model configuration dict (for building ReActAgent)
                Optional keys:
                - tools: List of tool instances
                - max_iterations: Max iterations for ReActAgent (default: 10)
                - name, mode, description, language: Grader settings

        Returns:
            A new AgenticGrader instance.

        Example:
            >>> config = {
            ...     "model": {"model": "gpt-4", "api_key": "..."},
            ...     "tools": [my_search_tool],
            ...     "template": "Evaluate: {response}",
            ...     "max_iterations": 10,
            ... }
            >>> grader = AgenticGrader.from_config(config)
        """
        config = config.copy()

        # Extract grader-level config
        name = config.pop("name", "agentic_grader")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "Tool-augmented agentic grader")
        template = config.pop("template", None)
        language = config.pop("language", None)

        # Extract agent-level config
        model_config = config.pop("model", None)
        tools = config.pop("tools", None)
        max_iterations = config.pop("max_iterations", 10)
        callback = config.pop("callback", None)

        # Build the agent from config
        if model_config is None:
            raise ValueError(
                "Model configuration is required in config for from_config(). "
                "Please provide 'model' key with model configuration dict."
            )

        agent = ReActAgent(
            model=model_config,
            tools=tools,
            max_iterations=max_iterations,
            callback=callback,
        )

        return cls(
            agent=agent,
            template=template,
            name=name,
            mode=mode,
            description=description,
            language=language,
            **config,
        )

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return metadata about the AgenticGrader."""
        return {"aevaluate": AgenticGrader.aevaluate.__doc__, "prompt": {}}
