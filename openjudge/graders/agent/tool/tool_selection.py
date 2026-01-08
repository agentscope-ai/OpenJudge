# -*- coding: utf-8 -*-
"""
Tool Selection Grader

Evaluates the tool selection made by the agent to address the user query.
"""

import json
import textwrap
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
TOOL_SELECTION_PROMPT_EN = textwrap.dedent(
    """
You are an expert in analyzing tool selection decisions. Your task is to evaluate the  of tool selection made by an agent to address the user query.

<Evaluation Dimension: Tool Selection >
Evaluate whether the agent selected the most appropriate tool(s) from the available tools to effectively address the user's query. This includes assessing relevance, completeness, and efficiency of tool selection.
</Evaluation Dimension>

<Rubrics for Evaluation>
1. The selected tool is highly relevant to the user's query and request
2. The tool selection directly addresses the core intent of the query
3. All necessary tools are selected (no critical tools missing)
4. No unnecessary or irrelevant tools are selected
5. The tool selection is efficient (avoids redundant tool calls)
6. The tool is capable of providing the information or performing the action requested
7. The tool selection demonstrates good understanding of tool capabilities and limitations
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Assess relevance: Does the selected tool match the query intent?
2. Check completeness: Are all necessary tools selected?
3. Evaluate efficiency: Are there unnecessary or redundant tool selections?
4. Verify capability: Can the selected tool(s) fulfill the user's request?
5. Consider alternatives: Are there better tool choices available?
</Evaluation Criteria>

<query>
{query}
</query>

<available_tools>
{available_tools}
</available_tools>

<selected_tools>
{selected_tools}
</selected_tools>

# Scoring Instructions
Use a scale from 1 to 5:
- 5: Optimal tool selection - Accurately identifies the task intent and selects the most direct, efficient, and semantically relevant tool from the available options.
- 4: Reasonable tool selection - The selected tool can complete the task, but it is not the optimal choice, or a tool with overlapping but slightly redundant functionality is chosen.
- 3: Acceptable tool selection - The selected tool is related to the task but not a direct match.
- 2: Poor tool selection - The selected tool is clearly mismatched with the task and cannot directly support achieving the goal.
- 1: Completely incorrect tool selection - No tool is selected and the answer is given without using any tools, or a completely irrelevant or non-existent tool is chosen.

Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5>,
    "reason": "<detailed explanation of tool selection quality, including strengths and weaknesses>"
}}

JSON:
"""
).strip()

# Chinese Prompt
TOOL_SELECTION_PROMPT_ZH = textwrap.dedent(
    """
你是一名分析工具选择决策的专家。你的任务是评估智能体为解决用户查询而做出的工具选择的质量。

<评估维度：工具选择质量>
评估智能体是否从可用工具中选择了最合适的工具来有效地解决用户的查询。这包括评估工具选择的相关性、完整性和效率。
</评估维度>

<评估准则>
1. 选择的工具与用户的查询和请求高度相关
2. 工具选择直接针对查询的核心意图
3. 选择了所有必要的工具（没有遗漏关键工具）
4. 没有选择不必要或不相关的工具
5. 工具选择是高效的（避免冗余的工具调用）
6. 工具能够提供请求的信息或执行请求的操作
7. 工具选择表现出对工具能力和局限性的良好理解
</评估准则>

<评估标准>
进行分析时：
1. 评估相关性：选择的工具是否与查询意图匹配？
2. 检查完整性：是否选择了所有必要的工具？
3. 评估效率：是否有不必要或冗余的工具选择？
4. 验证能力：选择的工具是否能够满足用户的请求？
5. 考虑替代方案：是否有更好的工具选择可用？
</评估标准>

<查询>
{query}
</查询>

<可用工具>
{available_tools}
</可用工具>

<选择工具>
{selected_tools}
</选择工具>

# 评分指令
使用 1 到 5 的评分标准：
- 5：最优的工具选择 - 精准识别任务意图，从可用工具中选出最直接、高效、语义匹配度最高的工具。
- 4：合理的工具选择 - 所选工具能完成任务，但非最优，或选择了功能覆盖但略显冗余的工具。
- 3：可接受的工具选择 - 所选工具与任务相关但不直接匹配。
- 2：较差的工具选择 - 所选工具与任务明显不匹配，无法直接支持目标达成。
- 1：完全错误的工具选择 - 选择工具为空，或选择完全无关或不存在的工具。

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1 到 5 之间的整数>,
    "reason": "<关于工具选择质量的详细解释，包括优点和缺点>"
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_TOOL_SELECTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TOOL_SELECTION_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TOOL_SELECTION_PROMPT_ZH,
            ),
        ],
    },
)


class ToolSelectionGrader(LLMGrader):
    """
    Tool Selection Grader

    Evaluates the tool selection made by the agent to address the user query.

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.schema.template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = ToolSelectionGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Find all Python files modified in the last week",
        ...     tool_definitions=[
        ...         {"name": "search_files", "description": "Search for files"},
        ...         {"name": "git_log", "description": "Get git history"}
        ...     ],
        ...     tool_calls=[
        ...         {"name": "search_files", "arguments": {"pattern": "*.py"}},
        ...         {"name": "git_log", "arguments": {"days": 7}}
        ...     ]
        ... ))
        >>> print(f"Score: {result.score}")  # Score from 1 to 5
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_TOOL_SELECTION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="tool_selection",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool selection ",
            model=model,
            template=template or DEFAULT_TOOL_SELECTION_TEMPLATE,
            language=language,
        )

    async def aevaluate(
        self,
        query: Union[str, List[Dict[str, Any]]],
        tool_definitions: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> GraderScore:
        """
        Evaluate tool selection

        Args:
            query: Query or chat history. Can be a string for simple queries or a list of
                  message dictionaries for conversation history.
            tool_definitions: List of all available tool definitions.
                             Each definition includes name, description, and parameters.
            tool_calls: List of tool calls actually made by the agent.
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score from 1 to 5 indicating tool selection quality

        Example:
            >>> conversation = [
            ...     {"role": "user", "content": "Search for configuration files"}
            ... ]
            >>> tool_defs = [
            ...     {"name": "search_files", "description": "Search files"},
            ...     {"name": "read_file", "description": "Read a file"}
            ... ]
            >>> tool_calls = [{"name": "search_files", "arguments": {...}}]
            >>> result = await grader.aevaluate(
            ...     query=conversation,
            ...     tool_definitions=tool_defs,
            ...     tool_calls=tool_calls
            ... )
        """

        # Ensure tool_calls and tool_definitions are lists
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list):
            tool_definitions = [tool_definitions] if tool_definitions else []

        # Format query as string for the prompt
        if isinstance(query, list):
            query = "\n".join(
                [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in query],
            )
        else:
            query = str(query)

        # Format available tools
        available_tools = json.dumps(tool_definitions, indent=2, ensure_ascii=False)

        # Format selected tools
        selected_tools = json.dumps(tool_calls, indent=2, ensure_ascii=False)

        try:
            result = await super().aevaluate(
                query=query,
                available_tools=available_tools,
                selected_tools=selected_tools,
            )
            score = result.score
            reason = result.reason

        except Exception as e:
            logger.error(f"Error evaluating tool selection: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ToolSelectionGrader",
    "DEFAULT_TOOL_SELECTION_TEMPLATE",
]
