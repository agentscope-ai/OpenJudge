# -*- coding: utf-8 -*-
"""
Tool Parameter Check Grader

Evaluates whether the generated tool call extracts completely correct parameters from the user query and the matching
tool of tool definitions
"""

import json
import textwrap
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
TOOL_PARAMETER_CHECK_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing tool calls. Your task is to evaluate whether the generated tool call extracts completely correct parameters from the user query and the matching tool of tool definitions. This includes checking if parameters are accurate and complete.

<Rubrics>
1. All required parameters are present and grounded in the matching tool of tool definitions
2. All required parameter values are extracted from the query
3. All parameter data types and formats are grounded in the matching tool of tool definitions
4. Optional parameter values are used appropriately when present in the query
5. It is a completion extraction if a optional parameter value is not present in query and use null/none or equivalent value as a placeholder
6. Avoid checking tool selection accuracy
</Rubrics>

<Steps>
1. Verify parameter completeness: Check if all required parameter values present in the query are extracted
2. Verify parameter accuracy: Ensure parameter values match the query exactly if present in the query
3. Check data types: Ensure the data types and formats of all parameter values are grounded in the matching tool of tool definitions
</Steps>

<Scale>
- **Score 1.0**: All parameters are correct and complete (excellent parameter extraction)
- **Score 0.0**: Parameters have issues (poor parameter extraction)
</Scale>

<Query>
{query}
</Query>

<Tool Definitions>
{tool_definitions}
</Tool Definitions>

<Tool Calls>
{tool_calls}
</Tool Calls>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of parameter quality and correctness>",
    "score": <0.0 or 1.0>
}}
</Output Schema>
JSON:
"""
).strip()

# Chinese Prompt
TOOL_PARAMETER_CHECK_PROMPT_ZH = textwrap.dedent(
    """你是一名分析工具调用的专家。你的任务是评估生成的工具调用是否从用户查询中提取了完全正确的参数。这包括检查参数是否准确、完整且格式正确。

<评分标准>
1. 所有必需的参数都存在并从查询中提取
2. 参数值与查询中指定的完全匹配
3. 所有参数都基于查询（没有捏造的值）
4. 参数数据类型和格式正确
5. 当查询中指定时，适当使用可选参数
6. 查询中提到的所有参数都被捕获
7. 参数正确映射，没有混淆
</评分标准>

<评估步骤>
1. 验证参数完整性：检查查询中提到的所有参数是否都被提取
2. 验证参数准确性：确保参数值与查询完全匹配
3. 检测幻觉：识别查询中不存在的任何参数
4. 检查数据类型：验证参数使用正确的数据类型和格式
5. 评估整体正确性：确定工具调用是否可以用正确的参数执行
</评估步骤>

<评分量表>
- **分数 1.0**：所有参数都正确且完整（优秀的参数提取）
- **分数 0.0**：参数存在问题（参数提取不佳）
</评分量表>

<查询>
{query}
</查询>

<工具定义>
{tool_definitions}
</工具定义>

<工具调用>
{tool_calls}
</工具调用>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于参数质量和正确性的详细解释>",
    "score": <0.0 或 1.0>
}}
</输出格式>
JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_TOOL_PARAMETER_CHECK_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TOOL_PARAMETER_CHECK_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TOOL_PARAMETER_CHECK_PROMPT_ZH,
            ),
        ],
    },
)


class ToolParameterCheckGrader(LLMGrader):
    """
    Tool Parameter Check Grader

    Evaluates whether the generated tool call extracts completely correct parameters
    from the user query and the matching tool of tool definitions.

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
        >>> grader = ToolParameterCheckGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Search for Python files in the src directory",
        ...     tool_definition="search_files(pattern: str, directory: str)",
        ...     tool_calls='search_files(pattern="*.py", directory="src")'
        ... ))
        >>> print(f"Score: {result.score}")  # 1.0 (correct parameters)
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_TOOL_PARAMETER_CHECK_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ToolParameterCheckGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts (default: DEFAULT_TOOL_PARAMETER_CHECK_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="tool_parameter_check",
            mode=GraderMode.POINTWISE,
            description="Evaluate tool parameter extraction correctness",
            model=model,
            template=template or DEFAULT_TOOL_PARAMETER_CHECK_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        query: Union[str, List[Dict[str, Any]]],
        tool_definitions: Union[Dict[str, Any], List[Dict[str, Any]]],
        tool_calls: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> GraderScore:
        """
        Evaluate tool parameter extraction correctness

        Args:
            query: Query or chat history. Can be a string for simple queries or a list of
                  message dictionaries for conversation history.
            tool_definitions: List of tool definitions available to the agent.
                             Each definition includes name, description, and parameters.
            tool_calls: List of tool calls made by the agent, including tool name and arguments.

        Returns:
            GraderScore: Score with binary value (1.0 = correct, 0.0 = incorrect)

        Example:
            >>> conversation = [
            ...     {"role": "user", "content": "Find JSON files in config folder"}
            ... ]
            >>> tool_defs = [{
            ...     "name": "search_files",
            ...     "parameters": {"pattern": "str", "directory": "str"}
            ... }]
            >>> tool_calls = [{
            ...     "name": "search_files",
            ...     "arguments": {"pattern": "*.json", "directory": "config"}
            ... }]
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

        try:
            result = await super()._aevaluate(
                query=query,
                tool_definitions=json.dumps(tool_definitions, indent=2),
                tool_calls=json.dumps(tool_calls, indent=2),
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating tool parameter check: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "error_type": "tool_parameter_check",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ToolParameterCheckGrader",
    "DEFAULT_TOOL_PARAMETER_CHECK_TEMPLATE",
]
