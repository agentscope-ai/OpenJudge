# -*- coding: utf-8 -*-
"""Task-based rubric generator for automatic evaluation criteria generation.

This module provides functionality to automatically generate evaluation rubrics
based on task descriptions, enabling zero-shot evaluation pipelines.

The generator uses an LLM to analyze the task description and sample queries
to produce relevant evaluation criteria without requiring labeled training data.

Classes:
    TaskBasedRubricGenerator: Generator for evaluation rubrics.
"""

from typing import List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# =============================================================================
# Constants
# =============================================================================

DEFAULT_RUBRICS: List[str] = [
    "Accuracy: Whether the response is factually correct",
    "Relevance: Whether the response addresses the query",
    "Completeness: Whether the response is comprehensive",
]

# =============================================================================
# Prompt Templates
# =============================================================================

RUBRIC_GENERATION_PROMPT_EN = """# Task
Generate evaluation rubrics for pairwise comparison of model responses.

## Task Description
{task_description}

## Scenario
{scenario}

## Sample Queries (for context)
{sample_queries}

## Requirements
- Generate 3-5 clear evaluation criteria for comparing two responses
- Each criterion should be objective and measurable
- Criteria should be relevant to the task and scenario
- Focus on aspects that distinguish good responses from poor ones

## Output Format
Return a JSON object with:
- rubrics: list of evaluation criteria strings
- reason: brief explanation of why these criteria are important

Example:
{{
    "rubrics": [
        "Accuracy: Whether the response contains correct and factual information",
        "Completeness: Whether the response fully addresses the query",
        "Clarity: Whether the response is well-organized and easy to understand"
    ],
    "reason": "These criteria capture the key aspects for evaluating..."
}}
"""

RUBRIC_GENERATION_PROMPT_ZH = """# 任务
为模型回答的成对比较生成评估标准。

## 任务描述
{task_description}

## 使用场景
{scenario}

## 示例查询（用于上下文理解）
{sample_queries}

## 要求
- 生成3-5个清晰的评估标准用于比较两个回答
- 每个标准应该客观且可测量
- 标准应与任务和场景相关
- 聚焦于能够区分好回答和差回答的方面

## 输出格式
返回一个JSON对象，包含：
- rubrics: 评估标准字符串列表
- reason: 简要解释为什么这些标准是重要的

示例：
{{
    "rubrics": [
        "准确性：回答是否包含正确和真实的信息",
        "完整性：回答是否完整地解决了问题",
        "清晰度：回答是否组织良好、易于理解"
    ],
    "reason": "这些标准捕捉了评估的关键方面..."
}}
"""

RUBRIC_GENERATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are an expert at designing evaluation criteria for AI systems.",
            ),
            ChatMessage(role="user", content=RUBRIC_GENERATION_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content="你是一位设计AI系统评估标准的专家。",
            ),
            ChatMessage(role="user", content=RUBRIC_GENERATION_PROMPT_ZH),
        ],
    },
)


# =============================================================================
# Output Schema
# =============================================================================


class RubricGenerationOutput(BaseModel):
    """Output schema for rubric generation."""

    rubrics: List[str] = Field(..., description="List of evaluation rubrics")
    reason: str = Field(default="", description="Reasoning for these rubrics")


# =============================================================================
# TaskBasedRubricGenerator
# =============================================================================


class TaskBasedRubricGenerator:
    """Generate evaluation rubrics based on task description.

    This generator creates evaluation rubrics that can be used for pairwise
    comparison or other evaluation scenarios. It uses an LLM to generate
    task-specific criteria based on the provided task description.

    Example:
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.generator.simple_rubric import TaskBasedRubricGenerator
        >>>
        >>> model = OpenAIChatModel(model="gpt-4o-mini")
        >>> generator = TaskBasedRubricGenerator(
        ...     model=model,
        ...     task_description="Medical question answering system",
        ...     scenario="Healthcare professionals seeking quick answers"
        ... )
        >>> rubrics = await generator.generate(sample_queries=["What are the symptoms of flu?"])
    """

    def __init__(
        self,
        model: BaseChatModel,
        task_description: str,
        scenario: Optional[str] = None,
        language: LanguageEnum = LanguageEnum.EN,
        default_rubrics: Optional[List[str]] = None,
        max_retries: int = 3,
    ):
        """Initialize TaskBasedRubricGenerator.

        Args:
            model: Language model for generating rubrics.
            task_description: Description of the task for evaluation.
            scenario: Optional usage scenario for context.
            language: Language for prompts (ZH or EN). Defaults to EN.
            default_rubrics: Fallback rubrics if generation fails.
            max_retries: Maximum number of retry attempts for LLM calls.
        """
        self.model = model
        self.task_description = task_description
        self.scenario = scenario
        self.language = language
        self.default_rubrics = default_rubrics or DEFAULT_RUBRICS.copy()
        self.max_retries = max_retries

    async def generate(
        self,
        sample_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate evaluation rubrics.

        Args:
            sample_queries: Optional sample queries for context.
                           These help the LLM understand what kind of
                           queries will be evaluated.

        Returns:
            List of rubric strings
        """

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def _generate() -> List[str]:
            queries_text = "None provided"
            if sample_queries:
                queries_text = "\n".join(f"- {q}" for q in sample_queries[:5])

            messages = RUBRIC_GENERATION_TEMPLATE.format(
                task_description=self.task_description,
                scenario=self.scenario or "General usage",
                sample_queries=queries_text,
                language=self.language,
            )

            response = await self.model.achat(
                messages=list(messages),
                structured_model=RubricGenerationOutput,
            )

            if not response.parsed or "rubrics" not in response.parsed:
                raise ValueError("Failed to parse rubric generation response")

            return response.parsed["rubrics"]

        try:
            rubrics = await _generate()
            logger.info(f"Generated {len(rubrics)} evaluation rubrics")
            for i, rubric in enumerate(rubrics, 1):
                logger.debug(f"  {i}. {rubric}")
            return rubrics
        except Exception as e:
            logger.error(f"Rubric generation failed: {e}")
            logger.warning("Using default rubrics as fallback")
            return self.default_rubrics
