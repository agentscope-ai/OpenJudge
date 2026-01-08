# -*- coding: utf-8 -*-
"""
LLM-Based Rubric Categorizer

This module provides semantic categorization and aggregation of evaluation rubrics
using LLM-based classification. It merges similar rubrics into structured Theme-Tips
format to reduce redundancy and improve usability.

Features:
- LLM-powered semantic classification into Theme-Tips structure
- Configurable number of output categories
- Multi-language support (Chinese/English)
- Structured output with Pydantic models
- Integrated with AutoRubrics pipeline for automatic aggregation
- Compatible with both Single Mode and Batch Mode

All prompts are defined inline in this module.
"""

import textwrap
from typing import List

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# ========== Categorization Prompts ==========

RUBRIC_CATEGORIZATION_PROMPT_ZH = """
请将以下评估建议聚合成{num_categories}个或更少的结构化评估标准。

## 输入评估建议
{rubrics}

## 任务要求
- 评估标准必须完全自包含，非专业读者无需查阅任何外部信息
- 每个标准应评估独立维度，彼此不矛盾
- 确保整体判断在所有示例中保持一致

## 标准格式
每个标准包含两部分：
- 主题：简洁明确的陈述，捕捉标准的核心焦点
- 技巧：多个要点，扩展或补充标准的具体指导

## 输出格式
请严格按照以下JSON格式输出：
{{
    "categories": [
        {{
            "theme": "第一个主题的陈述",
            "tips": [
                "具体指导要点",
                "具体指导要点"
            ]
        }},
        {{
            "theme": "第二个主题的陈述",
            "tips": [
                "具体指导要点",
                "具体指导要点"
            ]
        }}
    ],
    "reason": "聚合这些评估标准的原因和依据"
}}

请生成聚合后的评估标准
"""

RUBRIC_CATEGORIZATION_PROMPT_EN = """
Please aggregate the following evaluation suggestions into {num_categories} or fewer structured evaluation rubrics.

## Input Evaluation Suggestions
{rubrics}

## Task Requirements
- Rubrics must be fully self-contained so that non-expert readers need not consult any external information
- Each rubric should assess an independent dimension and be non-contradictory with others
- Ensure overall judgment remains aligned and consistent for all examples

## Rubric Format
Each rubric consists of two parts:
- Theme: A concise and clear statement that captures the core focus of the rubric
- Tips: Multiple bullet points that expand on or supplement the rubric with specific guidance

## Output Format
Please output strictly in the following JSON format:
{{
    "categories": [
        {{
            "theme": "First theme statement",
            "tips": [
                "Specific guidance point",
                "Specific guidance point"
            ]
        }},
        {{
            "theme": "Second theme statement",
            "tips": [
                "Specific guidance point",
                "Specific guidance point"
            ]
        }}
    ],
    "reason": "Reason and basis for aggregating these evaluation criteria"
}}

Please generate the aggregated evaluation criteria
"""

# ========== Build PromptTemplate ==========

RUBRIC_CATEGORIZATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content="你是一个专业的评估标准聚合专家，擅长将多个零散的评估标准整合成结构化的主题-技巧格式。",
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(RUBRIC_CATEGORIZATION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are a professional evaluation criteria aggregation expert, "
                "skilled at integrating multiple scattered evaluation criteria into "
                "structured Theme-Tips format.",
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(RUBRIC_CATEGORIZATION_PROMPT_EN),
            ),
        ],
    },
)


class RubricCategory(BaseModel):  # rubric item/summarizer
    """Represents a categorized rubric with theme and tips.

    This model defines the structure for a categorized group of rubrics
    with a central theme and specific tips.

    Attributes:
        theme (str): Theme statement for the category.
                    A general description that encompasses all grouped rubrics.
        tips (List[str]): List of tips for the category.
                     Specific evaluation points that fall under this theme.
    """

    theme: str = Field(description="Theme statement for the category")
    tips: List[str] = Field(description="List of tips for the category")


class RubricCategorizationOutput(BaseModel):
    """Output model for rubric categorization.

    Structured output from the rubric categorization process.

    Attributes:
        categories (List[RubricCategory]): List of categorized rubrics.
                                         Each category contains a theme and related tips.
        reason (str): Reasoning for the categorization.
                    Explanation of how and why the rubrics were grouped.
    """

    categories: List[RubricCategory] = Field(
        description="List of categorized rubrics",
    )
    reason: str = Field(description="Reasoning for the categorization")


class LLMRubricCategorizer:
    """LLM-based rubric categorizer for semantic aggregation.

    This class uses LLMs to semantically categorize evaluation rubrics into
    thematic groups, reducing redundancy and improving organization.

    The categorizer works by analyzing the semantic meaning of rubrics and
    grouping similar ones under common themes, each with specific tips.
    """

    def __init__(
        self,
        num_categories: int = 5,
        model: BaseChatModel | None = None,
        language: str = "zh",
    ):
        """
        Initialize the rubric categorizer.

        Args:
            num_categories: Number of categories to generate.
                           Controls how many thematic groups to create.
            model: Chat model to use for categorization.
                   If None, a default model may be used.
            language: Language for prompts ('zh' or 'en').
                     Determines which language to use for the categorization prompts.
        """
        self.num_categories = num_categories
        self.language = LanguageEnum(language)
        self.model = model

        # Initialize categorization template
        self.categorization_template = RUBRIC_CATEGORIZATION_TEMPLATE

    async def categorize_rubrics(
        self,
        rubrics: List[str],
    ) -> tuple[List[str], dict]:
        """
        Main method: perform semantic classification of rubrics.

        Args:
            rubrics: List of rubrics to categorize.
                    Each string should be a distinct evaluation criterion.

        Returns:
            tuple: (categorized_rubrics, aggregation_info)
                  categorized_rubrics: List of formatted rubrics organized by category
                  aggregation_info: Dictionary with metadata about the categorization process

        Example:
            >>> categorizer = LLMRubricCategorizer(num_categories=3)
            >>> rubrics = [
            ...     "The response should be factually accurate",
            ...     "The response should avoid hallucinations",
            ...     "The response should be concise and clear"
            ... ]
            >>> categorized_rubrics, info = await categorizer.categorize_rubrics(rubrics)
        """

        if not rubrics:
            logger.error("Input rubrics list is empty")
            return [], {
                "categorization_successful": False,
                "error": "Empty input",
            }

        try:
            # Format rubrics text
            rubrics_text = "\n".join(
                [f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics)],
            )

            # Call LLM using Chat with structured output
            messages = self.categorization_template.format(
                rubrics=rubrics_text,
                num_categories=self.num_categories,
                language=self.language,
            )
            response_obj = await self.model.achat(
                messages,
                structured_model=RubricCategorizationOutput,
            )

            # Get structured data from parsed
            if not response_obj.parsed or "categories" not in response_obj.parsed:
                raise ValueError("No categories in structured response")

            categories = response_obj.parsed["categories"]

            # Generate directly usable string list
            ready_to_use_list = []
            for category in categories:
                theme = category.get("theme", "")
                tips = category.get("tips", [])

                # Assemble into single string: Theme + Tips
                theme_str = f"Theme: {theme}"
                tips_str = "\n".join(
                    [f"- Tip{i}: {tip}" for i, tip in enumerate(tips, start=1)],
                )

                # Combine into complete evaluation rubric string
                complete_rubric = f"{theme_str}\n{tips_str}"
                ready_to_use_list.append(complete_rubric)

            logger.info(
                f"Generated {len(ready_to_use_list)} categorized rubrics",
            )

            # Return both rubrics and aggregation info as expected
            aggregation_info = {
                "num_categories": len(ready_to_use_list),
                "original_rubrics_count": len(rubrics),
                "categorization_successful": True,
            }
            return ready_to_use_list, aggregation_info

        except Exception as e:
            logger.error(f"Rubric categorization failed: {e}")
            return [], {"categorization_successful": False, "error": str(e)}
