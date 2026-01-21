# -*- coding: utf-8 -*-
"""Simple rubrics generator implementation.

This module implements a task-description-based approach to generating
evaluation rubrics. It creates LLMGrader instances with rubrics generated
from task descriptions and sample queries.

This is a simpler alternative to the iterative_rubric module, which learns
rubrics from preference data through an iterative refinement process.

Usage:
    >>> from openjudge.generator.simple_rubric import SimpleRubricsGenerator, SimpleRubricsGeneratorConfig
    >>> from openjudge.models.openai_chat_model import OpenAIChatModel
    >>>
    >>> config = SimpleRubricsGeneratorConfig(
    ...     grader_name="Medical QA Grader",
    ...     model=OpenAIChatModel(model="gpt-4o-mini"),
    ...     task_description="Medical question answering system",
    ...     scenario="Healthcare professionals seeking quick answers"
    ... )
    >>> generator = SimpleRubricsGenerator(config)
    >>> grader = await generator.generate(dataset=[], sample_queries=["What are the symptoms of flu?"])
"""

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from openjudge.generator.iterative_rubric.query_rubric_generator import (
    LISTWISE_EVALUATION_TEMPLATE,
    POINTWISE_EVALUATION_TEMPLATE,
)
from openjudge.generator.llm_grader_generator import (
    LLMGraderGenerator,
    LLMGraderGeneratorConfig,
)
from openjudge.generator.simple_rubric.rubric_generator import TaskBasedRubricGenerator
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


@dataclass
class SimpleRubricsGeneratorConfig(LLMGraderGeneratorConfig):
    """Configuration for simple rubrics generator.

    This configuration extends LLMGraderGeneratorConfig with parameters
    specific to task-description-based rubric generation.

    Attributes:
        task_description: Description of the task for evaluation.
        scenario: Optional usage scenario for context.
        language: Language for prompts (ZH or EN). Defaults to EN.
        default_rubrics: Fallback rubrics if generation fails.
        max_retries: Maximum number of retry attempts for LLM calls.
        min_score: Minimum score for pointwise evaluation.
        max_score: Maximum score for pointwise evaluation.

    Inherited from LLMGraderGeneratorConfig:
        grader_name: Human-readable name for the generated grader.
        model: Language model to use for generation.
        grader_mode: Mode for the generated grader (POINTWISE or LISTWISE).
        custom_evaluation_prompt: Custom template for evaluation.
    """

    task_description: str = ""
    scenario: Optional[str] = None
    language: LanguageEnum = LanguageEnum.EN
    default_rubrics: List[str] = field(default_factory=list)
    max_retries: int = 3
    min_score: int = 0
    max_score: int = 1

    def __post_init__(self):
        """Process model configuration if provided as dict."""
        if isinstance(self.model, dict):
            self.model = OpenAIChatModel(**self.model)


class SimpleRubricsGenerator(LLMGraderGenerator):
    """Generator for creating LLM-based graders with task-description-based rubrics.

    This generator implements a simple approach to rubric generation:
    1. Takes a task description and optional sample queries
    2. Uses an LLM to generate relevant evaluation criteria
    3. Creates an LLMGrader configured with these rubrics

    Example:
        >>> config = SimpleRubricsGeneratorConfig(
        ...     grader_name="Medical QA Grader",
        ...     model=OpenAIChatModel(model="gpt-4o-mini"),
        ...     task_description="Medical question answering system",
        ...     scenario="Healthcare professionals seeking quick answers"
        ... )
        >>> generator = SimpleRubricsGenerator(config)
        >>> grader = await generator.generate(
        ...     dataset=[],
        ...     sample_queries=["What are the symptoms of flu?"]
        ... )
    """

    def __init__(self, config: SimpleRubricsGeneratorConfig) -> None:
        """Initialize the simple rubrics generator.

        Args:
            config: Configuration for rubric generation.
        """
        super().__init__(config)
        self.config: SimpleRubricsGeneratorConfig = config

        self._rubric_generator = TaskBasedRubricGenerator(
            model=config.model,
            task_description=config.task_description,
            scenario=config.scenario,
            language=config.language,
            default_rubrics=config.default_rubrics,
            max_retries=config.max_retries,
        )

    async def generate(
        self,
        dataset: List[dict],
        sample_queries: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMGrader:
        """Generate an LLMGrader with rubrics from task description.

        Args:
            dataset: List of data dictionaries (used to extract sample queries
                    if sample_queries is not provided).
            sample_queries: Optional list of sample queries for context.
            **kwargs: Additional arguments (currently unused).

        Returns:
            LLMGrader: Configured grader instance with generated rubrics.
        """
        if sample_queries is None and dataset:
            sample_queries = [d.get("query", "") for d in dataset[:5] if d.get("query")]

        rubrics = await self._generate_rubrics(sample_queries)

        grader_kwargs = {
            "name": self.config.grader_name,
            "model": self.config.model,
            "mode": self.config.grader_mode,
            "rubrics": rubrics,
            "language": self.config.language,
        }

        if self.config.grader_mode == GraderMode.POINTWISE:
            grader_kwargs["min_score"] = self.config.min_score
            grader_kwargs["max_score"] = self.config.max_score

        if self.config.custom_evaluation_prompt is not None:
            grader_kwargs["template"] = self.config.custom_evaluation_prompt
        else:
            if self.config.grader_mode == GraderMode.POINTWISE:
                grader_kwargs["template"] = POINTWISE_EVALUATION_TEMPLATE
            else:
                grader_kwargs["template"] = LISTWISE_EVALUATION_TEMPLATE

        # Add task_description_section for template formatting
        if self.config.task_description:
            if self.config.language == LanguageEnum.ZH:
                grader_kwargs["task_description_section"] = f"\n## 任务场景描述\n{self.config.task_description}\n"
            else:
                grader_kwargs["task_description_section"] = f"\n## Task Description\n{self.config.task_description}\n"
        else:
            grader_kwargs["task_description_section"] = ""

        return LLMGrader(**grader_kwargs)

    async def _generate_rubrics(
        self,
        dataset: Optional[List[str]] = None,  # pylint: disable=arguments-renamed
    ) -> str:
        """Generate rubrics from task description.

        Args:
            dataset: Optional list of sample queries for context.

        Returns:
            str: Formatted string containing evaluation rubrics.
        """
        rubrics_list = await self._rubric_generator.generate(sample_queries=dataset)

        formatted_rubrics = "\n\n".join([f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics_list)])

        logger.info(f"Generated {len(rubrics_list)} rubrics from task description")

        return formatted_rubrics
