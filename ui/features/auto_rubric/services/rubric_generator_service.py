# -*- coding: utf-8 -*-
"""Rubric generator service for Auto Rubric feature.

This service wraps the openjudge generator modules to provide a clean
interface for the UI layer.

Phase 1: Simple Rubric generation (zero-shot from task description)
Phase 2: Iterative Rubric generation (data-driven from labeled data)
"""

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from openjudge.generator.iterative_rubric.generator import (
    IterativeListwiseRubricsGeneratorConfig,
    IterativePointwiseRubricsGeneratorConfig,
    IterativeRubricsGenerator,
)
from openjudge.generator.simple_rubric import (
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
)
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


@dataclass
class SimpleRubricConfig:
    """Configuration for Simple Rubric generation.

    Attributes:
        grader_name: Unique name for the generated grader.
        task_description: Description of the task for evaluation.
        scenario: Optional usage scenario for context.
        sample_queries: Optional list of sample queries.
        grader_mode: POINTWISE or LISTWISE evaluation mode.
        language: Language for prompts (EN or ZH).
        min_score: Minimum score for pointwise mode.
        max_score: Maximum score for pointwise mode.
        max_retries: Maximum retry attempts for LLM calls.
        api_endpoint: API endpoint URL.
        api_key: API key for authentication.
        model_name: Model name to use.
    """

    grader_name: str
    task_description: str
    scenario: str | None = None
    sample_queries: list[str] | None = None
    grader_mode: GraderMode = GraderMode.POINTWISE
    language: LanguageEnum = LanguageEnum.EN
    min_score: int = 0
    max_score: int = 5
    max_retries: int = 3
    api_endpoint: str = ""
    api_key: str = ""
    model_name: str = ""


@dataclass
class GenerationResult:
    """Result of rubric generation.

    Attributes:
        success: Whether generation was successful.
        grader: The generated LLMGrader instance (if successful).
        rubrics: The generated rubrics text (if successful).
        error: Error message (if failed).
        grader_config: Configuration used for generation.
    """

    success: bool
    grader: LLMGrader | None = None
    rubrics: str | None = None
    error: str | None = None
    grader_config: dict[str, Any] | None = None


class RubricGeneratorService:
    """Service for generating evaluation rubrics.

    This service provides a clean interface for the UI to generate rubrics
    using the openjudge generator modules. Phase 1 supports Simple Rubric mode.

    Example:
        >>> service = RubricGeneratorService()
        >>> config = SimpleRubricConfig(
        ...     grader_name="medical_qa_grader",
        ...     task_description="Medical Q&A system",
        ...     api_endpoint="https://api.example.com/v1",
        ...     api_key="sk-xxx",
        ...     model_name="gpt-4o-mini",
        ... )
        >>> result = await service.generate_simple(config)
        >>> if result.success:
        ...     print(result.rubrics)
    """

    async def generate_simple(self, config: SimpleRubricConfig) -> GenerationResult:
        """Generate rubrics using Simple Rubric mode.

        This method creates an LLMGrader with rubrics generated from the
        task description and optional sample queries.

        Args:
            config: Configuration for Simple Rubric generation.

        Returns:
            GenerationResult with the generated grader and rubrics,
            or error information if generation failed.
        """
        try:
            logger.info(f"Starting Simple Rubric generation for '{config.grader_name}'")

            # Create the model instance
            model = OpenAIChatModel(
                model=config.model_name,
                base_url=config.api_endpoint,
                api_key=config.api_key,
            )

            # Create generator config
            generator_config = SimpleRubricsGeneratorConfig(
                grader_name=config.grader_name,
                model=model,
                grader_mode=config.grader_mode,
                task_description=config.task_description,
                scenario=config.scenario,
                language=config.language,
                min_score=config.min_score,
                max_score=config.max_score,
                max_retries=config.max_retries,
            )

            # Create generator and generate
            generator = SimpleRubricsGenerator(generator_config)
            grader = await generator.generate(
                dataset=[],
                sample_queries=config.sample_queries,
            )

            # Extract rubrics from grader
            rubrics = grader.kwargs.get("rubrics", "")

            # Build grader config for export
            grader_config = {
                "grader_name": config.grader_name,
                "task_description": config.task_description,
                "scenario": config.scenario,
                "grader_mode": config.grader_mode.value,
                "language": config.language.value,
                "min_score": config.min_score,
                "max_score": config.max_score,
                "rubrics": rubrics,
                "model_name": config.model_name,
                "api_endpoint": config.api_endpoint,
            }

            logger.info(f"Successfully generated rubrics for '{config.grader_name}'")

            return GenerationResult(
                success=True,
                grader=grader,
                rubrics=rubrics,
                grader_config=grader_config,
            )

        except Exception as e:
            logger.error(f"Failed to generate rubrics: {e}")
            return GenerationResult(
                success=False,
                error=str(e),
            )

    async def generate_iterative(
        self,
        config: "IterativeRubricConfig",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> GenerationResult:
        """Generate rubrics using Iterative Rubric mode.

        This method learns evaluation criteria from labeled preference data
        using an iterative refinement process.

        Args:
            config: Configuration for Iterative Rubric generation.
            progress_callback: Optional callback for progress updates.
                              Receives (stage_name, progress_percentage).

        Returns:
            GenerationResult with the generated grader and rubrics,
            or error information if generation failed.
        """
        try:
            logger.info(f"Starting Iterative Rubric generation for '{config.grader_name}'")

            if progress_callback:
                progress_callback("init", 0.0)

            # Create the model instance
            model = OpenAIChatModel(
                model=config.model_name,
                base_url=config.api_endpoint,
                api_key=config.api_key,
            )

            if progress_callback:
                progress_callback("init", 0.1)

            # Create generator config based on grader_mode
            if config.grader_mode == GraderMode.LISTWISE:
                generator_config = IterativeListwiseRubricsGeneratorConfig(
                    grader_name=config.grader_name,
                    model=model,
                    language=config.language,
                    enable_categorization=config.enable_categorization,
                    categories_number=config.categories_number,
                    query_specific_generate_number=config.query_specific_generate_number,
                    task_description=config.task_description,
                    max_retries=config.max_retries,
                )
            else:
                # Default to Pointwise mode
                generator_config = IterativePointwiseRubricsGeneratorConfig(
                    grader_name=config.grader_name,
                    model=model,
                    language=config.language,
                    min_score=config.min_score,
                    max_score=config.max_score,
                    enable_categorization=config.enable_categorization,
                    categories_number=config.categories_number,
                    query_specific_generate_number=config.query_specific_generate_number,
                    task_description=config.task_description,
                    max_retries=config.max_retries,
                )

            if progress_callback:
                progress_callback("generating", 0.2)

            # Create generator and generate
            generator = IterativeRubricsGenerator(generator_config)
            grader = await generator.generate(dataset=config.dataset)

            if progress_callback:
                progress_callback("processing", 0.9)

            # Extract rubrics from grader
            rubrics = grader.kwargs.get("rubrics", "")

            # Build grader config for export
            grader_config: dict[str, Any] = {
                "grader_name": config.grader_name,
                "task_description": config.task_description,
                "grader_mode": config.grader_mode.value,
                "language": config.language.value,
                "rubrics": rubrics,
                "model_name": config.model_name,
                "api_endpoint": config.api_endpoint,
                "data_count": len(config.dataset),
                "enable_categorization": config.enable_categorization,
                "categories_number": config.categories_number,
            }

            # Add score range only for pointwise mode
            if config.grader_mode == GraderMode.POINTWISE:
                grader_config["min_score"] = config.min_score
                grader_config["max_score"] = config.max_score

            if progress_callback:
                progress_callback("complete", 1.0)

            logger.info(f"Successfully generated iterative rubrics for '{config.grader_name}'")

            return GenerationResult(
                success=True,
                grader=grader,
                rubrics=rubrics,
                grader_config=grader_config,
            )

        except Exception as e:
            logger.error(f"Failed to generate iterative rubrics: {e}")
            return GenerationResult(
                success=False,
                error=str(e),
            )

    async def test_grader(
        self,
        grader: LLMGrader,
        query: str,
        response: str,
    ) -> dict[str, Any]:
        """Test a generated grader with sample input (Pointwise mode).

        Args:
            grader: The LLMGrader instance to test.
            query: Test query/question.
            response: Response to evaluate.

        Returns:
            Dictionary containing score and reasoning.
        """
        try:
            result = await grader.aevaluate(
                query=query,
                response=response,
            )
            return {
                "success": True,
                "score": result.score,
                "reason": result.reason,
                "name": result.name,
            }
        except Exception as e:
            logger.error(f"Grader test failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def test_grader_listwise(
        self,
        grader: LLMGrader,
        query: str,
        responses: list[str],
    ) -> dict[str, Any]:
        """Test a generated grader with multiple responses (Listwise mode).

        Args:
            grader: The LLMGrader instance to test.
            query: Test query/question.
            responses: List of responses to rank.

        Returns:
            Dictionary containing rank and reasoning.
        """
        try:
            # Build kwargs for listwise evaluation
            # The grader template expects 'responses' as formatted string and 'num_responses'
            # Format: "Response 1:\n{content}\n\nResponse 2:\n{content}\n\n..."
            responses_text = "\n\n".join([f"Response {i + 1}:\n{resp}" for i, resp in enumerate(responses)])

            result = await grader.aevaluate(
                query=query,
                responses=responses_text,
                num_responses=len(responses),
            )
            return {
                "success": True,
                "rank": result.rank,
                "reason": result.reason,
                "name": result.name,
            }
        except Exception as e:
            logger.error(f"Grader listwise test failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }


@dataclass
class IterativeRubricConfig:
    """Configuration for Iterative Rubric generation.

    Attributes:
        grader_name: Unique name for the generated grader.
        dataset: List of labeled training data dictionaries.
        grader_mode: POINTWISE or LISTWISE evaluation mode.
        task_description: Optional task description for context.
        language: Language for prompts (EN or ZH).
        min_score: Minimum score for pointwise mode.
        max_score: Maximum score for pointwise mode.
        enable_categorization: Whether to group similar rubrics.
        categories_number: Target number of categories.
        query_specific_generate_number: Rubrics per training sample.
        max_retries: Maximum retry attempts for LLM calls.
        api_endpoint: API endpoint URL.
        api_key: API key for authentication.
        model_name: Model name to use.
    """

    grader_name: str
    dataset: list[dict[str, Any]]
    grader_mode: GraderMode = GraderMode.POINTWISE
    task_description: str | None = None
    language: LanguageEnum = LanguageEnum.EN
    min_score: int = 0
    max_score: int = 5
    enable_categorization: bool = True
    categories_number: int = 5
    query_specific_generate_number: int = 2
    max_retries: int = 3
    api_endpoint: str = ""
    api_key: str = ""
    model_name: str = ""
