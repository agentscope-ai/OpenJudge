# -*- coding: utf-8 -*-
"""Iterative rubric generator implementation.

This module implements a training-free framework for automatically extracting
evaluation rubrics from preference data. The framework uses a two-stage approach:

1. Query-specific generation: Generates tailored rubrics for each training example
   using an iterative Propose-Evaluate-Revise loop with validation.

2. Aggregation and categorization: Consolidates query-specific rubrics into a unified,
   non-redundant set using information-theoretic selection (MCR²) and optional
   semantic categorization.

The generated rubrics are used to configure LLM-based graders that can evaluate
new data without requiring model training, achieving high data efficiency through
the generalization ability of evaluation criteria.

This implementation is based on the paper:
    Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling
    https://arxiv.org/abs/2510.17314
"""

import asyncio
from dataclasses import dataclass
from typing import List, Union

from loguru import logger

from openjudge.generator.iterative_rubric.categorizer import LLMRubricCategorizer
from openjudge.generator.iterative_rubric.mcr_selector import SuperFastAdaptiveMCR2
from openjudge.generator.iterative_rubric.query_rubric_generator import (
    QuerySpecificRubricGenerator,
)
from openjudge.generator.llm_grader_generator import (
    LLMGraderGenerator,
    LLMGraderGeneratorConfig,
)
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


@dataclass
class IterativeRubricsGeneratorConfig(LLMGraderGeneratorConfig):
    """Base configuration parameters for rubric generators.

    This class encapsulates all configuration parameters for rubric generation,
    including sampling strategies, generation parameters, batch processing settings,
    and categorization options. Parameters are organized by functionality for clarity.

    Attributes:
        language: Language for prompts (ZH or EN).
                 Determines which language to use for prompt templates.
                 Defaults to EN.
        enable_categorization: Whether to enable LLM-based categorization to merge similar rubrics.
                              If False, keeps all rubrics as individual items.
                              If True, uses LLM-based categorization to merge similar rubrics into categories.
                              Defaults to False.
        query_specific_generate_number: Number of rubrics to generate per sample.
                                       Controls how many evaluation criteria are created per data item.
                                       Defaults to 1.
        categories_number: Target number of categories when categorization is enabled.
                          Controls how many thematic groups to create during categorization.
                          Defaults to 5.
        max_retries: Maximum LLM API retry attempts on failure.
                    Used for handling transient failures in LLM interactions.
                    Defaults to 5.
        max_epochs: Maximum iterative refinement epochs per sample.
                   Limits how many times rubrics can be revised during validation.
                   Defaults to 3.
        batch_size: Number of eval cases to process per batch iteration.
                   Controls how many samples are processed in each batch.
                   Defaults to 10.
        mcr_batch_size: Number of rubrics selected by MCR² per iteration.
                       Controls the selection size in smart sampling mode.
                       Defaults to 10.
        min_increment_threshold: Minimum information gain to continue iteration.
                                Threshold for convergence detection.
                                Defaults to 0.002.
        patience: Consecutive low-increment iterations before early stopping.
                 Stops iteration when information gain is consistently low.
                 Defaults to 2.
        max_iterations: Maximum batch iterations allowed.
                       Limits the total number of batch processing iterations.
                       Defaults to 50.
        max_total_rubrics: Maximum total rubrics to maintain in pool.
                          Caps the total number of rubrics collected.
                          Defaults to 200.
    """

    # Core configuration
    language: LanguageEnum = LanguageEnum.EN
    enable_categorization: bool = False
    query_specific_generate_number: int = 1
    categories_number: int = 5

    # Generation parameters
    max_retries: int = 5
    max_epochs: int = 3

    # Batch processing parameters
    batch_size: int = 10

    # MCR² parameters
    mcr_batch_size: int = 10
    min_increment_threshold: float = 0.002
    patience: int = 2
    max_iterations: int = 50
    max_total_rubrics: int = 200

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = OpenAIChatModel(**self.model)
        else:
            self.model = self.model


@dataclass
class IterativePointwiseRubricsGeneratorConfig(IterativeRubricsGeneratorConfig):
    """Configuration parameters for Pointwise rubrics generator.

    This configuration class is for pointwise (scoring) rubric generation.
    It automatically sets grader_mode to POINTWISE. All other parameters
    are inherited from IterativeRubricsGeneratorConfig.

    Attributes:
        min_score: Minimum score value for pointwise mode. Defaults to 0.
        max_score: Maximum score value for pointwise mode. Defaults to 1.
    """

    # Pointwise-specific parameters
    min_score: int = 0
    max_score: int = 1

    def __post_init__(self):
        """Automatically set grader_mode to POINTWISE."""
        # Set grader_mode automatically
        object.__setattr__(self, "grader_mode", GraderMode.POINTWISE)


@dataclass
class IterativeListwiseRubricsGeneratorConfig(IterativeRubricsGeneratorConfig):
    """Configuration parameters for Listwise rubrics generator.

    This configuration class is for listwise (ranking) rubric generation.
    It automatically sets grader_mode to LISTWISE. All other parameters
    are inherited from IterativeRubricsGeneratorConfig.
    """

    def __post_init__(self):
        """Automatically set grader_mode to LISTWISE."""
        # Set grader_mode automatically
        object.__setattr__(self, "grader_mode", GraderMode.LISTWISE)


class IterativeRubricsGenerator(LLMGraderGenerator):
    """Generator for creating LLM-based graders with automatically generated rubrics.

    This generator implements a training-free framework that extracts evaluation
    rubrics from preference data. It uses a two-stage approach:

    1. Query-specific generation: For each training example, generates tailored
       rubrics using an iterative Propose-Evaluate-Revise loop that ensures
       rubric quality through validation.

    2. Aggregation and categorization: Consolidates query-specific rubrics into
       a unified, non-redundant set using information-theoretic selection
       (MCR²) and optional semantic categorization.

    The generated rubrics are then used to configure an LLMGrader that can
    evaluate new data without requiring model training. This approach achieves
    high data efficiency by leveraging the generalization ability of evaluation
    criteria across diverse queries.

    Supports both pointwise (scoring) and listwise (ranking) evaluation modes.
    """

    def __init__(
        self,
        config: Union[IterativePointwiseRubricsGeneratorConfig, IterativeListwiseRubricsGeneratorConfig],
    ) -> None:
        """Initialize the rubrics generator with the provided configuration.

        Args:
            config (Union[IterativePointwiseRubricsGeneratorConfig, IterativeListwiseRubricsGeneratorConfig]):
                Configuration for rubric generation. Can be:
                - IterativePointwiseRubricsGeneratorConfig for pointwise evaluation
                - IterativeListwiseRubricsGeneratorConfig for listwise evaluation
                - RubricsGeneratorConfig for backward compatibility
                The grader_mode is automatically set based on the config type.

                The configuration includes parameters from the inheritance hierarchy:
                From GraderGeneratorConfig:
                - grader_name (str): Human-readable name for the generated grader.

                From LLMGraderGeneratorConfig:
                - model (BaseChatModel | None): Language model to use for generation.
                - grader_mode (GraderMode): Mode for the generated grader (POINTWISE or LISTWISE).
                - custom_evaluation_prompt (PromptTemplate | None): Custom template for evaluation.

                From IterativeRubricsGeneratorConfig:
                - language (LanguageEnum): Language for prompts (ZH or EN). Defaults to EN.
                - enable_categorization (bool): Whether to enable LLM-based categorization. Defaults to False.
                - query_specific_generate_number (int): Number of rubrics to generate per sample. Defaults to 1.
                - categories_number (int): Target number of categories when categorization is enabled. Defaults to 5.
                - max_retries (int): Maximum LLM API retry attempts on failure. Defaults to 5.
                - max_epochs (int): Maximum iterative refinement epochs per sample. Defaults to 3.
                - batch_size (int): Number of eval cases to process per batch iteration. Defaults to 10.
                - mcr_batch_size (int): Number of rubrics selected by MCR² per iteration. Defaults to 10.
                - min_increment_threshold (float): Minimum information gain to continue iteration. Defaults to 0.002.
                - patience (int): Consecutive low-increment iterations before early stopping. Defaults to 2.
                - max_iterations (int): Maximum batch iterations allowed. Defaults to 50.
                - max_total_rubrics (int): Maximum total rubrics to maintain in pool. Defaults to 200.

                From IterativePointwiseRubricsGeneratorConfig (only for pointwise evaluation):
                - min_score (int): Minimum score value for pointwise mode. Defaults to 0.
                - max_score (int): Maximum score value for pointwise mode. Defaults to 1.
        """
        self.config = config

    async def generate(
        self,
        dataset: List[dict],
        **kwargs,
    ) -> LLMGrader:
        """Generate an LLMGrader with iteratively-generated rubrics from training data.

        This method generates evaluation rubrics from the provided training data
        and creates an LLMGrader instance configured with these rubrics. The training
        data is used only for rubric generation, not for evaluation.

        Args:
            dataset: List of training data dictionaries to generate rubrics from.
                 For pointwise mode, each dict should contain:
                 - "query": str, the input query
                 - "response": str, the response to evaluate
                 - "score": int (optional), the expected score for validation
                 For listwise mode, each dict should contain:
                 - "query": str, the input query
                 - "responses": List[str], multiple responses to rank
                 - "rank": List[int] (optional), expected ranking for validation
            **kwargs: Additional arguments passed to sub-methods.

        Returns:
            LLMGrader: Configured grader instance with generated rubrics.
                     Can be used to evaluate new data via aevaluate() method.
        """

        # Generate rubrics
        rubrics = await self._generate_rubrics(dataset, **kwargs)

        # Prepare grader kwargs
        grader_kwargs = {
            "name": self.config.grader_name,
            "model": self.config.model,
            "mode": self.config.grader_mode,
            "rubrics": rubrics,
            "language": self.config.language,
        }

        # Add min_score and max_score only for pointwise mode
        if self.config.grader_mode == GraderMode.POINTWISE:
            grader_kwargs["min_score"] = self.config.min_score
            grader_kwargs["max_score"] = self.config.max_score

        # Add custom template if provided
        if hasattr(self.config, "custom_evaluation_prompt") and self.config.custom_evaluation_prompt:
            grader_kwargs["template"] = self.config.custom_evaluation_prompt

        return LLMGrader(**grader_kwargs)

    async def _generate_rubrics(
        self,
        dataset: List[dict],
        **kwargs,
    ) -> str:
        """Generate evaluation rubrics from training data through a two-step process.

        This method implements the core rubric generation pipeline:
        1. Query-specific generation: Creates tailored rubrics for each training example
           using an iterative Propose-Evaluate-Revise loop
        2. Aggregation and categorization: Consolidates query-specific rubrics into
           a unified set of evaluation criteria

        The process supports two sampling modes:
        - ALL_SAMPLES: Processes all samples concurrently, generating rubrics for each
        - SMART_SAMPLING: Uses MCR²-based selection to iteratively build an optimal
          rubric subset with maximum information diversity

        Args:
            dataset: List of training data dictionaries for rubric generation.
                 Format depends on grader_mode:
                 - Pointwise: {"query": str, "response": str, "label_score": int}
                 - Listwise: {"query": str, "responses": List[str], "rank": List[int]}
            **kwargs: Additional arguments passed to sub-methods.

        Returns:
            str: Formatted string containing consolidated evaluation rubrics.
                Ready to be used as rubrics parameter in LLMGrader.

        Example:
            >>> dataset = [
            ...     {"query": "Explain photosynthesis", "response": "Photosynthesis is...", "label_score": 1},
            ...     {"query": "Calculate 2+2", "response": "2+2=4", "label_score": 1}
            ... ]
            >>> generator = RubricsGenerator()
            >>> rubrics = await generator._generate_rubrics(data)
        """
        # Generate rubrics for each query
        query_rubrics = await self._generate_query_rubrics(dataset, **kwargs)

        # Categorize rubrics across queries
        categorized_rubrics = await self._categorize_query_rubrics(
            query_rubrics,
            **kwargs,
        )

        return categorized_rubrics

    # pylint: disable=unused-argument, too-many-statements
    async def _generate_query_rubrics(
        self,
        dataset: List[dict],
        **kwargs,
    ) -> List[str]:
        """Generate query-specific rubrics for each training example.

        This method creates tailored evaluation criteria for each training sample
        using an iterative Propose-Evaluate-Revise loop. Each iteration proposes
        rubrics, evaluates them against the training example, and revises if needed
        until validation succeeds or max_epochs is reached.

        The sampling strategy is automatically selected based on data size:
        - <= 100 samples: Uses ALL_SAMPLES mode, processing all samples concurrently
        - > 100 samples: Uses SMART_SAMPLING mode with MCR² selection for optimal subsets

        Args:
            dataset: List of training data dictionaries. Each dict should contain:
                 - "query": str, the input query
                 - "response" or "responses": str or List[str], response(s) to evaluate
                 - "label_score" or "label_rank": int or List[int] (optional), for validation
            **kwargs: Additional arguments passed to QuerySpecificRubricGenerator.

        Returns:
            List[str]: List of generated rubrics. In ALL_SAMPLES mode, contains rubrics
                      from all samples. In SMART_SAMPLING mode, contains the selected
                      optimal subset based on MCR² information-theoretic criteria.
        """
        # Initialize query-specific rubric generator
        generator_kwargs = {
            "model": self.config.model,
            "grader_mode": self.config.grader_mode,
            "generate_number": self.config.query_specific_generate_number,
            "max_retries": self.config.max_retries,
            "max_epochs": self.config.max_epochs,
            "language": self.config.language,
        }

        # Add min_score and max_score only for pointwise mode
        if self.config.grader_mode == GraderMode.POINTWISE:
            generator_kwargs["min_score"] = self.config.min_score
            generator_kwargs["max_score"] = self.config.max_score

        query_generator = QuerySpecificRubricGenerator(**generator_kwargs)

        # Automatically select sampling mode based on data size
        # <= 100 samples: all_samples, > 100 samples: smart_sampling
        sampling_mode = "all_samples" if len(dataset) <= 100 else "smart_sampling"
        logger.info(f"Auto-selected sampling mode: {sampling_mode} (data size: {len(dataset)})")

        # ALL_SAMPLES mode: Process all samples concurrently, generating rubrics independently
        if sampling_mode == "all_samples":
            logger.info(f"Using ALL_SAMPLES mode: processing all {len(dataset)} samples concurrently")

            # Create coroutines for all samples
            all_coroutines = []
            for data_item in dataset:
                all_coroutines.append(
                    query_generator.generate_iterative(data_item),
                )

            # Execute all coroutines concurrently
            all_results = await asyncio.gather(*all_coroutines)

            # Extract rubrics from results
            all_rubrics = []
            successful_count = 0
            failed_count = 0

            for idx, result in enumerate(all_results):
                rubric_valid = result.get("rubric_valid", False)
                rubrics = result.get("rubrics", [])

                if rubrics and rubric_valid:
                    all_rubrics.extend(rubrics)
                    successful_count += 1
                    logger.debug(
                        f"Data item {idx + 1}: {len(rubrics)} rubrics, "
                        f"valid={rubric_valid}, epoch={result.get('rubric_epoch', 'N/A')}",
                    )
                elif rubrics and not rubric_valid:
                    logger.warning(
                        f"Data item {idx + 1}: {len(rubrics)} rubrics generated but "
                        f"failed validation (valid={rubric_valid}), skipping",
                    )
                    failed_count += 1
                else:
                    logger.warning(f"Data item {idx + 1}: no rubrics generated")
                    failed_count += 1

            logger.info(
                f"ALL_SAMPLES completed: {len(all_rubrics)} total rubrics from "
                f"{successful_count} successful samples ({failed_count} failed)",
            )
            return all_rubrics

        # SMART_SAMPLING mode: Iteratively build optimal rubric subset using MCR² selection
        else:
            logger.info("Using SMART_SAMPLING mode with MCR² selection")

            # Initialize MCR² selector
            mcr_selector = SuperFastAdaptiveMCR2(batch_size=self.config.mcr_batch_size)

            selected_rubrics = []
            coding_rates = [0.0]
            low_increment_count = 0
            current_index = 0
            iteration = 0

            while iteration < self.config.max_iterations:
                iteration += 1

                # Get next batch
                start_idx = current_index
                end_idx = min(start_idx + self.config.batch_size, len(dataset))
                if start_idx >= len(dataset):
                    # Reset for new cycle
                    current_index = 0
                    start_idx = 0
                    end_idx = min(self.config.batch_size, len(dataset))

                batch_data = dataset[start_idx:end_idx]
                current_index = end_idx

                logger.info(
                    f"Iteration {iteration}: Processing batch {start_idx}-{end_idx - 1} "
                    f"({len(batch_data)} samples)",
                )

                # Generate rubrics for batch concurrently
                batch_coroutines = [query_generator.generate_iterative(data_item) for data_item in batch_data]
                batch_results = await asyncio.gather(*batch_coroutines)

                batch_rubrics = []
                for result in batch_results:
                    rubric_valid = result.get("rubric_valid", False)
                    rubrics = result.get("rubrics", [])

                    if rubrics and rubric_valid:
                        batch_rubrics.extend(rubrics)
                    elif rubrics and not rubric_valid:
                        logger.debug(
                            f"Skipping invalid rubrics: {len(rubrics)} rubrics "
                            f"failed validation (valid={rubric_valid})",
                        )

                if not batch_rubrics:
                    logger.warning(f"No rubrics generated in iteration {iteration}")
                    continue

                # Combine existing and new rubrics, then apply MCR² information-theoretic selection
                combined_rubrics = selected_rubrics + batch_rubrics
                logger.info(
                    f"MCR² evaluation: {len(selected_rubrics)} existing + "
                    f"{len(batch_rubrics)} new = {len(combined_rubrics)} total",
                )

                mcr_results = mcr_selector.ultra_fast_adaptive_selection(
                    combined_rubrics,
                    batch_size=self.config.mcr_batch_size,
                    min_increment_threshold=self.config.min_increment_threshold,
                    patience=self.config.patience,
                    max_samples=min(self.config.max_total_rubrics, len(combined_rubrics)),
                )

                # Update selected rubrics
                selected_rubrics = mcr_results["selected_texts"]
                current_rate = mcr_results["final_coding_rate"]
                previous_rate = coding_rates[-1]
                increment = current_rate - previous_rate
                coding_rates.append(current_rate)

                logger.info(
                    f"MCR² results: {len(selected_rubrics)} selected, "
                    f"rate={current_rate:.6f}, increment={increment:.6f}",
                )

                # Check convergence
                if increment < self.config.min_increment_threshold:
                    low_increment_count += 1
                    logger.info(
                        f"Low increment: {increment:.6f} < {self.config.min_increment_threshold:.6f} "
                        f"(count: {low_increment_count}/{self.config.patience})",
                    )
                    if low_increment_count >= self.config.patience:
                        logger.info(
                            f"Converged after {iteration} iterations "
                            f"({self.config.patience} consecutive low increments)",
                        )
                        break
                else:
                    low_increment_count = 0

                # Check max rubrics limit
                if len(selected_rubrics) >= self.config.max_total_rubrics:
                    logger.info(f"Reached max rubrics limit: {self.config.max_total_rubrics}")
                    break

            logger.info(
                f"SMART_SAMPLING completed: {len(selected_rubrics)} rubrics selected " f"after {iteration} iterations",
            )
            return selected_rubrics

    # pylint: disable=unused-argument
    async def _categorize_query_rubrics(
        self,
        query_rubrics: List[str],
        **kwargs,
    ) -> str:
        """Categorize and format query-specific rubrics into a unified rubric set.

        This method consolidates query-specific rubrics into a single, coherent
        set of evaluation criteria suitable for use across all queries. The
        categorization process helps organize rubrics and reduce redundancy.

        Categorization behavior:
        - If enable_categorization is False: Preserves all rubrics as individual numbered items.
                                           Simple formatting with no semantic grouping.
        - If enable_categorization is True: Uses LLM-based semantic analysis to group similar rubrics
                                           into thematic categories (Theme-Tips structure).
                                           Attempts to merge into categories_number groups.

        Args:
            query_rubrics: List of query-specific rubrics to categorize.
                          Each string is a rubric generated for a specific query.
            **kwargs: Additional arguments passed to LLMRubricCategorizer.

        Returns:
            str: Formatted string containing consolidated rubrics.
                If categorization disabled: numbered list format.
                If categorization enabled: Theme-Tips hierarchical format.
                Falls back to numbered list format if categorization fails.
        """
        if not query_rubrics:
            logger.warning("No rubrics to categorize")
            return ""

        # If categorization is disabled: return all rubrics without categorization
        if not self.config.enable_categorization:
            logger.info(f"Categorization disabled: keeping all {len(query_rubrics)} rubrics")
            formatted_rubrics = "\n\n".join(
                [f"{i + 1}. {rubric}" for i, rubric in enumerate(query_rubrics)],
            )
            return formatted_rubrics

        # If categorization is enabled: use LLM-based categorization
        logger.info(f"Categorization enabled: categorizing {len(query_rubrics)} rubrics...")

        # Initialize rubric categorizer
        categorizer = LLMRubricCategorizer(
            num_categories=self.config.categories_number,
            model=self.config.model,
            language=self.config.language,
        )

        # Categorize rubrics
        try:
            categorized_rubrics, categorization_info = await categorizer.categorize_rubrics(
                query_rubrics,
            )

            if not categorized_rubrics:
                logger.error("Rubric categorization failed, falling back to numbered list format")
                # Fallback: return original rubrics as formatted string
                return "\n\n".join(
                    [f"{i + 1}. {rubric}" for i, rubric in enumerate(query_rubrics)],
                )

            logger.info(
                f"Successfully categorized into {categorization_info.get('num_categories', 0)} categories",
            )

            # Format categorized rubrics into a single string
            formatted_rubrics = "\n\n".join(
                [f"Rubric {i + 1}:\n{rubric}" for i, rubric in enumerate(categorized_rubrics)],
            )

            return formatted_rubrics

        except Exception as e:
            logger.error(f"Categorization error: {e}, falling back to numbered list format")
            return "\n\n".join(
                [f"{i + 1}. {rubric}" for i, rubric in enumerate(query_rubrics)],
            )
