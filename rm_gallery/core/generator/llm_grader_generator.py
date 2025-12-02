# -*- coding: utf-8 -*-
"""Base class for LLM-based grader generators.

This module provides the foundation for grader generators that create
LLM-based graders by generating rubrics or other evaluation criteria
using language models.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import List

from rm_gallery.core.generator.base_generator import (
    BaseGraderGenerator,
    GraderGeneratorConfig,
)
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderMode
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.prompt_template import PromptTemplate


@dataclass
class LLMGraderGeneratorConfig(GraderGeneratorConfig):
    """Configuration for LLM-based grader generator.

    Extends the base grader generator configuration with parameters specific
    to LLM-based rubric generation.

    Attributes:
        model (BaseChatModel | None): Language model to use for generation.
                                     If None, a default model may be used.
        grader_mode (GraderMode): Mode for the generated grader (POINTWISE or LISTWISE).
                                 Defaults to POINTWISE.
        custom_evaluation_prompt (Template | None): Custom template for evaluation.
                                                   If None, a default template is used.
    """

    model: BaseChatModel | None = None
    grader_mode: GraderMode = GraderMode.POINTWISE
    custom_evaluation_prompt: PromptTemplate | None = None


class LLMGraderGenerator(BaseGraderGenerator):
    """Abstract base class for LLM-based grader generators.

    This class extends BaseGraderGenerator to provide specific functionality
    for generating LLM-based graders. It handles the common pattern of
    generating rubrics from data and then creating an LLMGrader that uses
    those rubrics for evaluation.

    Subclasses should implement the generate_rubrics method to define how
    rubrics are created from the provided data.
    """

    def __init__(self, config: LLMGraderGeneratorConfig) -> None:
        """Initialize the LLM grader generator with the provided configuration.

        Args:
            config (LLMGraderGeneratorConfig): Configuration object containing parameters for
                LLM-based grader generation. The configuration includes:
                - grader_name (str): Human-readable name for the generated grader.
                - model (BaseChatModel | None): Language model to use for generation.
                - grader_mode (GraderMode): Mode for the generated grader (POINTWISE or LISTWISE).
                - custom_evaluation_prompt (PromptTemplate | None): Custom template for evaluation.
        """
        super().__init__(config)

    async def generate(
        self,
        dataset: List[dict],
        **kwargs,
    ) -> LLMGrader:
        """
        Generate an LLMGrader based on rubrics created from provided data.

        This method orchestrates the grader generation process by first creating
        appropriate rubrics based on the input data, then constructing an LLMGrader
        that uses those rubrics for evaluation.

        Args:
            dataset: List of data dictionaries to generate rubrics from.
                 Each dictionary typically contains query, response, and score information.
            **kwargs: Additional arguments for the generation process.
                     Passed to the generate_rubrics method.

        Returns:
            LLMGrader: Generated LLMGrader with rubrics for evaluation.
                      Ready to evaluate new query-response pairs.

        Example:
            >>> config = LLMGraderGeneratorConfig(
            ...     grader_name="My LLM Grader",
            ...     model=my_model,
            ...     grader_mode=GraderMode.POINTWISE
            ... )
            >>> dataset = [
            ...     {"query": "What is 2+2?", "response": "4", "label_score": 5},
            ...     {"query": "What is 3+3?", "response": "6", "label_score": 5}
            ... ]
            >>> generator = MyLLMGraderGenerator(config)
            >>> grader = await generator.generate(dataset, config)
        """
        rubrics = await self._generate_rubrics(dataset, **kwargs)
        return LLMGrader(
            model=self.config.model,
            mode=self.config.grader_mode,
            template=self.config.custom_evaluation_prompt,
            rubrics=rubrics,
        )

    @abstractmethod
    async def _generate_rubrics(
        self,
        dataset: List[dict],
        **kwargs,
    ) -> str:
        """
        Generate rubrics from data.

        This abstract method must be implemented by subclasses to define
        how rubrics are generated from the provided data. The generated
        rubrics will be used to configure the final LLMGrader.

        Args:
            dataset : List of data dictionaries containing examples for rubric generation.
                 Each dictionary should contain query, response, and potentially score information.
            **kwargs: Additional arguments for the rubric generation process.
                     May include things like language preferences or specific guidelines.

        Returns:
            str: Generated rubrics in a format suitable for LLMGrader.
                Typically a formatted string that can be inserted into evaluation prompts.

        Example:
            >>> dataset = [{"query": "What is 2+2?", "response": "4", "label_score": 5}]
            >>> rubrics = await generator.generate_rubrics(dataset)
            >>> print(rubrics)
            "1. Mathematical accuracy: The response should contain correct mathematical calculations..."
        """
