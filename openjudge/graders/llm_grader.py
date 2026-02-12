# -*- coding: utf-8 -*-
"""
LLM-based grader implementation for evaluating model responses.

This module provides the LLMGrader class, which uses large language models to evaluate
the quality of model responses based on customizable rubrics and templates. The grader
supports both pointwise evaluation (scoring individual responses) and listwise evaluation
(ranking multiple responses).

The module integrates with various language models through the BaseChatModel interface
and provides structured output parsing for consistent evaluation results.

Classes:
    LLMGrader: Main class for LLM-based evaluation with configurable templates and rubrics.
"""

import os
import textwrap
from typing import Any, Callable, Dict, Type

from pydantic import BaseModel

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import (
    BaseGrader,
    GraderMode,
    GraderRank,
    GraderScore,
)
from openjudge.graders.schema import GraderRankCallback, GraderScoreCallback
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate


class LLMGrader(BaseGrader):
    """LLM-based grader that uses large language models for evaluation.

    This class extends the base Grader class to provide LLM-based evaluation capabilities.
    It uses a language model to perform evaluations according to specified rubrics and templates.

    The LLMGrader constructs prompts using a template, sends them to an LLM, and parses
    the structured response into either a GraderScore or GraderRank depending on the mode.

    Attributes:
        template (Template): The template for generating prompts.
        model (BaseChatModel): The language model used for evaluation.
        rubrics (str): The rubrics used for evaluation.
        language (LanguageEnum): The language for the evaluation.
        structured_model (Type[BaseModel]): Pydantic model to process model response
                                             into GraderScore or GraderRank.
        callback (Callable): Function to process model response metadata.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        language: LanguageEnum | str | None = None,
        description: str = "",
        template: str | dict | PromptTemplate | None = None,
        structured_model: Type[BaseModel] | None = None,
        callback: Callable | None = None,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """Initialize an LLMGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation)
                  or LISTWISE (joint evaluation of multiple samples).
                  Defaults to POINTWISE.
            language: The language of the grader. Can be LanguageEnum, string, or None.
                     If None, defaults to environment variable LANGUAGE or "en".
            description: Human-readable description of what this grader evaluates.
            template: The template for generating prompts. Defines how inputs are formatted
                     for the LLM. Can be a dict or PromptTemplate object.
            structured_model: The Pydantic model for structured output parsing.
                      Can be one of the following:
                      1. A Pydantic BaseModel subclass for structured output parsing
                      2. None, in which case uses GraderScoreCallback for POINTWISE mode
                         or GraderRankCallback for LISTWISE mode
            callback: The callback function for processing model response metadata.
                      Can be one of the following:
                      1. A Callable that processes the response and populates metadata
                      2. None, in which case no callback processing is performed
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional keyword arguments passed to the parent Grader class and
                     used in the template rendering.
        """
        super().__init__(name=name, mode=mode, description=description, strategy=strategy, **kwargs)

        # Handle language parameter
        if not language:
            language = os.environ.get("LANGUAGE", "en")

        if isinstance(language, str):
            # Convert string to LanguageEnum
            self.language = (
                LanguageEnum(language) if language in [item.value for item in LanguageEnum] else LanguageEnum.EN
            )
        else:
            self.language = language

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
            # Support list of message dicts or ChatMessage objects
            self.template = PromptTemplate.from_prompt(template)
        elif isinstance(template, dict):
            self.template = PromptTemplate(**template)
        else:
            raise ValueError("Template must be a str, list, dict or PromptTemplate object")

        # Initialize model
        if isinstance(model, dict):
            self._model_config = model
            self.model = OpenAIChatModel(**model)
        else:
            self._model_config = None
            self.model = model

        # Store parameters
        self.structured_model = structured_model
        self.callback = callback

        # Set default structured_model if not provided
        if not self.structured_model:
            if self.mode == GraderMode.LISTWISE:
                self.structured_model = GraderRankCallback
            else:
                self.structured_model = GraderScoreCallback

        if not (self.language and self.template and self.model):
            raise RuntimeError("Missing required attributes: language, template, or model")

    def to_dict(self) -> dict:
        """Convert the LLMGrader to a dictionary representation.

        This method serializes the LLMGrader's properties (name, mode, description, template,
        model, and rubrics) into a dictionary. The mode is converted to its string value,
        and the template and model are converted to dictionaries if they are not already.

        Returns:
            A dictionary containing the serialized LLMGrader information.
        """
        d = {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "template": (self.template.model_dump() if isinstance(self.template, PromptTemplate) else self.template),
            **self.kwargs,
        }

        # Include model config if the data type is dict
        if hasattr(self, "_model_config") and self._model_config:
            d["model"] = self._model_config
        return d

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "LLMGrader":
        """Create an LLMGrader from a configuration dictionary.

        This class method creates a new LLMGrader instance using the provided configuration.
        It extracts standard grader properties (name, mode, description) and LLM-specific
        properties (template, model, rubrics) from the config.

        Args:
            config: A dictionary containing the LLMGrader configuration.
                   Expected keys include 'name', 'mode', 'description', 'template',
                   'model', 'rubrics', and any additional parameters.

        Returns:
            A new LLMGrader instance.
        """
        # Extract standard grader properties
        name = config.pop("name", "")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "")

        # Extract LLMGrader-specific properties
        template = config.pop("template", {})
        model = config.pop("model", {})

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            template=template,
            model=model,
            **config,
        )

    async def _aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Performs evaluation using a large language model according to the configured
        template and rubrics. The method formats the input parameters according to the
        template, sends the request to the LLM, and parses the structured response into
        either a GraderScore or GraderRank depending on the grader mode.

        The callback mechanism supports two modes:
        1. Callable functions that process the response and populate metadata
        2. Pydantic BaseModel subclasses for structured output parsing

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     These are passed to the LLM template and typically include fields
                     like 'query', 'answer', 'context', etc. The specific fields depend
                     on the template definition.

        Returns:
            GraderScore | GraderRank: The evaluation result from the LLM.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - name (str): Name of the grader
                    - score (float): Numerical score assigned by the LLM
                    - reason (str): LLM's explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - name (str): Name of the grader
                    - rank (List[int]): Ranking of items as determined by the LLM
                    - reason (str): LLM's explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

        Raises:
            ValueError: If the grader mode is unsupported or chat template is not set.

        Example:
            >>> # Example for pointwise LLM grader
            >>> from openjudge.model.dashscope_llm import DashScopeLLM
            >>> grader = LLMGrader(
            ...     name="helpfulness",
            ...     mode=GraderMode.POINTWISE,
            ...     template=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "{query}\\n{answer}\\n\\nRate helpfulness:"}
            ...     ],
            ...     model=DashScopeLLM(model="qwen-plus")
            ... )
            >>> result = await grader.aevaluate(
            ...     query="How do I make a cake?",
            ...     answer="Preheat oven to 350F, mix ingredients, bake for 30 minutes."
            ... )
            >>> print(result.score, result.reason)
            0.9 Well-structured answer providing essential steps

            >>> # Example for listwise LLM grader
            >>> ranking_grader = LLMGrader(
            ...     name="relevance_ranking",
            ...     mode=GraderMode.LISTWISE,
            ...     template=[
            ...         {"role": "system", "content": "Rank the following answers by relevance."},
            ...         {"role": "user",
            ...          "content": "Query: {query}\\nAnswers:\\n1. {answer_1}\\n2. {answer_2}"}
            ...     ],
            ...     model=DashScopeLLM(model="qwen-plus")
            ... )
            >>> result = await ranking_grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     answer_1="Paris is the capital of France.",
            ...     answer_2="France is a country in Europe."
            ... )
            >>> print(result.rank, result.reason)
            [1, 2] First answer directly addresses the query while second is tangential
        """

        params = {**self.kwargs}
        params.update(kwargs)
        messages = self.template.format(language=self.language, **params)
        chat_response = await self.model.achat(
            messages=list(messages),
            structured_model=self.structured_model,
            callback=self.callback,
        )

        # Handle both streaming and non-streaming responses
        if hasattr(chat_response, "__aiter__"):
            # Collect the last chunk from the async iterator
            async for chunk in chat_response:  # type: ignore
                chat_response = chunk  # Iterate through all chunks, keeping the last one

        parsed = getattr(chat_response, "parsed", {}) or {}

        if self.mode == GraderMode.LISTWISE:
            rank = parsed.pop("rank")
            reason = parsed.pop("reason")
            result = GraderRank(
                name=self.name,
                rank=rank,  # type: ignore
                reason=reason,  # type: ignore
                metadata=parsed,  # type: ignore
            )
        elif self.mode == GraderMode.POINTWISE:
            score = parsed.pop("score")
            reason = parsed.pop("reason")
            result = GraderScore(
                name=self.name,
                score=score,  # type: ignore
                reason=reason,  # type: ignore
                metadata=parsed,  # type: ignore
            )
        else:
            raise ValueError(f"Unsupported grader mode: {self.mode}")
        return result

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return the docstring of the aevaluate method to explain how LLMGrader works with LLM."""
        return {"aevaluate": LLMGrader._aevaluate.__doc__, "prompt": {}}
