# -*- coding: utf-8 -*-
import asyncio

# import inspect
from abc import ABC, abstractmethod
from functools import partial
import os
from typing import Callable, List

from loguru import logger

from rm_gallery.core.model.openai_llm import OpenAIChatModel
from rm_gallery.core.schema.data import (
    DataSample,
    DataSampleParser,
    parse_data_sample,
)
from rm_gallery.core.model.base import ChatModelBase
from rm_gallery.core.schema.grader import (
    _GraderRank,
    _GraderScore,
    GraderError,
    GraderInfo,
    GraderMode,
    GraderRank,
    GraderScore,
)
from rm_gallery.core.schema.template import Chat, Template
from rm_gallery.core.utils.concurrency import ConcurrencyManager


class Grader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the evaluate method.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description: The description of the grader.
    """

    def __init__(
        self,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        **kwargs,
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader.
            mode: The grader mode. Defaults to POINTWISE.
            description: The description of the grader.
        """
        self._name = name
        self.mode = mode
        self.description = description
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Get the name of the grader.

        Returns:
            str: The name of the grader.
        """
        return self._name

    @property
    def meta(self) -> GraderInfo:
        """Get the metadata of the grader.

        Returns:
            GraderInfo: The metadata of the grader.
        """
        return GraderInfo(
            name=self.name,
            mode=self.mode,
            description=self.description,
        )

    @abstractmethod
    async def a_evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate method to be implemented by subclasses.

        This abstract method must be implemented by all Grader subclasses. It performs
        the actual evaluation logic and returns either a score or a ranking based on
        the grader's mode (pointwise or listwise).

        In pointwise mode, each sample is evaluated independently, returning a
        GraderScore with a numerical value and explanation. In listwise mode, all
        samples are evaluated together, returning a GraderRank with a ranked list and
        explanation.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     The specific arguments depend on the grader implementation but
                     typically include fields like 'query', 'answer', 'context', etc.

        Returns:
            GraderScore | GraderRank: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score (typically 0.0-1.0 or 1-5 scale)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items (e.g., [1, 3, 2] means first
                      item is best, third item is second best, second item is worst)
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise grader
            >>> class AccuracyGrader(Grader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="accuracy",
            ...             mode=GraderMode.POINTWISE,
            ...             description="Evaluates factual accuracy of answers"
            ...         )
            ...
            ...     async def a_evaluate(self, query: str, answer: str, **kwargs) -> GraderScore:
            ...         # Implementation would evaluate accuracy
            ...         return GraderScore(
            ...             score=0.8,
            ...             reason="Answer is mostly accurate but missing some details"
            ...         )
            ...
            >>> # Example for listwise grader
            >>> class RelevanceRanker(Grader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="relevance_ranking",
            ...             mode=GraderMode.LISTWISE,
            ...             description="Ranks answers by relevance"
            ...         )
            ...
            ...     async def a_evaluate(self, query: str, answer_1: str, answer_2: str, **kwargs) -> GraderRank:
            ...         # Implementation would rank answers by relevance
            ...         return GraderRank(
            ...             rank=[1, 2],
            ...             reason="First answer is more relevant to the query than the second"
            ...         )
        """
        ...

    async def _a_safe_evaluate(
        self,
        **kwargs,
    ) -> GraderScore | GraderRank | GraderError:
        """Safely evaluate, handling exceptions gracefully and control concurrency.

        Args:
            **kwargs: Arguments for the evaluation.

        Returns:
            GraderScore | GraderRank: The grader result or error result.
        """
        concurrency_manager = ConcurrencyManager()

        async def _evaluate_task():
            try:
                # Check if self.evaluate is a coroutine function
                if asyncio.iscoroutinefunction(self.a_evaluate):
                    result = await self.a_evaluate(**kwargs)
                else:
                    # If it's a synchronous function, run it in a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: self.a_evaluate(**kwargs),
                    )
                return result
            except Exception as e:
                error_msg = f"Error in {self.name} during evaluation: {str(e)}"
                logger.error(error_msg)
                return GraderError(name=self.name, reason=error_msg)

        # Use the concurrency manager to control execution
        return await concurrency_manager.run_with_concurrency_control(
            _evaluate_task(),
        )

    async def _a_evaluate_data_sample(
        self,
        data_sample: DataSample,
        parser: DataSampleParser | None = None,
        *args,
        **kwargs,
    ) -> List[GraderScore]:
        """Evaluate a data sample using this grader.

        This method evaluates a data sample based on the grader's mode (pointwise or listwise).
        In pointwise mode, each sample is evaluated individually. In listwise mode, all samples
        are evaluated together in one call.

        Args:
            data_sample (DataSample): The data sample to evaluate.
            parser (DataSampleParser | None, optional): Optional parser to transform the
                data sample before evaluation. Defaults to None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[GraderScore]: A list of grader scores, one for each sample in the data sample.

        Raises:
            ValueError: If the grader mode is invalid.
        """
        if parser is not None:
            data_sample = parse_data_sample(data_sample, parser)

        if self.mode == GraderMode.POINTWISE:
            # Pointwise: Evaluate each sample individually
            coroutines = [
                self._a_safe_evaluate(**data_sample.data, **sample)
                for sample in data_sample.samples
            ]
            results: List[GraderScore] = await asyncio.gather(*coroutines)  # type: ignore
            _results = []
            for result in results:
                if isinstance(result, GraderScore):
                    _results.append(result)
                elif isinstance(result, GraderError):
                    _results.append(
                        GraderScore(
                            name=result.name, score=0.0, reason=result.reason
                        ),
                    )
                else:
                    raise ValueError(f"Invalid result type: {type(result)}")
            return results

        elif self.mode == GraderMode.LISTWISE:
            # Listwise: Evaluate all samples together in one call
            params = {key: value for key, value in kwargs.items()}
            params.update(data_sample.data)
            if len(data_sample.samples) > 1:
                if data_sample.samples:
                    for key in data_sample.samples[0].keys():
                        params[key] = [
                            sample[key] for sample in data_sample.samples
                        ]

            result = await self._a_safe_evaluate(**params)
            if isinstance(result, GraderRank):
                results = [
                    GraderScore(
                        name=result.name,
                        score=score,
                        reason=result.reason,
                        metadata=result.metadata,
                    )
                    for score in result.rank
                ]
            elif isinstance(result, GraderError):
                results = [
                    GraderScore(
                        name=result.name, score=0.0, reason=result.reason
                    )
                ]
            else:
                raise ValueError(f"Invalid result type: {type(result)}")

            return results
        else:
            raise ValueError(f"Invalid grader mode: {self.mode}")

    async def a_evaluate_data_samples(
        self,
        data_samples: List[DataSample] | DataSample,
        parser: DataSampleParser | Callable | None = None,
        *args,
        **kwargs,
    ) -> List[List[GraderScore]] | List[GraderScore]:
        """Main entry point to evaluate data sample.

        Evaluates one data samples using the  grader.

        Args:
            data_sample (DataSample):
                The data sample to evaluate.
                DataSample consists of:
                    - data: A dictionary containing shared data for all samples
                    - samples: A list of dictionaries, each representing an individual
                    sample to evaluate

            parser (DataSampleParser | Callable | None, optional):
                Optional parser to transform the data sample before evaluation. This
                allows for mapping field names between the data structure and what the
                grader expects. Can be:
                1. A dictionary with direct field mappings where paths start with "data" or "sample"
                2. A callable function that takes a DataSample and returns a DataSample
                3. None, in which case the data sample is used as is
                Defaults to None.
            *args:
                Additional positional arguments to pass to the grader.
            **kwargs:
                Additional keyword arguments to pass to the grader.

        Returns:
            List[GraderScore] | List[List[GraderScore]]:
                For a single DataSample: a list of GraderScore objects, one for each
                sample within the DataSample.

                For a list of DataSamples: a list of lists of GraderScore objects,
                where each inner list contains the scores for the corresponding
                DataSample in the input list.

                Each GraderScore contains:
                    - score: A numerical score assigned by the grader
                    - reason: Explanation of how the score was determined
                    - metadata: Optional additional information from the evaluation

        Raises:
            ValueError: If grader function signature is invalid.

        Example:
            >>> from rm_gallery.core.schema.data import DataSample
            >>> from rm_gallery.core.grader.base import LLMGrader, evaluate
            >>>
            >>> # Create data sample
            >>> data_sample = DataSample(
            ...     data={"query": "What is the capital of France?"},
            ...     samples=[
            ...         {"answer": "Paris"},
            ...         {"answer": "London"}
            ...     ]
            ... )
            >>>
            >>> # Create grader
            >>> grader = LLMGrader(
            ...     name="factual_accuracy",
            ...     template=[
            ...         {
            ...             "role": "system",
            ...             "content": "You are evaluating factual accuracy."
            ...         },
            ...         {
            ...             "role": "user",
            ...             "content": "Question: {query}\\nAnswer: {answer}\\nRate accuracy (0-1):"
            ...         }
            ...     ],
            ...     model={"model_name": "qwen-plus"}
            ... )
            >>>
            >>> # Evaluate
            >>> results = await grader.a_evaluate_data_samples(data_sample=data_sample)
            >>> print(results)
        """
        if isinstance(data_samples, list):
            corutines: List = [
                self._a_evaluate_data_sample(
                    data_sample=_data_sample,
                    parser=parser,
                    *args,
                    **kwargs,
                )
                for _data_sample in data_samples
            ]
            results = await asyncio.gather(*corutines)
            return list(results)  # type: ignore
        else:
            return await self._a_evaluate_data_sample(
                data_sample=data_samples, parser=parser, *args, **kwargs
            )

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "Grader":
        """Create a grader from a configuration dictionary.

        This class method creates a new grader instance using the provided configuration.
        It extracts standard grader properties (name, mode, description) from the config
        and passes any remaining items as additional keyword arguments.

        Args:
            config: A dictionary containing the grader configuration.
                   Expected keys include 'name', 'mode', 'description', and any
                   additional parameters required by specific grader implementations.

        Returns:
            A new instance of the grader subclass.
        """
        # Extract standard grader properties
        name = config.pop("name", "")
        mode = config.pop("mode", GraderMode.POINTWISE)
        description = config.pop("description", "")

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            **config,
        )

    def to_dict(self) -> dict:
        """Convert the grader to a dictionary representation.

        This method serializes the grader's essential properties (name, mode, description)
        and any additional keyword arguments into a dictionary. The mode is converted to
        its string value for serialization purposes.

        Returns:
            A dictionary containing the serialized grader information.
        """
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "kwargs": self.kwargs,
        }


class LLMGrader(Grader):
    """LLM-based grader.

    This class extends the base Grader class to provide LLM-based evaluation capabilities.
    It uses a language model to perform evaluations according to specified rubrics.

    Attributes:
        template: The template for generating prompts.
        model: The language model used for evaluation.
        rubrics: The rubrics used for evaluation.
    """

    def __init__(
        self,
        model: ChatModelBase | dict,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        template: dict | Template = {},
        callback: Callable | None = None,
        rubrics: str = "",
    ):
        """Initialize an LLMGrader.

        Args:
            model: The language model used for evaluation. Can be either a ChatModelBase
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            name: The name of the grader.
            mode: The grader mode. Defaults to POINTWISE.
            description: The description of the grader.
            template: The template for generating prompts.
            callback: The callback function for processing model response to GraderScore or GraderRank.
            rubrics: The rubrics used for evaluation.
        """
        super().__init__(name=name, mode=mode, description=description)

        self.template = (
            template
            if isinstance(template, Template)
            else Template(**template)
        )

        if callback:
            self.callback = callback
        else:
            self.callback = (
                _GraderScore
                if self.mode == GraderMode.POINTWISE
                else _GraderRank
            )

        if isinstance(model, dict):
            model = OpenAIChatModel(**model)

        self.model = model
        self.rubrics = rubrics
        if self.template is not None and self.model is not None:
            self.chat = Chat(template=self.template, model=self.model)
        else:
            raise ValueError("Chat template or model is not set")

    @property
    def meta(self) -> GraderInfo:
        """Get the metadata of the grader.

        Returns:
            GraderInfo: The metadata of the grader.
        """
        return GraderInfo(
            name=self.name,
            mode=self.mode,
            description=self.description,
        )

    def to_dict(self) -> dict:
        """Convert the LLMGrader to a dictionary representation.

        This method serializes the LLMGrader's properties (name, mode, description, template,
        model, and rubrics) into a dictionary. The mode is converted to its string value,
        and the template and model are converted to dictionaries if they are not already.

        Returns:
            A dictionary containing the serialized LLMGrader information.
        """
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "template": (
                self.template.model_dump()
                if isinstance(self.template, Template)
                else self.template
            ),
            "rubrics": self.rubrics,
        }

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
        rubrics = config.pop("rubrics", "")

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            template=template,
            model=model,
            rubrics=rubrics,
            **config,
        )

    async def a_evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using LLM.

        Performs evaluation using a large language model according to the configured
        template and rubrics. The method formats the input parameters according to the
        template, sends the request to the LLM, and parses the structured response into
        either a GraderScore or GraderRank depending on the grader mode.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     These are passed to the LLM template and typically include fields
                     like 'query', 'answer', 'context', etc. The specific fields depend
                     on the template definition.

        Returns:
            GraderScore | GraderRank: The evaluation result from the LLM.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score assigned by the LLM
                    - reason (str): LLM's explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items as determined by the LLM
                    - reason (str): LLM's explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional information from the LLM

        Raises:
            ValueError: If the grader mode is unsupported or chat template is not set.

        Example:
            >>> # Example for pointwise LLM grader
            >>> from rm_gallery.core.model.dashscope_llm import DashScopeLLM
            >>> grader = LLMGrader(
            ...     name="helpfulness",
            ...     mode=GraderMode.POINTWISE,
            ...     template=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "{query}\\n{answer}\\n\\nRate helpfulness:"}
            ...     ],
            ...     model=DashScopeLLM(model_name="qwen-plus"),
            ...     rubrics="Rate the helpfulness of the answer (0.0-1.0)"
            ... )
            >>> result = await grader.evaluate(
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
            ...         {"role": "user", "content": "Query: {query}\\nAnswers:\\n1. {answer_1}\\n2. {answer_2}"}
            ...     ],
            ...     model=DashScopeLLM(model_name="qwen-plus"),
            ...     rubrics="Rank answers by relevance (1 is most relevant)"
            ... )
            >>> result = await ranking_grader.evaluate(
            ...     query="What is the capital of France?",
            ...     answer_1="Paris is the capital of France.",
            ...     answer_2="France is a country in Europe."
            ... )
            >>> print(result.rank, result.reason)
            [1, 2] First answer directly addresses the query while second is tangential
        """
        # Check if chat is not None before calling it
        if self.chat is None:
            raise ValueError("Chat template is not set")
        params = {"rubrics": self.rubrics, **self.kwargs}
        params.update(kwargs)

        language = os.environ.get("LANGUAGE", "en")
        from rm_gallery.core.schema.template import LanguageEnum

        language_enum = (
            LanguageEnum(language)
            if language in [item.value for item in LanguageEnum]
            else LanguageEnum.EN
        )
        response = await self.chat(
            callback=self.callback, language=language_enum, **params
        )
        metadata = response.metadata if response.metadata else {}
        if self.mode == GraderMode.LISTWISE:
            rank = metadata.pop("rank")
            reason = metadata.pop("reason")
            result = GraderRank(
                name=self.name,
                rank=rank,  # type: ignore
                reason=reason,  # type: ignore
                metadata=metadata,  # type: ignore
            )
        elif self.mode == GraderMode.POINTWISE:
            score = metadata.pop("score")
            reason = metadata.pop("reason")
            result = GraderScore(
                name=self.name,
                score=score,  # type: ignore
                reason=reason,  # type: ignore
                metadata=metadata,  # type: ignore
            )
        else:
            raise ValueError(f"Unsupported grader mode: {self.mode}")
        return result


class FunctionGrader(Grader):
    """Function-based grader.

    A grader that uses a provided function to perform evaluations.

    Attributes:
        func (Callable): The function to use for evaluation.
        name (str): The name of the grader.
        mode (GraderMode): The grader mode.
    """

    def __init__(
        self,
        func: Callable,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        **kwargs,
    ):
        """Initialize a FunctionGrader.

        Args:
            func: The function to use for evaluation.
            name: The name of the grader.
            mode: The grader mode.
            description: The description of the grader.
        """
        super().__init__(
            name,
            mode,
            description,
            **kwargs,
        )
        self.func = func

    async def a_evaluate(self, **kwargs) -> GraderScore | GraderRank:
        """Evaluate using a function.

        Performs evaluation by calling the wrapped function with the provided arguments.
        The function must return either a GraderScore (for pointwise mode) or a
        GraderRank (for listwise mode) object.

        Args:
            **kwargs: Arbitrary keyword arguments containing the data to be evaluated.
                     These are passed directly to the wrapped function and typically
                     include fields like 'query', 'answer', 'context', etc. The specific
                     fields depend on the function's requirements.

        Returns:
            GraderScore | GraderRank: The evaluation result from the wrapped function.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - score (float): Numerical score computed by the function
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - rank (List[int]): Ranking of items computed by the function
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Raises:
            TypeError: If result type doesn't match grader mode (e.g., function returns
                      GraderScore in listwise mode).

        Example:
            >>> # Example for pointwise function grader
            >>> def accuracy_function(query: str, answer: str) -> GraderScore:
            ...     # Simple accuracy function - checks if answer contains key facts
            ...     if "Paris" in answer and "capital" in answer.lower():
            ...         return GraderScore(score=1.0, reason="Correctly identifies Paris as capital")
            ...     else:
            ...         return GraderScore(score=0.0, reason="Missing key information")
            ...
            >>> grader = FunctionGrader(
            ...     func=accuracy_function,
            ...     name="accuracy_checker",
            ...     mode=GraderMode.POINTWISE
            ... )
            >>> result = await grader.evaluate(
            ...     query="What is the capital of France?",
            ...     answer="Paris is the capital of France."
            ... )
            >>> print(result.score, result.reason)
            1.0 Correctly identifies Paris as capital

            >>> # Example for listwise function grader
            >>> def relevance_ranker(query: str, answer_1: str, answer_2: str) -> GraderRank:
            ...     # Simple ranking function - longer answer assumed more relevant
            ...     if len(answer_1) > len(answer_2):
            ...         return GraderRank(rank=[1, 2], reason="First answer is more detailed")
            ...     else:
            ...         return GraderRank(rank=[2, 1], reason="Second answer is more detailed")
            ...
            >>> ranking_grader = FunctionGrader(
            ...     func=relevance_ranker,
            ...     name="length_ranker",
            ...     mode=GraderMode.LISTWISE
            ... )
            >>> result = await ranking_grader.evaluate(
            ...     query="Explain photosynthesis",
            ...     answer_1="Photosynthesis converts light to energy.",
            ...     answer_2="Photosynthesis is the process by which plants convert light energy into chemical energy."
            ... )
            >>> print(result.rank, result.reason)
            [2, 1] Second answer is more detailed
        """
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.func, **kwargs)

        # Check return type based on grader mode
        if self.mode == GraderMode.POINTWISE:
            if not isinstance(result, GraderScore):
                raise TypeError(
                    f"Expected GraderScore for pointwise mode, got {type(result)}",
                )
        elif self.mode == GraderMode.LISTWISE:
            if not isinstance(result, GraderRank):
                raise TypeError(
                    f"Expected GraderRank for listwise mode, got {type(result)}",
                )
        else:
            raise ValueError(f"Unsupported grader mode: {self.mode}")

        return result

    @classmethod
    def wrap(cls, func: Callable) -> Callable:
        """Wrap a function as a grader.

        Returns:
            The Callable grader.
        """

        return partial(FunctionGrader, func=func)
