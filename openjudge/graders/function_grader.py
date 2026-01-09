# -*- coding: utf-8 -*-
"""
Function-based grader implementation for evaluating model responses.

This module provides the FunctionGrader class, which allows users to define custom
evaluation functions for assessing model responses. Unlike the LLMGrader which uses
large language models for evaluation, FunctionGrader executes user-defined functions
that implement specific evaluation logic.

The grader supports both pointwise evaluation (scoring individual responses) and
listwise evaluation (ranking multiple responses) through custom functions that
return appropriate GraderScore or GraderRank objects.

Classes:
    FunctionGrader: Main class for function-based evaluation with support for both
                   synchronous and asynchronous evaluation functions.
"""

import asyncio
from functools import partial
from typing import Any, Callable

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderRank, GraderScore


class FunctionGrader(BaseGrader):
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
        **kwargs: Any,
    ):
        """Initialize a FunctionGrader.

        Args:
            func: The function to use for evaluation. This function will be called
                  with the evaluation data and must return either a GraderScore (for
                  pointwise mode) or a GraderRank (for listwise mode).

                  For pointwise mode, typical signature:
                  ```async def my_func(query: str, response: str, **kwargs) -> GraderScore:```

                  For listwise mode, typical signature:
                  ```async def my_func(query: str, responses: List[str], **kwargs) -> GraderRank:```
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation)
                  or LISTWISE (joint evaluation of multiple samples).
                  Defaults to POINTWISE.
            description: Human-readable description of what this grader evaluates.
            **kwargs: Additional keyword arguments passed to the parent Grader class.
        """
        super().__init__(
            name,
            mode,
            description,
            **kwargs,
        )
        self.func = func

    async def aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank:
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
            >>> def accuracy_function(query: str, response: str) -> GraderScore:
            ...     # Simple accuracy function - checks if answer contains key facts
            ...     if "Paris" in response and "capital" in response.lower():
            ...         return GraderScore(name=self.name,
            ...                            score=1.0,
            ...                            reason="Correctly identifies Paris as capital")
            ...     else:
            ...         return GraderScore(name=self.name,
            ...                            score=0.0,
            ...                            reason="Missing key information")
            ...
            >>> grader = FunctionGrader(
            ...     func=accuracy_function,
            ...     name="accuracy_checker",
            ...     mode=GraderMode.POINTWISE
            ... )
            >>> result = await grader.aevaluate(
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
            >>> result = await ranking_grader.aevaluate(
            ...     query="Explain photosynthesis",
            ...     answer_1="Photosynthesis converts light to energy.",
            ...     answer_2="Photosynthesis is the process by which plants convert light "
                             "energy into chemical energy."
            ... )
            >>> print(result.rank, result.reason)
            [2, 1] Second answer is more detailed
        """
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            # Fix: Pass kwargs as a single dictionary argument to the function
            result = await loop.run_in_executor(None, lambda: self.func(**kwargs))

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
        """Decorator to wrap a function as a FunctionGrader.

        This class method allows you to easily convert a regular Python function
        into a FunctionGrader instance. The wrapped function must follow the
        FunctionGrader requirements and return either a GraderScore or GraderRank.

        Args:
            func: The function to wrap as a grader. Must return GraderScore or GraderRank.

        Returns:
            A partially applied FunctionGrader constructor that can be instantiated
            with additional parameters like mode, name, and description.

        Example:
            >>> @FunctionGrader.wrap
            >>> def my_accuracy_function(query: str, response: str) -> GraderScore:
            >>>     # Custom accuracy evaluation logic
            >>>     score = calculate_accuracy(query, response)
            >>>     return GraderScore(name="accuracy", score=score, reason="Custom calculation")
            >>>
            >>> # Create the grader instance
            >>> accuracy_grader = my_accuracy_function(mode=GraderMode.POINTWISE,
            ...                                       name="my_accuracy",
            ...                                       description="My custom accuracy evaluator")
        """

        return partial(FunctionGrader, func=func)
