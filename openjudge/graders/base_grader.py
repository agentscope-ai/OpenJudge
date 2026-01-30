# -*- coding: utf-8 -*-
"""Base class for graders.

This module defines the abstract base class for all graders. Graders are responsible
for evaluating the quality of responses based on various criteria and returning
either scores or rankings.
"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.schema import GraderError, GraderMode, GraderRank, GraderScore
from openjudge.runner.resource_executor.base_resource_executor import (
    BaseResourceExecutor,
)


class BaseGrader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the _aevaluate method.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description (str): Description of what this grader evaluates.
        strategy (BaseEvaluationStrategy): The evaluation strategy to use.
        kwargs (Dict[str, Any]): Additional keyword arguments.

    Example:
        >>> class MyGrader(BaseGrader):
        ...     async def _aevaluate(self, **kwargs):
        ...         # Implementation here
        ...         pass
        >>> grader = MyGrader(
        ...     name="test_grader",
        ...     mode=GraderMode.POINTWISE,
        ...     description="A test grader"
        ... )
        >>> print(grader.name)
        test_grader
        >>> print(grader.mode)
        GraderMode.POINTWISE
    """

    def __init__(
        self,
        name: str = "",
        mode: GraderMode = GraderMode.POINTWISE,
        description: str = "",
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation)
                  or LISTWISE (joint evaluation of multiple samples).
                  Defaults to POINTWISE.
            description: Human-readable description of what this grader evaluates.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional keyword arguments that will be stored and
                     accessible to subclasses.

        Example:
            >>> class MyGrader(BaseGrader):
            ...     async def aevaluate(self, **kwargs):
            ...         pass
            >>> grader = MyGrader(
            ...     name="accuracy_grader",
            ...     mode=GraderMode.POINTWISE,
            ...     description="Evaluates answer accuracy"
            ... )
            >>> print(grader.name)
            accuracy_grader
        """
        self.name = name
        self.mode = mode
        self.description = description
        self.strategy = strategy
        self.kwargs = kwargs

    @abstractmethod
    async def _aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank | GraderError:
        """Abstract method for performing the actual evaluation logic.

        This method must be implemented by all Grader subclasses. It performs
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
            Union[GraderScore, GraderRank, GraderError]: The evaluation result.

            In pointwise mode:
                GraderScore: Contains a numerical score and explanation.
                    - name (str): Name of the grader
                    - score (float): Numerical score (typically 0.0-1.0 or 1-5 scale)
                    - reason (str): Explanation of how the score was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

            In listwise mode:
                GraderRank: Contains a ranked list and explanation.
                    - name (str): Name of the grader
                    - rank (List[int]): Ranking of items (e.g., [1, 3, 2] means first
                      item is best, third item is second best, second item is worst)
                    - reason (str): Explanation of how the ranking was determined
                    - metadata (Dict[str, Any]): Additional evaluation information

        Example:
            >>> # Example for pointwise grader
            >>> class AccuracyGrader(BaseGrader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="accuracy",
            ...             mode=GraderMode.POINTWISE,
            ...             description="Evaluates factual accuracy of answers"
            ...         )
            ...
            ...     async def _aevaluate(self, query: str, response: str, **kwargs):
            ...         # Implementation would evaluate accuracy
            ...         return GraderScore(
            ...             name=self.name,
            ...             score=0.8,
            ...             reason="Answer is mostly accurate but missing some details"
            ...         )
            ...
            >>> # Example for listwise grader
            >>> class RelevanceRanker(BaseGrader):
            ...     def __init__(self):
            ...         super().__init__(
            ...             name="relevance_ranking",
            ...             mode=GraderMode.LISTWISE,
            ...             description="Ranks answers by relevance"
            ...         )
            ...
            >>> # Implementation would rank answers by relevance
        """

    # === [Core Interface] ===
    async def aevaluate(self, executor: BaseResourceExecutor | None = None, **kwargs: Any) -> Any:
        """
        Called by the Runner to inject the resource.
        """

        # Wrap the atomic evaluation task to submit to the resource
        async def managed_fn(**runtime_kwargs):
            # Submit to executor for execution
            # pylint: disable=protected-access
            # Create a deep copy of the grader to avoid modification
            if self.strategy:
                runtime_self = self.copy()
            else:
                runtime_self = self

            bound_method = runtime_self._aevaluate
            if executor is None:
                return await bound_method(**runtime_kwargs)
            else:
                return await executor.submit(bound_method, **runtime_kwargs)

        # Execute the strategy
        # The strategy receives a function with resource management capabilities
        if self.strategy:
            return await self.strategy.execute(managed_fn, **kwargs)
        else:
            return await managed_fn(**kwargs)

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Return the information about the grader's evaluation process.

        Such information helps callers understand how a grader works and the meaning of its evaluation result.

        Such information could be, but not limited to,
        1. The mechanism how the evaluation process works.
        2. The meaning of the score in a GraderScore return value.
        3. The meaning of the ranking in a GraderRank return value.
        4. The content of the template/prompt for LLM calls, if applicable, used by the evaluation logic.

        Define this as a static method, because the explanation belongs to the class level, not the instance level.

        Each subclass must implement its own,
        otherwise calling this method from a subclass returns value from its parent's method.
        """
        return {"warning": "This Grader has not implemented its own get_metadata()."}

    @classmethod
    def from_config(
        cls,
        config: dict,
    ) -> "BaseGrader":
        """Create a grader from a configuration dictionary.

        This class method creates a new grader instance using the provided configuration.
        It extracts standard grader properties (name, mode, description) from the config
        and passes any remaining items as additional keyword arguments.

        Args:
            config: A dictionary containing the grader configuration.
                   Expected keys include 'name', 'mode', 'description', and any
                   additional parameters required by specific grader implementations.

        Returns:
            BaseGrader: A new instance of the grader subclass.

        Example:
            >>> config = {
            ...     "name": "test_grader",
            ...     "mode": "pointwise",
            ...     "description": "A test grader"
            ... }
            >>> # grader = BaseGrader.from_config(config)
            >>> # Note: Cannot instantiate abstract class
        """
        # Extract standard grader properties from a copy to avoid mutating the input config
        config_copy = dict(config)
        name = config_copy.pop("name", "")
        mode_value = config_copy.pop("mode", GraderMode.POINTWISE)
        # Convert string to GraderMode if necessary
        if isinstance(mode_value, str):
            mode = GraderMode(mode_value)
        else:
            mode = mode_value
        description = config_copy.pop("description", "")

        # Create and return new instance with remaining config items as kwargs
        return cls(
            name=name,
            mode=mode,
            description=description,
            **config_copy,
        )

    def to_dict(self) -> dict:
        """Convert the grader to a dictionary representation.

        This method serializes the grader's essential properties (name, mode, description)
        and any additional keyword arguments into a dictionary. The mode is converted to
        its string value for serialization purposes.

        Returns:
            dict: A dictionary containing the serialized grader information.

        Example:
            >>> class MyGrader(BaseGrader):
            ...     pass  # Abstract methods not implemented for demo
            >>> grader = MyGrader(
            ...     name="test_grader",
            ...     mode=GraderMode.POINTWISE,
            ...     description="A test grader"
            ... )
            >>> data = grader.to_dict()
            >>> print("name" in data and "mode" in data)
            True
        """
        return {
            "name": self.name,
            "mode": self.mode.value,
            "description": self.description,
            "kwargs": self.kwargs,
        }

    def copy(self):
        """Create a copy of this grader for evaluation to prevent state sharing between samples.

        This method is called by the runner to create an isolated instance of the grader
        for each evaluation to prevent state pollution. By default, it attempts to create
        a new instance with the same parameters, but subclasses can override this to
        provide more specific behavior, especially when dealing with non-serializable
        objects like model connections.

        Returns:
            BaseGrader: A new instance of the grader with the same configuration
        """

        # # Get the class of this grader
        # grader_class = self.__class__

        # # Get constructor parameters by inspecting the grader's __init__ signature
        # sig = inspect.signature(grader_class.__init__)
        # init_params = {}

        # for param_name in sig.parameters:
        #     if param_name in ("self", "args", "kwargs"):  # Skip special params
        #         continue
        #     if hasattr(self, param_name) or param_name in self.__dict__:
        #         # Get value from instance, defaulting to the parameter's default if available
        #         param_default = sig.parameters[param_name].default
        #         if param_default is not inspect.Parameter.empty:
        #             init_params[param_name] = getattr(self, param_name, param_default)
        #         else:
        #             init_params[param_name] = self.__dict__.get(param_name, getattr(self, param_name, None))

        # # Create new instance with preserved parameters
        # copied_grader = grader_class(**init_params)

        # # Copy over any remaining attributes that weren't part of __init__
        # for attr_name, attr_value in self.__dict__.items():
        #     if attr_name not in init_params and not attr_name.startswith("_"):
        #         setattr(copied_grader, attr_name, attr_value)

        # return copied_grader

        return copy.copy(self)
