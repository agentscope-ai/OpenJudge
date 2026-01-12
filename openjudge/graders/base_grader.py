# -*- coding: utf-8 -*-
"""Base class for graders.

This module defines the abstract base class for all graders. Graders are responsible
for evaluating the quality of responses based on various criteria and returning
either scores or rankings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from openjudge.graders.schema import GraderError, GraderMode, GraderRank, GraderScore


class BaseGrader(ABC):
    """Base class for graders.

    This abstract base class defines the interface for all graders.
    Subclasses must implement the aevaluate method.

    Attributes:
        name (str): The name of the grader.
        mode (GraderMode): The grader mode (pointwise or listwise).
        description (str): Description of what this grader evaluates.
        kwargs (Dict[str, Any]): Additional keyword arguments.

    Example:
        >>> class MyGrader(BaseGrader):
        ...     async def aevaluate(self, **kwargs):
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
        **kwargs: Any,
    ):
        """Initialize a Grader.

        Args:
            name: The name of the grader. Used for identification and logging.
            mode: The grader mode. Either POINTWISE (individual sample evaluation)
                  or LISTWISE (joint evaluation of multiple samples).
                  Defaults to POINTWISE.
            description: Human-readable description of what this grader evaluates.
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
        self.kwargs = kwargs

    @abstractmethod
    async def aevaluate(self, **kwargs: Any) -> GraderScore | GraderRank | GraderError:
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
            ...     async def aevaluate(self, query: str, response: str, **kwargs):
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
