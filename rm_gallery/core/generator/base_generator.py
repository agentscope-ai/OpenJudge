# -*- coding: utf-8 -*-
"""Base classes for grader generator implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from rm_gallery.core.graders.base_grader import BaseGrader


@dataclass
class GraderGeneratorConfig:
    """Configuration model for grader generator.

    This class defines the base configuration parameters that control
    how a grader generator creates new graders.

    Attributes:
        grader_name (str): Human-readable name for the generated grader.
                          Defaults to "Generated Grader".
    """

    # Grader configuration
    grader_name: str = "Generated Grader"
    """Human-readable name for the generated grader"""


class BaseGraderGenerator(ABC):
    """Abstract base class for all grader generators.

    This class defines the interface that all grader generators must implement.
    Grader generators are responsible for creating Grader instances based on
    provided data and configuration.

    The generate method is the primary interface that subclasses must implement
    to create new grader instances from example data.
    """

    def __init__(self, config: GraderGeneratorConfig) -> None:
        """Initialize the grader generator with the provided configuration.

        Args:
            config (GraderGeneratorConfig): Configuration object containing
                                          parameters for grader generation.

                                          The configuration includes:
                                          - grader_name (str): Human-readable name for the generated grader.
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        dataset: List[dict],
        **kwargs,
    ) -> BaseGrader:
        """
        Generate a grader object from the given data.

        This is the main entry point for creating new graders. Implementations
        should use the provided data and configuration to construct a suitable
        grader instance.

        Args:
            dataset: List of data dictionaries containing examples for grader generation.
                 Each dictionary should contain the necessary information for
                 creating a grader, such as queries, responses, and scores.
            **kwargs: Additional implementation-specific arguments that may be
                     needed by specific generator implementations.

        Returns:
            BaseGrader: Generated grader object ready for evaluation tasks.
                       The specific type of grader depends on the implementation.

        Raises:
            NotImplementedError: If not implemented by subclass.

        Example:
            >>> config = GraderGeneratorConfig(grader_name="My Custom Grader")
            >>> dataset = [
            ...     {"query": "What is 2+2?", "response": "4", "score": 5},
            ...     {"query": "What is 3+3?", "response": "6", "score": 5}
            ... ]
            >>> generator = MyGraderGenerator(config)
            >>> grader = await generator.generate(dataset)
        """
