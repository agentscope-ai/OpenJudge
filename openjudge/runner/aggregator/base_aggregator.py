# -*- coding: utf-8 -*-
"""
Base class for aggregating results from multiple graders.
"""

from abc import ABC, abstractmethod
from typing import Dict

from openjudge.graders.schema import GraderResult


class BaseAggregator(ABC):
    """
    Abstract base class for aggregating results from multiple graders.

    This class defines the interface for combining scores or rankings from
    multiple graders into a single result for a single sample.
    """

    def __init__(self, name: str):
        """
        Initialize the aggregator.

        Args:
            name: Name of the aggregator
        """
        self.name = name

    @property
    def __name__(self):
        """
        Get the name of the aggregator.

        Returns:
            Name of the aggregator
        """
        return self.name

    @abstractmethod
    def __call__(self, grader_results: Dict[str, GraderResult], **kwargs) -> GraderResult:
        """
        Aggregate results from multiple graders for a single sample.

        Args:
            grader_results: Dictionary mapping grader names to GraderResult objects for a single sample
            **kwargs: Additional arguments for aggregation

        Returns:
            Aggregated result as a GraderResult object
        """
