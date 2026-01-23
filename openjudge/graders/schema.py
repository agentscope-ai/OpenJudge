# -*- coding: utf-8 -*-
"""
Schemas for grading tasks.

This module defines the data schemas used in grading tasks, including grader modes,
result structures, and error handling.
"""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class GraderMode(str, Enum):
    """Grader modes for grader functions.

    This enum defines the two primary modes that graders can operate in:
    pointwise (evaluating individual samples) and listwise (ranking multiple samples).

    Attributes:
        POINTWISE: Pointwise grader mode.
        LISTWISE: Listwise grader mode.

    Example:
        >>> mode = GraderMode.POINTWISE
        >>> print(mode.value)
        pointwise
        >>>
        >>> mode = GraderMode.LISTWISE
        >>> print(mode.value)
        listwise
    """

    POINTWISE = "pointwise"
    LISTWISE = "listwise"


class GraderResult(BaseModel):
    """Base class for grader results.

    This Pydantic model defines the structure for grader results,
    which include a reason and optional metadata.

    Attributes:
        name (str): The name of the grader.
        reason (str): The reason for the result.
        metadata (Dict[str, Any]): The metadata of the grader result.

    Example:
        >>> result = GraderResult(
        ...     name="test_grader",
        ...     reason="Test evaluation completed",
        ...     metadata={"duration": 0.1}
        ... )
        >>> print(result.name)
        test_grader
        >>> result.model_dump()
        {'name': 'test_grader', 'reason': 'Test evaluation completed', 'metadata': {'duration': 0.1}}
    """

    name: str = Field(description="The name of the grader")
    reason: str = Field(default="", description="The reason for the result")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderScore(GraderResult):
    """Grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        score (float): A numerical score assigned by the grader.
        reason (str): Explanation of how the score was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> score_result = GraderScore(
        ...     name="accuracy_grader",
        ...     score=0.85,
        ...     reason="Answer is mostly accurate",
        ...     metadata={"confidence": 0.9}
        ... )
        >>> print(score_result.score)
        0.85
    """

    score: float = Field(description="score")
    reason: str = Field(description="reason")


class GraderScoreCallback(BaseModel):
    """Callback for grader score result.

    Represents a numerical score assigned by a grader along with a reason.

    Attributes:
        score (float): A numerical score assigned by the grader.
        reason (str): Explanation of how the score was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> callback = GraderScoreCallback(
        ...     score=0.9,
        ...     reason="High confidence in evaluation",
        ...     metadata={"model_used": ""}
        ... )
        >>> print(callback.score)
        0.9
    """

    score: float = Field(description="score")
    reason: str = Field(description="reason")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class RankValidation:
    """This class provides a field validator that can be inherited or mixed into Pydantic models
    to validate rank-related fields. The validator ensures that rank lists represent proper
    rankings according to standard ranking conventions:

    - Ranks must be positive integers starting from 1
    - All ranks in the list must be unique (no ties/duplicates)
    - The list must contain exactly all integers from 1 to n (where n is the list length)
    - Empty rank lists are not allowed

    This validation is particularly useful for listwise evaluation scenarios where
    LLMs or other systems output rankings of items, ensuring the output conforms to
    expected ranking formats before further processing.

    The validation will automatically be applied to the "rank" field during model
    instantiation and validation.
    """

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, rank: List[int]) -> List[int]:
        """Validate that the rank list is a valid permutation of consecutive positive integers starting from 1.

        This validator ensures that the rank list meets all requirements for a proper ranking:
        - Cannot be empty
        - Contains only positive integers (≥ 1)
        - All values are unique (no duplicates)
        - Forms a complete permutation of integers from 1 to n (where n is the list length)

        Args:
            rank: A list of integers representing ranks to be validated.

        Returns:
            The validated rank list unchanged if all validation checks pass.

        Raises:
            ValueError: If any of the following conditions are violated:
                - The rank list is empty
                - Any rank value is not a positive integer (≤ 0)
                - The rank list contains duplicate values
                - The rank list is not a complete permutation of [1, 2, ..., n]

        Examples:
            >>> validate_rank([1, 2, 3])  # Valid - sequential ranks
            [1, 2, 3]
            >>> validate_rank([3, 1, 2])  # Valid - permuted ranks
            [3, 1, 2]
            >>> validate_rank([1, 1, 2])  # Invalid - duplicates
            ValueError: Ranks should be unique
            >>> validate_rank([1, 3])     # Invalid - missing rank 2
            ValueError: Ranks should be a permutation of [1, 2]
            >>> validate_rank([0, 1])     # Invalid - contains zero
            ValueError: All ranks should be positive integers
        """
        if not rank:
            raise ValueError("Rank list cannot be empty")
        if any(x <= 0 for x in rank):
            raise ValueError("All ranks should be positive integers")
        if len(rank) != len(set(rank)):
            raise ValueError("Ranks should be unique")
        expected = set(range(1, len(rank) + 1))
        if set(rank) != expected:
            raise ValueError(f"Ranks should be a permutation of {sorted(expected)}")
        return rank


class GraderRank(GraderResult, RankValidation):
    """Grader rank result.

    Represents a ranking of items assigned by a grader along with a reason.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> rank_result = GraderRank(
        ...     name="relevance_ranker",
        ...     rank=[1, 3, 2],
        ...     reason="First response is most relevant",
        ...     metadata={"criteria": "relevance"}
        ... )
        >>> print(rank_result.rank)
        [1, 3, 2]
    """

    rank: List[int] = Field(description="rank")
    reason: str = Field(description="reason")


class GraderRankCallback(BaseModel, RankValidation):
    """Callback schema for LLM structured output in listwise grading.

    Used as the structured_model parameter in LLMGrader for LISTWISE mode.
    The LLM returns this schema which is then converted to GraderRank.

    Attributes:
        rank (List[int]): The ranking of items.
        reason (str): Explanation of how the ranking was determined.
        metadata (Dict[str, Any]): Optional additional information from the evaluation.

    Example:
        >>> callback = GraderRankCallback(
        ...     rank=[2, 1],
        ...     reason="Second response is more relevant",
        ...     metadata={"criteria": "clarity"}
        ... )
        >>> print(callback.rank)
        [2, 1]
    """

    rank: List[int] = Field(description="rank")
    reason: str = Field(description="reason")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="The metadata of the grader result",
    )


class GraderError(GraderResult):
    """Grader error result.

    Represents an error encountered during evaluation.

    Attributes:
        error (str): The error message.
        reason (str): Description of the error encountered during evaluation.
        metadata (Dict[str, Any]): Optional additional error information.

    Example:
        >>> error_result = GraderError(
        ...     name="test_grader",
        ...     error="Timeout occurred",
        ...     reason="Model took too long to respond",
        ...     metadata={"timeout_seconds": 30}
        ... )
        >>> print(error_result.error)
        Timeout occurred
    """

    error: str = Field(description="error")
