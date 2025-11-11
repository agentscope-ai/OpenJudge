"""
Metrics Schema Definitions

Define data models for evaluation metrics, following RM-Gallery's Pydantic design patterns.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """
    Evaluation result for a single metric

    Attributes:
        name: Metric name (e.g., "bleu", "rouge", "fuzzy_match")
        score: Normalized score, range [0, 1], 1 means perfect match
        raw_score: Raw score (not normalized), some metrics may need this
        details: Details dictionary containing metric-specific extra data
        metadata: Metadata such as configuration parameters, runtime information, etc.

    Example:
        >>> result = MetricResult(
        ...     name="bleu",
        ...     score=0.85,
        ...     raw_score=85.0,
        ...     details={"precisions": [0.9, 0.8, 0.7, 0.6]},
        ...     metadata={"max_ngram": 4}
        ... )
    """

    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score [0, 1]")
    raw_score: Optional[float] = Field(None, description="Raw score (not normalized)")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Details information"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def __str__(self) -> str:
        return f"{self.name}: {self.score:.4f}"

    def __repr__(self) -> str:
        return f"MetricResult(name='{self.name}', score={self.score:.4f})"


class ComparisonInput(BaseModel):
    """
    Text comparison input data

    Attributes:
        reference: Reference text, can be a single string or list of multiple reference texts
        candidate: Candidate text (text to be evaluated)
        language: Language code (ISO 639-1), defaults to English
        normalize: Whether to normalize text
        context: Optional context information

    Example:
        >>> # Single reference text
        >>> input1 = ComparisonInput(
        ...     reference="The cat is on the mat",
        ...     candidate="A cat is on the mat"
        ... )
        >>>
        >>> # Multiple reference texts
        >>> input2 = ComparisonInput(
        ...     reference=["The cat sits", "A cat is sitting"],
        ...     candidate="The cat is sitting"
        ... )
    """

    reference: Union[str, List[str]] = Field(..., description="Reference text")
    candidate: str = Field(..., description="Candidate text")
    language: str = Field(default="en", description="Language code (ISO 639-1)")
    normalize: bool = Field(default=True, description="Whether to normalize text")
    context: Optional[str] = Field(None, description="Context information")

    def get_references(self) -> List[str]:
        """Get reference text list"""
        if isinstance(self.reference, str):
            return [self.reference]
        return self.reference

    def has_multiple_references(self) -> bool:
        """Whether there are multiple reference texts"""
        return isinstance(self.reference, list) and len(self.reference) > 1


class BatchComparisonInput(BaseModel):
    """
    Batch text comparison input data

    Attributes:
        references: List of reference texts
        candidates: List of candidate texts
        language: Language code
        normalize: Whether to normalize

    Example:
        >>> batch_input = BatchComparisonInput(
        ...     references=["ref1", "ref2", "ref3"],
        ...     candidates=["cand1", "cand2", "cand3"]
        ... )
    """

    references: List[Union[str, List[str]]] = Field(
        ..., description="List of reference texts"
    )
    candidates: List[str] = Field(..., description="List of candidate texts")
    language: str = Field(default="en", description="Language code")
    normalize: bool = Field(default=True, description="Whether to normalize text")

    def __len__(self) -> int:
        """Return batch size"""
        return len(self.candidates)

    def validate_length(self) -> bool:
        """Validate whether reference and candidate lengths match"""
        return len(self.references) == len(self.candidates)


class MetricConfig(BaseModel):
    """
    Metric configuration

    Attributes:
        name: Metric name
        enabled: Whether this metric is enabled
        weight: Metric weight (for weighted average)
        params: Metric-specific parameter configuration

    Example:
        >>> config = MetricConfig(
        ...     name="bleu",
        ...     enabled=True,
        ...     weight=1.5,
        ...     params={"max_ngram_order": 4, "smooth_method": "exp"}
        ... )
    """

    name: str = Field(..., description="Metric name")
    enabled: bool = Field(default=True, description="Whether enabled")
    weight: float = Field(default=1.0, ge=0.0, description="Weight")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameter configuration"
    )


class AggregatedMetricResult(BaseModel):
    """
    Aggregated metric results (for batch evaluation)

    Attributes:
        metric_name: Metric name
        individual_results: List of results for each sample
        mean_score: Mean score
        std_score: Score standard deviation
        min_score: Minimum score
        max_score: Maximum score
        count: Number of samples

    Example:
        >>> agg_result = AggregatedMetricResult(
        ...     metric_name="bleu",
        ...     individual_results=[0.8, 0.9, 0.7],
        ...     mean_score=0.8,
        ...     std_score=0.1,
        ...     min_score=0.7,
        ...     max_score=0.9,
        ...     count=3
        ... )
    """

    metric_name: str = Field(..., description="Metric name")
    individual_results: List[float] = Field(..., description="Results for each sample")
    mean_score: float = Field(..., description="Mean score")
    std_score: float = Field(..., description="Standard deviation")
    min_score: float = Field(..., description="Minimum score")
    max_score: float = Field(..., description="Maximum score")
    count: int = Field(..., description="Number of samples")

    @classmethod
    def from_results(
        cls, metric_name: str, results: List[MetricResult]
    ) -> "AggregatedMetricResult":
        """Create aggregated result from result list"""
        import numpy as np

        scores = [r.score for r in results]
        return cls(
            metric_name=metric_name,
            individual_results=scores,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            count=len(scores),
        )
