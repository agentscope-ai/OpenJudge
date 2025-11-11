"""
Multimodal Metrics Schema Definitions

Define data models for multimodal evaluation metrics.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MLLMImage(BaseModel):
    """
    Multimodal LLM Image representation

    Supports both URL-based and base64-encoded images.

    Attributes:
        url: Image URL (HTTP/HTTPS)
        base64: Base64-encoded image data
        format: Image format (png, jpg, jpeg, etc.)

    Example:
        >>> # URL-based image
        >>> img1 = MLLMImage(url="https://example.com/image.jpg")
        >>>
        >>> # Base64-encoded image
        >>> img2 = MLLMImage(
        ...     base64="iVBORw0KGgoAAAANS...",
        ...     format="png"
        ... )
    """

    url: Optional[str] = Field(None, description="Image URL")
    base64: Optional[str] = Field(None, description="Base64-encoded image data")
    format: Optional[str] = Field(None, description="Image format (png, jpg, etc.)")

    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one of url or base64 is provided"""
        if not self.url and not self.base64:
            raise ValueError("Either 'url' or 'base64' must be provided")


class MLLMTestCaseParams(str, Enum):
    """
    Test case parameters for multimodal evaluation

    Defines which fields are required for different types of metrics.
    """

    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    RETRIEVAL_CONTEXT = "retrieval_context"
    CONTEXT = "context"
    TOOLS = "tools"
    EXPECTED_TOOLS = "expected_tools"


class MLLMTestCase(BaseModel):
    """
    Multimodal LLM Test Case

    Container for test data in multimodal evaluation scenarios.

    Attributes:
        input: Input data (can contain both text and images)
        actual_output: Actual model output
        expected_output: Expected output (for reference-based metrics)
        retrieval_context: Retrieved context (for RAG scenarios)
        context: Additional context information
        tools: Available tools (for tool-use metrics)
        expected_tools: Expected tool calls

    Example:
        >>> test_case = MLLMTestCase(
        ...     input=["Describe this image:", MLLMImage(url="...")],
        ...     actual_output=["This is a cat sitting on a mat."],
        ...     expected_output=["A cat on a mat."]
        ... )
    """

    input: List[Union[str, MLLMImage]] = Field(
        default_factory=list, description="Input data (text and images)"
    )
    actual_output: List[Union[str, MLLMImage]] = Field(
        default_factory=list, description="Actual model output"
    )
    expected_output: Optional[List[Union[str, MLLMImage]]] = Field(
        None, description="Expected output"
    )
    retrieval_context: Optional[List[Union[str, MLLMImage]]] = Field(
        None, description="Retrieved context for RAG"
    )
    context: Optional[List[Union[str, MLLMImage]]] = Field(
        None, description="Additional context"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Available tools")
    expected_tools: Optional[List[Dict[str, Any]]] = Field(
        None, description="Expected tool calls"
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ReasonScore(BaseModel):
    """
    Evaluation result with reasoning

    Used for metrics that provide both a score and reasoning.

    Attributes:
        score: Evaluation score (typically 0-10)
        reasoning: Explanation for the score

    Example:
        >>> result = ReasonScore(
        ...     score=8,
        ...     reasoning="The image matches the description well..."
        ... )
    """

    score: Union[int, float, List[int], List[float]] = Field(
        ..., description="Evaluation score (0-10 or list of scores)"
    )
    reasoning: str = Field(..., description="Reasoning for the score")


class MultimodalMetricResult(BaseModel):
    """
    Result of multimodal metric evaluation

    Extends the standard MetricResult with multimodal-specific fields.

    Attributes:
        name: Metric name
        score: Normalized score [0, 1]
        raw_score: Raw score before normalization
        reason: Reasoning for the score
        success: Whether the evaluation passed the threshold
        details: Additional details
        metadata: Metadata (model used, cost, etc.)

    Example:
        >>> result = MultimodalMetricResult(
        ...     name="image_coherence",
        ...     score=0.85,
        ...     raw_score=8.5,
        ...     reason="Image aligns well with context",
        ...     success=True
        ... )
    """

    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score [0, 1]")
    raw_score: Optional[float] = Field(None, description="Raw score")
    reason: Optional[str] = Field(None, description="Reasoning for the score")
    success: bool = Field(default=True, description="Whether evaluation passed")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata (model, cost, etc.)"
    )

    def __str__(self) -> str:
        return f"{self.name}: {self.score:.4f} (success={self.success})"

    def __repr__(self) -> str:
        return (
            f"MultimodalMetricResult(name='{self.name}', "
            f"score={self.score:.4f}, success={self.success})"
        )
