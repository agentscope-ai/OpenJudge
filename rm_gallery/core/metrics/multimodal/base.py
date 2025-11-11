"""
Base Multimodal Metric Classes

Define base abstract classes for multimodal evaluation metrics.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.metrics.multimodal.schema import (
    MLLMImage,
    MLLMTestCase,
    MLLMTestCaseParams,
    MultimodalMetricResult,
)


class BaseMultimodalMetric(BaseModule, ABC):
    """
    Base class for multimodal evaluation metrics

    This class provides the foundation for all multimodal metrics, supporting:
    - Synchronous and asynchronous evaluation
    - Cost tracking for API calls
    - Threshold-based success criteria
    - Verbose logging
    - Strict mode for binary pass/fail evaluation

    Attributes:
        name: Metric name for identification
        threshold: Success threshold [0, 1]
        async_mode: Whether to use async evaluation
        strict_mode: If True, score < threshold becomes 0
        verbose_mode: Whether to generate detailed logs
        evaluation_model: Name of the model used for evaluation
        evaluation_cost: Estimated cost of evaluation (if tracked)

    Example:
        >>> class MyMultimodalMetric(BaseMultimodalMetric):
        ...     name: str = "my_metric"
        ...
        ...     def measure(self, test_case: MLLMTestCase) -> float:
        ...         # Implement evaluation logic
        ...         return 0.85
    """

    name: str = Field(..., description="Metric name")
    threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Success threshold"
    )
    async_mode: bool = Field(
        default=True, description="Whether to use async evaluation"
    )
    strict_mode: bool = Field(
        default=False, description="If True, scores below threshold become 0"
    )
    verbose_mode: bool = Field(
        default=False, description="Whether to generate verbose logs"
    )

    # Evaluation results (populated after measure)
    score: Optional[float] = Field(None, description="Latest evaluation score")
    reason: Optional[str] = Field(None, description="Reasoning for the score")
    success: Optional[bool] = Field(None, description="Whether evaluation passed")
    error: Optional[str] = Field(None, description="Error message if evaluation failed")
    verbose_logs: Optional[str] = Field(None, description="Verbose evaluation logs")

    # Model and cost tracking
    evaluation_model: Optional[str] = Field(
        None, description="Model used for evaluation"
    )
    evaluation_cost: Optional[float] = Field(
        None, description="Estimated evaluation cost"
    )

    @abstractmethod
    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """
        Measure the metric score for a test case

        This is the core method that all subclasses must implement.

        Args:
            test_case: Multimodal test case to evaluate
            _show_indicator: Whether to show progress indicator
            _in_component: Whether this is called within another component

        Returns:
            float: Normalized score [0, 1]

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement measure()")

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """
        Async version of measure

        Default implementation runs measure in thread pool.
        Subclasses should override for true async support.

        Args:
            test_case: Multimodal test case to evaluate
            _show_indicator: Whether to show progress indicator
            _in_component: Whether this is called within another component

        Returns:
            float: Normalized score [0, 1]
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.measure,
            test_case,
            _show_indicator,
            _in_component,
        )

    def separate_images_from_text(
        self, multimodal_list: List[Union[MLLMImage, str]]
    ) -> Tuple[List[str], List[MLLMImage]]:
        """
        Separate text and images from a multimodal list

        Args:
            multimodal_list: List containing both text and images

        Returns:
            Tuple of (text_list, image_list)

        Example:
            >>> texts, images = metric.separate_images_from_text([
            ...     "Describe this:",
            ...     MLLMImage(url="..."),
            ...     "in detail"
            ... ])
            >>> len(texts)  # 2
            >>> len(images)  # 1
        """
        images: List[MLLMImage] = []
        texts: List[str] = []

        for item in multimodal_list:
            if isinstance(item, MLLMImage):
                images.append(item)
            elif isinstance(item, str):
                texts.append(item)

        return texts, images

    def get_image_indices(
        self, multimodal_list: List[Union[str, MLLMImage]]
    ) -> List[int]:
        """
        Get indices of images in a multimodal list

        Args:
            multimodal_list: List containing both text and images

        Returns:
            List of indices where images appear

        Example:
            >>> indices = metric.get_image_indices([
            ...     "Text",
            ...     MLLMImage(url="..."),
            ...     "More text",
            ...     MLLMImage(url="...")
            ... ])
            >>> indices  # [1, 3]
        """
        return [
            index
            for index, element in enumerate(multimodal_list)
            if isinstance(element, MLLMImage)
        ]

    def check_test_case_params(
        self,
        test_case: MLLMTestCase,
        required_params: List[MLLMTestCaseParams],
        min_images_in_input: Optional[int] = None,
        min_images_in_output: Optional[int] = None,
    ) -> None:
        """
        Validate that test case has required parameters

        Args:
            test_case: Test case to validate
            required_params: List of required parameters
            min_images_in_input: Minimum number of images required in input
            min_images_in_output: Minimum number of images required in output

        Raises:
            ValueError: If validation fails
        """
        # Check required fields
        for param in required_params:
            value = getattr(test_case, param.value, None)
            if value is None or (isinstance(value, list) and len(value) == 0):
                raise ValueError(f"{self.name} requires '{param.value}' in test case")

        # Check minimum images in input
        if min_images_in_input is not None:
            _, input_images = self.separate_images_from_text(test_case.input)
            if len(input_images) < min_images_in_input:
                raise ValueError(
                    f"{self.name} requires at least {min_images_in_input} "
                    f"image(s) in input, got {len(input_images)}"
                )

        # Check minimum images in output
        if min_images_in_output is not None:
            _, output_images = self.separate_images_from_text(test_case.actual_output)
            if len(output_images) < min_images_in_output:
                raise ValueError(
                    f"{self.name} requires at least {min_images_in_output} "
                    f"image(s) in output, got {len(output_images)}"
                )

    def is_successful(self) -> bool:
        """
        Check if evaluation was successful

        Returns:
            bool: True if score >= threshold and no errors
        """
        if self.error is not None:
            return False

        try:
            return self.score is not None and self.score >= self.threshold
        except Exception as e:
            logger.error(f"Error checking success: {e}")
            return False

    def to_result(self) -> MultimodalMetricResult:
        """
        Convert to MultimodalMetricResult

        Returns:
            MultimodalMetricResult: Result object
        """
        return MultimodalMetricResult(
            name=self.name,
            score=self.score if self.score is not None else 0.0,
            raw_score=self.score * 10 if self.score is not None else None,
            reason=self.reason,
            success=self.is_successful(),
            metadata={
                "evaluation_model": self.evaluation_model,
                "evaluation_cost": self.evaluation_cost,
                "threshold": self.threshold,
                "strict_mode": self.strict_mode,
            },
        )

    def run(self, **kwargs) -> MultimodalMetricResult:
        """
        Implement BaseModule's run method

        Used for compatibility with RM-Gallery's BaseModule interface.

        Args:
            **kwargs: Must contain 'test_case' key

        Returns:
            MultimodalMetricResult: Evaluation result
        """
        if "test_case" not in kwargs:
            raise ValueError("Must provide 'test_case' parameter")

        test_case = kwargs["test_case"]
        if not isinstance(test_case, MLLMTestCase):
            raise ValueError("test_case must be an MLLMTestCase instance")

        self.measure(test_case)
        return self.to_result()

    @property
    def __name__(self) -> str:
        """Get metric name"""
        return self.name
