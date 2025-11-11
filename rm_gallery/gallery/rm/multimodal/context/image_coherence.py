"""
Image Coherence Metric

Evaluates the coherence between images and their surrounding text context.
"""

import asyncio
from typing import List, Optional, Tuple, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.metrics.multimodal.base import BaseMultimodalMetric
from rm_gallery.core.metrics.multimodal.schema import (
    MLLMImage,
    MLLMTestCase,
    ReasonScore,
)
from rm_gallery.core.metrics.multimodal.utils import construct_verbose_logs
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.context.templates import ImageCoherenceTemplate


class ImageCoherenceMetric(BaseMultimodalMetric):
    """
    Image Coherence Metric

    Evaluates how well images cohere with their surrounding textual context.
    Considers both the text appearing above and below each image.

    The metric extracts context from surrounding text and evaluates whether
    the image is coherent with that context. For multiple images, it computes
    the average coherence score.

    Attributes:
        name: Metric name
        model: Qwen VL API instance for evaluation
        threshold: Success threshold [0, 1]
        max_context_size: Maximum characters to extract from context (default: 500)
        async_mode: Whether to use async evaluation
        strict_mode: If True, scores below threshold become 0
        verbose_mode: Whether to generate detailed logs

    Example:
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMTestCase, MLLMImage
        >>>
        >>> api = QwenVLAPI(api_key="your-key", model_name="qwen-vl-plus")
        >>> metric = ImageCoherenceMetric(model=api, threshold=0.7)
        >>>
        >>> test_case = MLLMTestCase(
        ...     actual_output=[
        ...         "Our sales grew significantly in Q3.",
        ...         MLLMImage(url="https://example.com/sales_chart.jpg"),
        ...         "This was driven by new product launches."
        ...     ]
        ... )
        >>>
        >>> score = metric.measure(test_case)
        >>> print(f"Coherence score: {score:.2f}")
    """

    name: str = Field(default="Image Coherence", description="Metric name")
    model: QwenVLAPI = Field(..., description="Qwen VL API instance")
    max_context_size: Optional[int] = Field(
        default=500, description="Maximum context size in characters"
    )

    # Internal state for evaluation results
    contexts_above: List[Optional[str]] = Field(default_factory=list)
    contexts_below: List[Optional[str]] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.evaluation_model = self.model.model_name

    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """
        Measure image coherence for a test case

        Args:
            test_case: Test case with actual_output containing text and images
            _show_indicator: Whether to show progress indicator
            _in_component: Whether called within another component

        Returns:
            Average coherence score [0, 1]
        """
        if self.async_mode:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.a_measure(test_case, _show_indicator, _in_component),
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator, _in_component)
                )

        # Synchronous evaluation
        self._check_test_case_params(test_case)
        self.evaluation_cost = 0.0

        actual_output = test_case.actual_output
        self.contexts_above = []
        self.contexts_below = []
        self.scores = []
        self.reasons = []

        # Find all images in actual_output
        image_indices = self.get_image_indices(actual_output)

        if not image_indices:
            logger.warning("No images found in actual_output")
            self.score = 0.0
            self.reason = "No images to evaluate"
            self.success = False
            return 0.0

        # Evaluate each image
        for image_index in image_indices:
            context_above, context_below = self.get_image_context(
                image_index, actual_output
            )
            image = actual_output[image_index]

            raw_score, reason = self._evaluate_image_coherence(
                image, context_above, context_below
            )

            # Normalize score to [0, 1]
            normalized_score = raw_score / 10.0

            self.contexts_above.append(context_above)
            self.contexts_below.append(context_below)
            self.scores.append(normalized_score)
            self.reasons.append(reason)

        # Calculate final score (average of all images)
        self.score = sum(self.scores) / len(self.scores) if self.scores else 0.0

        # Apply strict mode
        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        # Generate combined reason
        self.reason = self._generate_combined_reason()
        self.success = self.score >= self.threshold

        # Generate verbose logs if needed
        if self.verbose_mode:
            self.verbose_logs = self._generate_verbose_logs()

        return self.score

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """
        Async version of measure

        Args:
            test_case: Test case with actual_output containing text and images
            _show_indicator: Whether to show progress indicator
            _in_component: Whether called within another component

        Returns:
            Average coherence score [0, 1]
        """
        self._check_test_case_params(test_case)
        self.evaluation_cost = 0.0

        actual_output = test_case.actual_output
        self.contexts_above = []
        self.contexts_below = []
        self.scores = []
        self.reasons = []

        # Find all images
        image_indices = self.get_image_indices(actual_output)

        if not image_indices:
            logger.warning("No images found in actual_output")
            self.score = 0.0
            self.reason = "No images to evaluate"
            self.success = False
            return 0.0

        # Prepare evaluation tasks
        tasks = []
        for image_index in image_indices:
            context_above, context_below = self.get_image_context(
                image_index, actual_output
            )
            image = actual_output[image_index]

            self.contexts_above.append(context_above)
            self.contexts_below.append(context_below)

            tasks.append(
                self._a_evaluate_image_coherence(image, context_above, context_below)
            )

        # Evaluate all images in parallel
        results = await asyncio.gather(*tasks)

        # Process results
        for raw_score, reason in results:
            normalized_score = raw_score / 10.0
            self.scores.append(normalized_score)
            self.reasons.append(reason)

        # Calculate final score
        self.score = sum(self.scores) / len(self.scores) if self.scores else 0.0

        # Apply strict mode
        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.reason = self._generate_combined_reason()
        self.success = self.score >= self.threshold

        if self.verbose_mode:
            self.verbose_logs = self._generate_verbose_logs()

        return self.score

    def _evaluate_image_coherence(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Synchronous evaluation of image coherence"""
        prompt = ImageCoherenceTemplate.evaluate_image_coherence(
            context_above or "", context_below or ""
        )

        try:
            result = self.model.generate(
                text=prompt,
                images=[image],
                response_format=ReasonScore,
            )

            # Track cost if available
            if hasattr(self.model, "last_request_cost"):
                self.evaluation_cost += self.model.last_request_cost

            return result.score, result.reasoning
        except Exception as e:
            logger.error(f"Error evaluating image coherence: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _a_evaluate_image_coherence(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Async evaluation of image coherence"""
        prompt = ImageCoherenceTemplate.evaluate_image_coherence(
            context_above or "", context_below or ""
        )

        try:
            result = await self.model.a_generate(
                text=prompt,
                images=[image],
                response_format=ReasonScore,
            )

            if hasattr(self.model, "last_request_cost"):
                self.evaluation_cost += self.model.last_request_cost

            return result.score, result.reasoning
        except Exception as e:
            logger.error(f"Error evaluating image coherence: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    def get_image_context(
        self, image_index: int, output_list: List[Union[str, MLLMImage]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract text context surrounding an image

        Args:
            image_index: Index of the image in output_list
            output_list: List containing text and images

        Returns:
            Tuple of (context_above, context_below)
        """
        context_above = None
        context_below = None

        # Find context above (last text before image)
        for i in range(image_index - 1, -1, -1):
            if isinstance(output_list[i], str):
                context_above = output_list[i]
                if self.max_context_size and len(context_above) > self.max_context_size:
                    context_above = context_above[-self.max_context_size :]
                break

        # Find context below (first text after image)
        for i in range(image_index + 1, len(output_list)):
            if isinstance(output_list[i], str):
                context_below = output_list[i]
                if self.max_context_size and len(context_below) > self.max_context_size:
                    context_below = context_below[: self.max_context_size]
                break

        return context_above, context_below

    def get_image_indices(self, output_list: List[Union[str, MLLMImage]]) -> List[int]:
        """
        Find indices of all images in output list

        Args:
            output_list: List containing text and images

        Returns:
            List of image indices
        """
        return [
            index
            for index, element in enumerate(output_list)
            if isinstance(element, MLLMImage)
        ]

    def _check_test_case_params(self, test_case: MLLMTestCase):
        """Validate test case has required parameters"""
        if not test_case.actual_output:
            raise ValueError("Test case must have actual_output")

    def _generate_combined_reason(self) -> str:
        """Generate combined reasoning for all images"""
        if len(self.reasons) == 1:
            return self.reasons[0]

        combined = []
        for i, reason in enumerate(self.reasons, 1):
            combined.append(f"Image {i} (score: {self.scores[i-1]:.2f}): {reason}")

        return "\n".join(combined)

    def _generate_verbose_logs(self) -> str:
        """Generate verbose evaluation logs"""
        steps = []

        for i in range(len(self.scores)):
            step = []
            if self.contexts_above[i]:
                step.append(
                    f"Context Above Image {i+1}: {self.contexts_above[i][:50]}..."
                )
            if self.contexts_below[i]:
                step.append(
                    f"Context Below Image {i+1}: {self.contexts_below[i][:50]}..."
                )
            step.append(f"Image {i+1} Score: {self.scores[i]:.2f}")
            step.append(f"Image {i+1} Reason: {self.reasons[i]}")
            steps.append("\n".join(step))

        if len(self.scores) > 1:
            steps.append(f"Average Score: {self.score:.2f}")

        return construct_verbose_logs(self, steps)

    def run(self, **kwargs) -> float:
        """
        Run method for BaseModule interface

        Args:
            **kwargs: Should contain 'test_case' key with MLLMTestCase value

        Returns:
            Coherence score [0, 1]
        """
        test_case = kwargs.get("test_case")
        if not test_case:
            raise ValueError("'test_case' must be provided in kwargs")

        return self.measure(test_case)
