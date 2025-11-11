"""
Image Editing Quality Metric

Evaluates the quality of AI-based image editing.
"""

import asyncio
import math
import textwrap
from typing import List, Optional, Tuple

from loguru import logger
from pydantic import Field

from rm_gallery.core.metrics.multimodal.base import BaseMultimodalMetric
from rm_gallery.core.metrics.multimodal.schema import (
    MLLMImage,
    MLLMTestCase,
    ReasonScore,
)
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.generation.templates import ImageEditingTemplate


class ImageEditingMetric(BaseMultimodalMetric):
    """
    Image Editing Quality Metric

    Evaluates AI-edited images based on:
    1. Semantic Consistency: How well the edits follow the instruction (0-10)
    2. Perceptual Quality: Visual quality of the edited image
       - Naturalness (0-10): How natural the edited areas look
       - Artifacts (0-10): Whether there are editing artifacts

    Final score = sqrt(semantic_consistency * min(perceptual_quality)) / 10

    This metric checks:
    - Whether the specified edits were correctly applied
    - Whether unmentioned areas remained unchanged
    - Whether edits blend naturally with the original image

    Attributes:
        vlm_api: Qwen VL API client for evaluation
        threshold: Success threshold [0, 1]
        strict_mode: If True, scores below threshold become 0

    Example:
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMTestCase, MLLMImage
        >>>
        >>> vlm_api = QwenVLAPI(api_key="...", model_name="qwen-vl-plus")
        >>> metric = ImageEditingMetric(vlm_api=vlm_api, threshold=0.7)
        >>>
        >>> test_case = MLLMTestCase(
        ...     input=[
        ...         MLLMImage(url="https://example.com/original.jpg"),
        ...         "Change the sofa color to blue"
        ...     ],
        ...     actual_output=[MLLMImage(url="https://example.com/edited.jpg")]
        ... )
        >>>
        >>> score = metric.measure(test_case)
        >>> print(f"Score: {score:.4f}")
    """

    name: str = "image_editing"
    vlm_api: QwenVLAPI = Field(..., description="Qwen VL API client for evaluation")

    # Evaluation results (populated after measure)
    SC_scores: Optional[List[float]] = Field(
        None, description="Semantic consistency scores"
    )
    SC_reasoning: Optional[str] = Field(
        None, description="Semantic consistency reasoning"
    )
    PQ_scores: Optional[List[float]] = Field(
        None, description="Perceptual quality scores"
    )
    PQ_reasoning: Optional[str] = Field(
        None, description="Perceptual quality reasoning"
    )

    def __init__(
        self,
        vlm_api: QwenVLAPI,
        threshold: float = 0.5,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize ImageEditingMetric

        Args:
            vlm_api: Qwen VL API client
            threshold: Success threshold (0-1)
            async_mode: Whether to use async evaluation
            strict_mode: If True, scores below threshold become 0
            verbose_mode: Whether to generate verbose logs
        """
        super().__init__(
            vlm_api=vlm_api,
            threshold=threshold,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            **kwargs,
        )
        self.evaluation_model = vlm_api.get_model_name()

    def run(self, **kwargs):
        """
        Run method required by BaseModule

        Delegates to measure() for backward compatibility.
        """
        test_case = kwargs.get("test_case")
        if not test_case:
            raise ValueError("test_case is required")
        return self.measure(test_case)

    def measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """
        Measure image editing quality score

        Args:
            test_case: Test case containing:
                - input: [original_image, edit_instruction] or [edit_instruction, original_image]
                - actual_output: [edited_image]
            _show_indicator: Whether to show progress indicator
            _in_component: Whether called within another component

        Returns:
            float: Quality score [0, 1]
        """
        self._check_test_case_params(test_case)

        if self.async_mode:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.a_measure(test_case, _show_indicator, _in_component)
            )
        else:
            return self._sync_measure(test_case)

    async def a_measure(
        self,
        test_case: MLLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        """
        Async version of measure

        Args:
            test_case: Test case to evaluate
            _show_indicator: Whether to show progress indicator
            _in_component: Whether called within another component

        Returns:
            float: Quality score [0, 1]
        """
        self._check_test_case_params(test_case)

        # Extract original image, edit instruction, and edited image
        input_texts, input_images = self.separate_images_from_text(test_case.input)
        _, output_images = self.separate_images_from_text(test_case.actual_output)

        original_image = input_images[0]
        edit_instruction = "\n".join(input_texts)
        edited_image = output_images[0]

        # Evaluate semantic consistency and perceptual quality in parallel
        (self.SC_scores, self.SC_reasoning), (
            self.PQ_scores,
            self.PQ_reasoning,
        ) = await asyncio.gather(
            self._a_evaluate_semantic_consistency(
                original_image, edit_instruction, edited_image
            ),
            self._a_evaluate_perceptual_quality(edited_image),
        )

        # Calculate final score
        self.score = self._calculate_score()

        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.reason = self._generate_reason()
        self.success = self.score >= self.threshold

        if self.verbose_mode:
            self.verbose_logs = self._generate_verbose_logs()

        return self.score

    def _sync_measure(self, test_case: MLLMTestCase) -> float:
        """
        Synchronous measurement implementation

        Args:
            test_case: Test case to evaluate

        Returns:
            float: Quality score [0, 1]
        """
        # Extract original image, edit instruction, and edited image
        input_texts, input_images = self.separate_images_from_text(test_case.input)
        _, output_images = self.separate_images_from_text(test_case.actual_output)

        original_image = input_images[0]
        edit_instruction = "\n".join(input_texts)
        edited_image = output_images[0]

        # Evaluate semantic consistency
        self.SC_scores, self.SC_reasoning = self._evaluate_semantic_consistency(
            original_image, edit_instruction, edited_image
        )

        # Evaluate perceptual quality
        self.PQ_scores, self.PQ_reasoning = self._evaluate_perceptual_quality(
            edited_image
        )

        # Calculate final score
        self.score = self._calculate_score()

        if self.strict_mode and self.score < self.threshold:
            self.score = 0.0

        self.reason = self._generate_reason()
        self.success = self.score >= self.threshold

        if self.verbose_mode:
            self.verbose_logs = self._generate_verbose_logs()

        return self.score

    async def _a_evaluate_semantic_consistency(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """
        Evaluate semantic consistency asynchronously

        Args:
            original_image: Original image before editing
            edit_instruction: Editing instruction
            edited_image: Edited image to evaluate

        Returns:
            Tuple of (scores_list, reasoning)
        """
        prompt = ImageEditingTemplate.generate_semantic_consistency_prompt(
            edit_instruction
        )

        try:
            result = await self.vlm_api.a_generate(
                text=prompt,
                images=[original_image, edited_image],
                response_format=ReasonScore,
            )

            # Cost is tracked automatically by vlm_api

            # Ensure score is a list
            scores = result.score if isinstance(result.score, list) else [result.score]
            return scores, result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    def _evaluate_semantic_consistency(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """
        Evaluate semantic consistency synchronously

        Args:
            original_image: Original image before editing
            edit_instruction: Editing instruction
            edited_image: Edited image to evaluate

        Returns:
            Tuple of (scores_list, reasoning)
        """
        prompt = ImageEditingTemplate.generate_semantic_consistency_prompt(
            edit_instruction
        )

        try:
            result = self.vlm_api.generate(
                text=prompt,
                images=[original_image, edited_image],
                response_format=ReasonScore,
            )

            # Cost is tracked automatically by vlm_api

            # Ensure score is a list
            scores = result.score if isinstance(result.score, list) else [result.score]
            return scores, result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _a_evaluate_perceptual_quality(
        self,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """
        Evaluate perceptual quality asynchronously

        Args:
            edited_image: Edited image to evaluate

        Returns:
            Tuple of ([naturalness_score, artifacts_score], reasoning)
        """
        prompt = ImageEditingTemplate.generate_perceptual_quality_prompt()

        try:
            result, cost = await self.vlm_api.a_generate(
                text=prompt, images=[edited_image], schema=ReasonScore
            )

            if self.evaluation_cost is None:
                self.evaluation_cost = 0.0
            self.evaluation_cost += cost

            # Ensure score is a list with 2 elements
            scores = (
                result.score
                if isinstance(result.score, list)
                else [result.score, result.score]
            )
            if len(scores) < 2:
                scores = [scores[0], scores[0]]

            return scores[:2], result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating perceptual quality: {e}")
            return [5.0, 5.0], f"Error during evaluation: {str(e)}"

    def _evaluate_perceptual_quality(
        self,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """
        Evaluate perceptual quality synchronously

        Args:
            edited_image: Edited image to evaluate

        Returns:
            Tuple of ([naturalness_score, artifacts_score], reasoning)
        """
        prompt = ImageEditingTemplate.generate_perceptual_quality_prompt()

        try:
            result, cost = self.vlm_api.generate(
                text=prompt, images=[edited_image], schema=ReasonScore
            )

            if self.evaluation_cost is None:
                self.evaluation_cost = 0.0
            self.evaluation_cost += cost

            # Ensure score is a list with 2 elements
            scores = (
                result.score
                if isinstance(result.score, list)
                else [result.score, result.score]
            )
            if len(scores) < 2:
                scores = [scores[0], scores[0]]

            return scores[:2], result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating perceptual quality: {e}")
            return [5.0, 5.0], f"Error during evaluation: {str(e)}"

    def _calculate_score(self) -> float:
        """
        Calculate final score using geometric mean

        Formula: sqrt(min(SC) * min(PQ)) / 10

        This ensures both semantic consistency and perceptual quality must be good.

        Returns:
            float: Final score [0, 1]
        """
        if not self.SC_scores or not self.PQ_scores:
            return 0.0

        min_sc = min(self.SC_scores)
        min_pq = min(self.PQ_scores)

        # Geometric mean normalized to [0, 1]
        score = math.sqrt(min_sc * min_pq) / 10.0

        return min(1.0, max(0.0, score))

    def _generate_reason(self) -> str:
        """
        Generate reasoning for the score

        Returns:
            str: Explanation of the score
        """
        if not self.SC_scores or not self.PQ_scores:
            return "Evaluation incomplete"

        return textwrap.dedent(
            f"""
            Overall score: {self.score:.4f}

            Semantic Consistency: {min(self.SC_scores)}/10
            Reasoning: {self.SC_reasoning}

            Perceptual Quality: {min(self.PQ_scores)}/10
            - Naturalness: {self.PQ_scores[0]}/10
            - Artifacts: {self.PQ_scores[1]}/10
            Reasoning: {self.PQ_reasoning}

            The score combines semantic consistency and perceptual quality using geometric mean,
            ensuring both aspects must be strong for a high overall score.
        """
        ).strip()

    def _generate_verbose_logs(self) -> str:
        """
        Generate verbose logs for debugging

        Returns:
            str: Detailed evaluation logs
        """
        logs = [
            "=" * 80,
            "IMAGE EDITING METRIC EVALUATION",
            "=" * 80,
            f"Model: {self.evaluation_model}",
            f"Cost: ${self.evaluation_cost:.6f}"
            if self.evaluation_cost
            else "Cost: N/A",
            "",
            "SEMANTIC CONSISTENCY EVALUATION:",
            f"  Scores: {self.SC_scores}",
            f"  Reasoning: {self.SC_reasoning}",
            "",
            "PERCEPTUAL QUALITY EVALUATION:",
            f"  Scores: {self.PQ_scores} (Naturalness, Artifacts)",
            f"  Reasoning: {self.PQ_reasoning}",
            "",
            "FINAL RESULTS:",
            f"  Score: {self.score:.4f}",
            f"  Success: {self.success}",
            f"  Threshold: {self.threshold}",
            "=" * 80,
        ]
        return "\n".join(logs)

    def _check_test_case_params(self, test_case: MLLMTestCase) -> None:
        """
        Validate test case has required parameters

        Args:
            test_case: Test case to validate

        Raises:
            ValueError: If required parameters are missing
        """
        if not test_case.input:
            raise ValueError(
                "test_case.input is required (should contain original image and edit instruction)"
            )

        if not test_case.actual_output:
            raise ValueError(
                "test_case.actual_output is required (should contain edited image)"
            )

        # Check that input contains both text (instruction) and image (original)
        input_texts, input_images = self.separate_images_from_text(test_case.input)
        if not input_texts:
            raise ValueError("test_case.input must contain edit instruction (text)")
        if not input_images:
            raise ValueError("test_case.input must contain original image")

        # Check that output contains edited image
        _, output_images = self.separate_images_from_text(test_case.actual_output)
        if not output_images:
            raise ValueError("test_case.actual_output must contain edited image")

    def is_successful(self) -> bool:
        """
        Check if evaluation passed the threshold

        Returns:
            bool: True if score >= threshold
        """
        if self.error is not None:
            return False
        if self.score is None:
            return False
        return self.score >= self.threshold
