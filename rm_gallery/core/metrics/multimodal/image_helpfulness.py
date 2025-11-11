"""
Image Helpfulness Grader

Evaluates how helpful images are in understanding surrounding text.
Restructured to work with Grader framework.
"""

import asyncio
from typing import List, Optional, Tuple, Union

from loguru import logger

from rm_gallery.core.grader import Grader, GraderMode, GraderScore
from rm_gallery.core.metrics.multimodal.schema import MLLMImage, ReasonScore
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.context.templates import ImageHelpfulnessTemplate


class ImageHelpfulnessGrader(Grader):
    """
    Image Helpfulness Grader

    Evaluates how helpful images are in enabling readers to understand the
    surrounding text. Unlike coherence (which measures matching), helpfulness
    measures whether the image provides additional value for comprehension.

    Key evaluation dimensions:
    - Information enhancement: Does the image provide new information?
    - Understanding assistance: Does it make concepts clearer?
    - Practical value: Is it genuinely useful for the reader?

    Attributes:
        name: Grader name
        model: QwenVLAPI instance for evaluation
        max_context_size: Maximum characters to extract from context (default: 500)
        threshold: Success threshold [0, 1] (default: 0.7)

    Example:
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMImage
        >>>
        >>> api = QwenVLAPI(api_key="your-key", model_name="qwen-vl-plus")
        >>> grader = ImageHelpfulnessGrader(model=api, threshold=0.7)
        >>>
        >>> result = await grader.evaluate(
        ...     actual_output=[
        ...         "The architecture consists of multiple layers.",
        ...         MLLMImage(url="https://example.com/architecture_diagram.jpg"),
        ...         "Each layer serves a specific purpose."
        ...     ]
        ... )
        >>> print(f"Helpfulness score: {result.score:.2f}")
    """

    def __init__(
        self,
        model: QwenVLAPI,
        name: str = "image_helpfulness",
        max_context_size: int = 500,
        threshold: float = 0.7,
        description: str = "Evaluate image helpfulness for understanding text",
    ):
        super().__init__(
            name=name,
            grader_mode=GraderMode.POINTWISE,
            description=description,
        )
        self.model = model
        self.max_context_size = max_context_size
        self.threshold = threshold
        self.evaluation_cost = 0.0

    def _get_image_indices(self, output_list: List[Union[str, MLLMImage]]) -> List[int]:
        """Find indices of all images in output list"""
        return [
            index
            for index, element in enumerate(output_list)
            if isinstance(element, MLLMImage)
        ]

    def _get_image_context(
        self, image_index: int, output_list: List[Union[str, MLLMImage]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract text context surrounding an image"""
        context_above = None
        context_below = None

        # Find context above
        for i in range(image_index - 1, -1, -1):
            if isinstance(output_list[i], str):
                context_above = output_list[i]
                if self.max_context_size and len(context_above) > self.max_context_size:
                    context_above = context_above[-self.max_context_size :]
                break

        # Find context below
        for i in range(image_index + 1, len(output_list)):
            if isinstance(output_list[i], str):
                context_below = output_list[i]
                if self.max_context_size and len(context_below) > self.max_context_size:
                    context_below = context_below[: self.max_context_size]
                break

        return context_above, context_below

    def _evaluate_single_image(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Synchronous evaluation of single image helpfulness"""
        prompt = ImageHelpfulnessTemplate.evaluate_image_helpfulness(
            context_above or "", context_below or ""
        )

        try:
            result = self.model.generate(
                text=prompt,
                images=[image],
                response_format=ReasonScore,
            )

            if hasattr(self.model, "last_request_cost"):
                self.evaluation_cost += self.model.last_request_cost

            return float(result.score), result.reasoning
        except Exception as e:
            logger.error(f"Error evaluating image helpfulness: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    async def _a_evaluate_single_image(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Async evaluation of single image helpfulness"""
        prompt = ImageHelpfulnessTemplate.evaluate_image_helpfulness(
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

            return float(result.score), result.reasoning
        except Exception as e:
            logger.error(f"Error evaluating image helpfulness: {e}")
            return 0.0, f"Evaluation error: {str(e)}"

    def _compute(
        self, actual_output: List[Union[str, MLLMImage]], **kwargs
    ) -> Tuple[float, dict]:
        """Compute image helpfulness score (synchronous)"""
        self.evaluation_cost = 0.0

        image_indices = self._get_image_indices(actual_output)

        if not image_indices:
            return 0.0, {
                "error": "No images found in actual_output",
                "num_images": 0,
            }

        scores = []
        reasons = []

        for image_index in image_indices:
            context_above, context_below = self._get_image_context(
                image_index, actual_output
            )
            image = actual_output[image_index]

            raw_score, reason = self._evaluate_single_image(
                image, context_above, context_below
            )

            normalized_score = raw_score / 10.0
            scores.append(normalized_score)
            reasons.append(reason)

        final_score = sum(scores) / len(scores) if scores else 0.0

        details = {
            "num_images": len(image_indices),
            "individual_scores": scores,
            "individual_reasons": reasons,
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return final_score, details

    async def _a_compute(
        self, actual_output: List[Union[str, MLLMImage]], **kwargs
    ) -> Tuple[float, dict]:
        """Compute image helpfulness score (asynchronous)"""
        self.evaluation_cost = 0.0

        image_indices = self._get_image_indices(actual_output)

        if not image_indices:
            return 0.0, {
                "error": "No images found in actual_output",
                "num_images": 0,
            }

        tasks = []
        for image_index in image_indices:
            context_above, context_below = self._get_image_context(
                image_index, actual_output
            )
            image = actual_output[image_index]
            tasks.append(
                self._a_evaluate_single_image(image, context_above, context_below)
            )

        results = await asyncio.gather(*tasks)

        scores = []
        reasons = []
        for raw_score, reason in results:
            normalized_score = raw_score / 10.0
            scores.append(normalized_score)
            reasons.append(reason)

        final_score = sum(scores) / len(scores) if scores else 0.0

        details = {
            "num_images": len(image_indices),
            "individual_scores": scores,
            "individual_reasons": reasons,
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return final_score, details

    async def evaluate(
        self,
        actual_output: List[Union[str, MLLMImage]],
        async_mode: bool = True,
        **kwargs,
    ) -> GraderScore:
        """
        Evaluate image helpfulness

        Args:
            actual_output: List containing text and images (mixed)
            async_mode: Whether to use async evaluation
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized helpfulness value [0, 1]

        Example:
            >>> result = await grader.evaluate(
            ...     actual_output=[
            ...         "The system architecture:",
            ...         MLLMImage(url="diagram.jpg"),
            ...         "shows the component interactions"
            ...     ]
            ... )
        """
        if async_mode:
            score, details = await self._a_compute(actual_output, **kwargs)
        else:
            score, details = self._compute(actual_output, **kwargs)

        if "error" in details:
            return GraderScore(
                score=0.0,
                reason=details["error"],
                metadata=details,
            )

        # Generate combined reason
        if len(details["individual_reasons"]) == 1:
            reason = details["individual_reasons"][0]
        else:
            reason_parts = []
            for i, (s, r) in enumerate(
                zip(details["individual_scores"], details["individual_reasons"]), 1
            ):
                reason_parts.append(f"Image {i} (score: {s:.2f}): {r}")
            reason = "\n".join(reason_parts)

        return GraderScore(
            score=score,
            reason=f"Image helpfulness score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageHelpfulnessGrader"]
