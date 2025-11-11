"""
Image Editing Quality Grader

Evaluates the quality of AI-based image editing.
Restructured to work with Grader framework.
"""

import asyncio
import math
from typing import List, Tuple, Union

from loguru import logger

from rm_gallery.core.grader import Grader, GraderMode, GraderScore
from rm_gallery.core.metrics.multimodal.schema import MLLMImage, ReasonScore
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.generation.templates import ImageEditingTemplate


class ImageEditingGrader(Grader):
    """
    Image Editing Quality Grader

    Evaluates AI-edited images based on:
    1. Semantic Consistency (SC): How well the edits follow the instruction (0-10)
    2. Perceptual Quality (PQ): Visual quality of the edited image
       - Naturalness (0-10): How natural the edited areas look
       - Artifacts (0-10): Whether there are editing artifacts

    Final score = sqrt(semantic_consistency * min(perceptual_quality)) / 10

    This metric checks:
    - Whether the specified edits were correctly applied
    - Whether unmentioned areas remained unchanged
    - Whether edits blend naturally with the original image

    Attributes:
        name: Grader name
        model: QwenVLAPI instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.5)

    Example:
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMImage
        >>>
        >>> vlm_api = QwenVLAPI(api_key="...", model_name="qwen-vl-plus")
        >>> grader = ImageEditingGrader(model=vlm_api, threshold=0.7)
        >>>
        >>> result = await grader.evaluate(
        ...     original_image=MLLMImage(url="https://example.com/original.jpg"),
        ...     edit_instruction="Change the sofa color to blue",
        ...     edited_image=MLLMImage(url="https://example.com/edited.jpg")
        ... )
        >>> print(f"Score: {result.score:.4f}")
    """

    def __init__(
        self,
        model: QwenVLAPI,
        name: str = "image_editing",
        threshold: float = 0.5,
        description: str = "Evaluate image editing quality",
    ):
        super().__init__(
            name=name,
            grader_mode=GraderMode.POINTWISE,
            description=description,
        )
        self.model = model
        self.threshold = threshold
        self.evaluation_cost = 0.0

    def _evaluate_semantic_consistency(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency synchronously"""
        prompt = ImageEditingTemplate.generate_semantic_consistency_prompt(
            edit_instruction
        )

        try:
            result = self.model.generate(
                text=prompt,
                images=[original_image, edited_image],
                response_format=ReasonScore,
            )

            scores = result.score if isinstance(result.score, list) else [result.score]
            return scores, result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _a_evaluate_semantic_consistency(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        prompt = ImageEditingTemplate.generate_semantic_consistency_prompt(
            edit_instruction
        )

        try:
            result = await self.model.a_generate(
                text=prompt,
                images=[original_image, edited_image],
                response_format=ReasonScore,
            )

            scores = result.score if isinstance(result.score, list) else [result.score]
            return scores, result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    def _evaluate_perceptual_quality(
        self,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality synchronously"""
        prompt = ImageEditingTemplate.generate_perceptual_quality_prompt()

        try:
            result = self.model.generate(
                text=prompt, images=[edited_image], response_format=ReasonScore
            )

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

    async def _a_evaluate_perceptual_quality(
        self,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        prompt = ImageEditingTemplate.generate_perceptual_quality_prompt()

        try:
            result = await self.model.a_generate(
                text=prompt, images=[edited_image], response_format=ReasonScore
            )

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

    def _compute(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
        **kwargs,
    ) -> Tuple[float, dict]:
        """
        Compute image editing quality score (synchronous)

        Args:
            original_image: Original image before editing
            edit_instruction: Editing instruction
            edited_image: Edited image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Evaluate semantic consistency
        sc_scores, sc_reasoning = self._evaluate_semantic_consistency(
            original_image, edit_instruction, edited_image
        )

        # Evaluate perceptual quality
        pq_scores, pq_reasoning = self._evaluate_perceptual_quality(edited_image)

        # Calculate final score using geometric mean
        if not sc_scores or not pq_scores:
            final_score = 0.0
        else:
            min_sc = min(sc_scores)
            min_pq = min(pq_scores)
            final_score = math.sqrt(min_sc * min_pq) / 10.0
            final_score = min(1.0, max(0.0, final_score))

        details = {
            "semantic_consistency_scores": sc_scores,
            "semantic_consistency_reasoning": sc_reasoning,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reasoning": pq_reasoning,
            "min_sc": min(sc_scores) if sc_scores else 0.0,
            "min_pq": min(pq_scores) if pq_scores else 0.0,
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return final_score, details

    async def _a_compute(
        self,
        original_image: MLLMImage,
        edit_instruction: str,
        edited_image: MLLMImage,
        **kwargs,
    ) -> Tuple[float, dict]:
        """
        Compute image editing quality score (asynchronous)

        Args:
            original_image: Original image before editing
            edit_instruction: Editing instruction
            edited_image: Edited image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reasoning), (pq_scores, pq_reasoning) = await asyncio.gather(
            self._a_evaluate_semantic_consistency(
                original_image, edit_instruction, edited_image
            ),
            self._a_evaluate_perceptual_quality(edited_image),
        )

        # Calculate final score using geometric mean
        if not sc_scores or not pq_scores:
            final_score = 0.0
        else:
            min_sc = min(sc_scores)
            min_pq = min(pq_scores)
            final_score = math.sqrt(min_sc * min_pq) / 10.0
            final_score = min(1.0, max(0.0, final_score))

        details = {
            "semantic_consistency_scores": sc_scores,
            "semantic_consistency_reasoning": sc_reasoning,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reasoning": pq_reasoning,
            "min_sc": min(sc_scores) if sc_scores else 0.0,
            "min_pq": min(pq_scores) if pq_scores else 0.0,
            "evaluation_cost": self.evaluation_cost,
            "threshold": self.threshold,
        }

        return final_score, details

    async def evaluate(
        self,
        original_image: Union[MLLMImage, List[MLLMImage]],
        edit_instruction: str,
        edited_image: Union[MLLMImage, List[MLLMImage]],
        async_mode: bool = True,
        **kwargs,
    ) -> GraderScore:
        """
        Evaluate image editing quality

        Args:
            original_image: Original image before editing (MLLMImage or list)
            edit_instruction: Editing instruction (string)
            edited_image: Edited image to evaluate (MLLMImage or list)
            async_mode: Whether to use async evaluation
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized quality value [0, 1]

        Example:
            >>> result = await grader.evaluate(
            ...     original_image=MLLMImage(url="original.jpg"),
            ...     edit_instruction="Change the sofa color to blue",
            ...     edited_image=MLLMImage(url="edited.jpg")
            ... )
        """
        # Handle if images are lists
        if isinstance(original_image, list):
            if not original_image:
                return GraderScore(
                    score=0.0,
                    reason="No original image provided",
                    metadata={"error": "Empty original image list"},
                )
            original_image = original_image[0]

        if isinstance(edited_image, list):
            if not edited_image:
                return GraderScore(
                    score=0.0,
                    reason="No edited image provided",
                    metadata={"error": "Empty edited image list"},
                )
            edited_image = edited_image[0]

        if not isinstance(original_image, MLLMImage) or not isinstance(
            edited_image, MLLMImage
        ):
            return GraderScore(
                score=0.0,
                reason="Invalid image type",
                metadata={"error": "Images must be MLLMImage"},
            )

        if async_mode:
            score, details = await self._a_compute(
                original_image, edit_instruction, edited_image, **kwargs
            )
        else:
            score, details = self._compute(
                original_image, edit_instruction, edited_image, **kwargs
            )

        # Generate comprehensive reason
        reason = f"""Image Editing Quality Score: {score:.4f}

Semantic Consistency: {details['min_sc']:.2f}/10
{details['semantic_consistency_reasoning']}

Perceptual Quality: {details['min_pq']:.2f}/10
- Naturalness: {details['perceptual_quality_scores'][0]:.2f}/10
- Artifacts: {details['perceptual_quality_scores'][1]:.2f}/10
{details['perceptual_quality_reasoning']}

The score combines semantic consistency and perceptual quality using geometric mean.
"""

        return GraderScore(
            score=score,
            reason=reason.strip(),
            metadata=details,
        )


__all__ = ["ImageEditingGrader"]
