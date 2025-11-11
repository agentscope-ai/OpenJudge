"""
Text-to-Image Quality Grader

Evaluates the quality of AI-generated images based on text prompts.
Restructured to work with Grader framework.
"""

import asyncio
import math
from typing import List, Tuple, Union

from loguru import logger

from rm_gallery.core.grader import Grader, GraderMode, GraderScore
from rm_gallery.core.metrics.multimodal.schema import MLLMImage, ReasonScore
from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
from rm_gallery.gallery.rm.multimodal.generation.templates import TextToImageTemplate


class TextToImageGrader(Grader):
    """
    Text-to-Image Quality Grader

    Evaluates AI-generated images based on:
    1. Semantic Consistency (SC): How well the image matches the text prompt (0-10)
    2. Perceptual Quality (PQ): Visual quality of the image
       - Naturalness (0-10)
       - Artifacts (0-10)

    Final score = sqrt(semantic_consistency * min(perceptual_quality)) / 10

    Attributes:
        name: Grader name
        model: QwenVLAPI instance for evaluation
        threshold: Success threshold [0, 1] (default: 0.5)

    Example:
        >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
        >>> from rm_gallery.core.metrics.multimodal import MLLMImage
        >>>
        >>> vlm_api = QwenVLAPI(api_key="...", model_name="qwen-vl-plus")
        >>> grader = TextToImageGrader(model=vlm_api, threshold=0.7)
        >>>
        >>> result = await grader.evaluate(
        ...     text_prompt="A cat sitting on a blue sofa",
        ...     generated_image=MLLMImage(url="https://example.com/cat.jpg")
        ... )
        >>> print(f"Score: {result.score:.4f}")
    """

    def __init__(
        self,
        model: QwenVLAPI,
        name: str = "text_to_image",
        threshold: float = 0.5,
        description: str = "Evaluate text-to-image generation quality",
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
        text_prompt: str,
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency synchronously"""
        prompt = TextToImageTemplate.generate_semantic_consistency_prompt(text_prompt)

        try:
            result = self.model.generate(
                text=prompt, images=[generated_image], response_format=ReasonScore
            )

            # Ensure score is a list
            scores = result.score if isinstance(result.score, list) else [result.score]
            return scores, result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _a_evaluate_semantic_consistency(
        self,
        text_prompt: str,
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        prompt = TextToImageTemplate.generate_semantic_consistency_prompt(text_prompt)

        try:
            result = await self.model.a_generate(
                text=prompt, images=[generated_image], response_format=ReasonScore
            )

            # Ensure score is a list
            scores = result.score if isinstance(result.score, list) else [result.score]
            return scores, result.reasoning

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    def _evaluate_perceptual_quality(
        self,
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality synchronously"""
        prompt = TextToImageTemplate.generate_perceptual_quality_prompt()

        try:
            result = self.model.generate(
                text=prompt, images=[generated_image], response_format=ReasonScore
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
        generated_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        prompt = TextToImageTemplate.generate_perceptual_quality_prompt()

        try:
            result = await self.model.a_generate(
                text=prompt, images=[generated_image], response_format=ReasonScore
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
        text_prompt: str,
        generated_image: MLLMImage,
        **kwargs,
    ) -> Tuple[float, dict]:
        """
        Compute text-to-image quality score (synchronous)

        Args:
            text_prompt: Original text prompt
            generated_image: Generated image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Evaluate semantic consistency
        sc_scores, sc_reasoning = self._evaluate_semantic_consistency(
            text_prompt, generated_image
        )

        # Evaluate perceptual quality
        pq_scores, pq_reasoning = self._evaluate_perceptual_quality(generated_image)

        # Calculate final score using geometric mean
        # Formula: sqrt(min(SC) * min(PQ)) / 10
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
        text_prompt: str,
        generated_image: MLLMImage,
        **kwargs,
    ) -> Tuple[float, dict]:
        """
        Compute text-to-image quality score (asynchronous)

        Args:
            text_prompt: Original text prompt
            generated_image: Generated image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """
        self.evaluation_cost = 0.0

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reasoning), (pq_scores, pq_reasoning) = await asyncio.gather(
            self._a_evaluate_semantic_consistency(text_prompt, generated_image),
            self._a_evaluate_perceptual_quality(generated_image),
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
        text_prompt: str,
        generated_image: Union[MLLMImage, List[MLLMImage]],
        async_mode: bool = True,
        **kwargs,
    ) -> GraderScore:
        """
        Evaluate text-to-image generation quality

        Args:
            text_prompt: Original text prompt (string)
            generated_image: Generated image (MLLMImage or list with single MLLMImage)
            async_mode: Whether to use async evaluation
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized quality value [0, 1]

        Example:
            >>> result = await grader.evaluate(
            ...     text_prompt="A cat sitting on a blue sofa",
            ...     generated_image=MLLMImage(url="cat.jpg")
            ... )
        """
        # Handle if generated_image is a list
        if isinstance(generated_image, list):
            if not generated_image:
                return GraderScore(
                    score=0.0,
                    reason="No generated image provided",
                    metadata={"error": "Empty image list"},
                )
            generated_image = generated_image[0]

        if not isinstance(generated_image, MLLMImage):
            return GraderScore(
                score=0.0,
                reason="Invalid image type",
                metadata={"error": "generated_image must be MLLMImage"},
            )

        if async_mode:
            score, details = await self._a_compute(
                text_prompt, generated_image, **kwargs
            )
        else:
            score, details = self._compute(text_prompt, generated_image, **kwargs)

        # Generate comprehensive reason
        reason = f"""Text-to-Image Quality Score: {score:.4f}

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


__all__ = ["TextToImageGrader"]
