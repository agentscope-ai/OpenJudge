# -*- coding: utf-8 -*-
"""
Text-to-Image Quality Grader

Evaluates the quality of AI-generated images based on text prompts.
Restructured to work with Grader framework.
"""

import asyncio
import math
import textwrap
from typing import Any, List, Tuple, Union

from loguru import logger

from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
from openjudge.graders.multimodal._internal import MLLMImage, format_image_content
from openjudge.graders.schema import GraderScoreCallback
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompts
TEXT_TO_IMAGE_SEMANTIC_PROMPT_EN = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{{
    "score" : [...],
    "reason" : "..."
}}

RULES:

The image is an AI-generated image according to the text prompt.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on the success in following the prompt.
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: {query}
"""

TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_EN = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{{
    "score" : [...],
    "reason" : "..."
}}

RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on image naturalness.
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting.
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts.
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized.
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

# Chinese Prompts
TEXT_TO_IMAGE_SEMANTIC_PROMPT_ZH = """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的有效性。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私机密问题。

你需要按以下方式给出输出（推理请保持简洁）：
{{
    "score" : [...],
    "reason" : "..."
}}

规则：

该图像是根据文本提示生成的AI图像。
目标是评估图像生成的成功程度。

从0到10的范围：
将根据遵循提示的成功程度给出0到10的分数。
（0表示AI生成的图像完全不遵循提示。10表示AI生成的图像完美地遵循提示。）

将分数放在列表中，输出分数 = [score]。

文本提示：{query}
"""

TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_ZH = """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的有效性。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私机密问题。

你需要按以下方式给出输出（推理请保持简洁）：
{{
    "score" : [...],
    "reason" : "..."
}}

规则：

该图像是AI生成的图像。
目标是评估图像生成的成功程度。

从0到10的范围：
将根据图像的自然度给出0到10的分数。
（
    0表示图像中的场景看起来完全不自然，或给人不自然的感觉，例如距离感错误、阴影错误或光照错误。
    10表示图像看起来自然。
）
第二个分数从0到10，将评估图像伪影。
（
    0表示图像包含大量失真、水印、划痕、模糊的面部、不寻常的身体部位或不协调的主体。
    10表示图像没有伪影。
）
将分数放在列表中，输出分数 = [自然度, 伪影]
"""

# Build default templates
DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_SEMANTIC_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_SEMANTIC_PROMPT_ZH),
            ),
        ],
    },
)

DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_ZH),
            ),
        ],
    },
)


class TextToImageGrader(BaseGrader):
    """
    Text-to-Image Quality Grader

    Purpose:
        Evaluates AI-generated images from text prompts by measuring semantic
        consistency (prompt following) and perceptual quality (visual realism).
        Essential for text-to-image model evaluation and benchmarking.

    What it evaluates:
        - Semantic Consistency: Image accurately reflects prompt description
        - Element Presence: All requested elements are included
        - Visual Quality: Image looks natural and realistic
        - Artifact Detection: No distortions, blur, or unnatural features
        - Composition: Proper spatial arrangement and aesthetics
        - Detail Fidelity: Specific details match prompt requirements

    When to use:
        - Text-to-image model benchmarking (DALL-E, Stable Diffusion, etc.)
        - Prompt engineering effectiveness evaluation
        - Generative model quality control
        - A/B testing different generation parameters
        - Research on text-to-image alignment

    Scoring:
        Formula: sqrt(semantic_consistency * min(perceptual_quality)) / 10
        - Semantic: 0-10 for prompt alignment
        - Perceptual: 0-10 for naturalness + 0-10 for artifact absence
        - Final: [0, 1] normalized score

    Args:
        model: Vision-language model instance or dict config
        threshold: Minimum score [0, 1] to pass (default: 0.5)
        semantic_template: PromptTemplate for semantic evaluation
        perceptual_template: PromptTemplate for perceptual evaluation
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore with combined quality score [0, 1]

    Example:
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.multimodal import TextToImageGrader, MLLMImage
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = TextToImageGrader(model=model)
        >>>
        >>> result = await grader.aevaluate(
        ...     query="A fluffy orange cat sitting on a blue sofa",
        ...     response=MLLMImage(url="https://example.com/generated.jpg")
        ... )
        >>> print(result.score)  # 0.92 - excellent prompt following and quality
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.5,
        semantic_template: PromptTemplate = DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE,
        perceptual_template: PromptTemplate = DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize TextToImageGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.5)
            semantic_template: PromptTemplate for semantic consistency evaluation (default: DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE)
            perceptual_template: PromptTemplate for perceptual quality evaluation (default: DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="text_to_image",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate text-to-image generation quality",
        )
        self.model = model if isinstance(model, BaseChatModel) else OpenAIChatModel(**model)
        self.threshold = threshold
        self.semantic_template = semantic_template
        self.perceptual_template = perceptual_template
        self.language = language

    async def _aevaluate_semantic_consistency(
        self,
        query: str,
        response: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        messages = self.semantic_template.to_messages(self.language)
        prompt = messages[0].format(query=query).content

        try:
            content = format_image_content(prompt, [response])
            chat_response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=GraderScoreCallback,
            )

            # Handle both streaming and non-streaming responses
            if hasattr(chat_response, "__aiter__"):
                # This is a streaming response, we need to collect it first
                collected_content = []
                parsed = {}
                async for chunk in chat_response:
                    if chunk.content:
                        collected_content.extend(chunk.content)
                    if chunk.parsed:
                        parsed.update(chunk.parsed)

                # Extract score and reason from metadata
                score = parsed.get("score", 0.0)
                reason = parsed.get("reason", "")
            else:
                # Non-streaming response
                score = chat_response.parsed["score"]
                score = score if isinstance(score, list) else [score]
                reason = chat_response.parsed["reason"]
            return score, reason

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _aevaluate_perceptual_quality(
        self,
        response: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        messages = self.perceptual_template.to_messages(self.language)
        prompt = messages[0].content

        try:
            content = format_image_content(prompt, [response])
            chat_response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=GraderScoreCallback,
            )
            score = chat_response.parsed["score"]
            score = score[:2] if isinstance(score, list) else [score, score]
            reason = chat_response.parsed["reason"]
            return score, reason

        except Exception as e:
            logger.error(f"Error evaluating perceptual quality: {e}")
            return [5.0, 5.0], f"Error during evaluation: {str(e)}"

    async def _a_compute(
        self,
        query: str,
        response: MLLMImage,
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """
        Compute text-to-image quality score (asynchronous)

        Args:
            query: Original text prompt
            response: Generated image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reason), (
            pq_scores,
            pq_reason,
        ) = await asyncio.gather(
            self._aevaluate_semantic_consistency(
                query,
                response,
            ),
            self._aevaluate_perceptual_quality(response),
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
            "semantic_consistency_reason": sc_reason,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reason": pq_reason,
            "min_sc": min(sc_scores) if sc_scores else 0.0,
            "min_pq": min(pq_scores) if pq_scores else 0.0,
            "threshold": self.threshold,
        }

        return final_score, details

    async def aevaluate(
        self,
        query: str,
        response: Union[MLLMImage, List[MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate text-to-image generation quality

        Args:
            query: Original text prompt (string)
            response: Generated image (MLLMImage or list with single MLLMImage)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized quality value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     query="A cat sitting on a blue sofa",
            ...     response=MLLMImage(url="cat.jpg")
            ... )
        """
        # Handle if response is a list
        if isinstance(response, list):
            if not response:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No generated image provided",
                    metadata={"error": "Empty image list"},
                )
            response = response[0]

        if not isinstance(response, MLLMImage):
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="Invalid image type",
                metadata={"error": "response must be MLLMImage"},
            )

        score, details = await self._a_compute(query, response, **kwargs)

        # Generate comprehensive reason
        reason = f"""Text-to-Image Quality Score: {score:.4f}

Semantic Consistency: {details['min_sc']:.2f}/10
{details['semantic_consistency_reason']}

Perceptual Quality: {details['min_pq']:.2f}/10
- Naturalness: {details['perceptual_quality_scores'][0]:.2f}/10
- Artifacts: {details['perceptual_quality_scores'][1]:.2f}/10
{details['perceptual_quality_reason']}

The score combines semantic consistency and perceptual quality using geometric mean.
"""

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason.strip(),
            metadata=details,
        )


__all__ = ["TextToImageGrader"]
