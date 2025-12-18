# -*- coding: utf-8 -*-
"""
Image Editing Quality Grader

Evaluates the quality of AI-based image editing.
Restructured to work with Grader framework.
"""

import asyncio
import math
import textwrap
from typing import Any, List, Tuple, Union

from loguru import logger

from rm_gallery.core.graders.base_grader import BaseGrader, GraderMode, GraderScore
from rm_gallery.core.graders.multimodal._internal import MLLMImage, format_image_content
from rm_gallery.core.graders.schema import GraderScoreCallback
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.oai.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompts
IMAGE_EDITING_SEMANTIC_PROMPT_EN = """
You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{{
    "score" : [...],
    "reason" : "..."
}}

RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

From scale 0 to 10:
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: {instruction}
"""

IMAGE_EDITING_PERCEPTUAL_PROMPT_EN = """
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
IMAGE_EDITING_SEMANTIC_PROMPT_ZH = """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的有效性。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私机密问题。

你需要按以下方式给出输出（推理请保持简洁）：
{{
    "score" : [...],
    "reason" : "..."
}}

规则：

将提供两张图像：第一张是原始AI生成的图像，第二张是第一张的编辑版本。
目标是评估编辑指令在第二张图像中的执行成功程度。

从0到10的范围：
将根据编辑的成功程度给出0到10的分数。（0表示编辑后的图像中的场景完全不遵循编辑指令。10表示编辑后的图像中的场景完美地遵循编辑指令文本。）
第二个分数从0到10，将评估第二张图像的过度编辑程度。（0表示编辑后的图像中的场景与原始图像完全不同。10表示编辑后的图像可以被识别为原始图像的最小编辑但有效的版本。）
将分数放在列表中，输出分数 = [score1, score2]，其中'score1'评估编辑成功程度，'score2'评估过度编辑程度。

编辑指令：{instruction}
"""

IMAGE_EDITING_PERCEPTUAL_PROMPT_ZH = """
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
DEFAULT_IMAGE_EDITING_SEMANTIC_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_EDITING_SEMANTIC_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_EDITING_SEMANTIC_PROMPT_ZH),
            ),
        ],
    },
)

DEFAULT_IMAGE_EDITING_PERCEPTUAL_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_EDITING_PERCEPTUAL_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(IMAGE_EDITING_PERCEPTUAL_PROMPT_ZH),
            ),
        ],
    },
)


class ImageEditingGrader(BaseGrader):
    """
    Image Editing Quality Grader

    Purpose:
        Evaluates AI-generated image edits by measuring both semantic consistency
        (does edit follow instruction?) and perceptual quality (does edit look natural?).
        Uses geometric mean to ensure both dimensions meet quality standards.

    What it evaluates:
        - Semantic Consistency: Edits correctly follow the instruction
        - Edit Precision: Only specified areas are modified
        - Naturalness: Edited areas look realistic and natural
        - Artifact-Free: No distortions, blurriness, or unnatural elements
        - Conservation: Un-instructed areas remain unchanged
        - Integration: Edits blend seamlessly with original

    When to use:
        - AI image editing model evaluation
        - Inpainting and outpainting quality assessment
        - Photo editing tool benchmarking
        - Image manipulation detection research
        - Generative model quality control

    Scoring:
        Formula: sqrt(min(semantic_scores) * min(perceptual_scores)) / 10
        - Semantic: 0-10 for instruction following + minimal over-editing
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
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.multimodal import ImageEditingGrader, MLLMImage
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = ImageEditingGrader(model=model)
        >>>
        >>> result = await grader.aevaluate(
        ...     original_image=MLLMImage(url="https://example.com/room.jpg"),
        ...     instruction="Change sofa color to blue",
        ...     edited_image=MLLMImage(url="https://example.com/room_edited.jpg")
        ... )
        >>> print(result.score)  # 0.85 - good edit quality
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.5,
        semantic_template: PromptTemplate = DEFAULT_IMAGE_EDITING_SEMANTIC_TEMPLATE,
        perceptual_template: PromptTemplate = DEFAULT_IMAGE_EDITING_PERCEPTUAL_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize ImageEditingGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.5)
            semantic_template: PromptTemplate for semantic consistency evaluation (default: DEFAULT_IMAGE_EDITING_SEMANTIC_TEMPLATE)
            perceptual_template: PromptTemplate for perceptual quality evaluation (default: DEFAULT_IMAGE_EDITING_PERCEPTUAL_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="image_editing",
            grader_mode=GraderMode.POINTWISE,
            description="Evaluate image editing quality",
        )
        self.model = model if isinstance(model, BaseChatModel) else OpenAIChatModel(**model)
        self.threshold = threshold
        self.semantic_template = semantic_template
        self.perceptual_template = perceptual_template
        self.language = language

    async def _aevaluate_semantic_consistency(
        self,
        original_image: MLLMImage,
        instruction: str,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        messages = self.semantic_template.to_messages(self.language)
        prompt = messages[0].format(instruction=instruction).content

        try:
            content = format_image_content(prompt, [original_image, edited_image])
            chat_response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=GraderScoreCallback,
            )
            score = chat_response.metadata["score"]
            score = score if isinstance(score, list) else [score]
            reason = chat_response.metadata["reason"]
            return score, reason

        except Exception as e:
            logger.error(f"Error evaluating semantic consistency: {e}")
            return [5.0], f"Error during evaluation: {str(e)}"

    async def _aevaluate_perceptual_quality(
        self,
        edited_image: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        messages = self.perceptual_template.to_messages(self.language)
        prompt = messages[0].content

        try:
            content = format_image_content(prompt, [edited_image])
            chat_response = await self.model.achat(
                messages=[{"role": "user", "content": content}],
                structured_model=GraderScoreCallback,
            )
            score = chat_response.metadata["score"]
            score = score[:2] if isinstance(score, list) else [score, score]
            reason = chat_response.metadata["reason"]
            return score, reason

        except Exception as e:
            logger.error(f"Error evaluating perceptual quality: {e}")
            return [5.0, 5.0], f"Error during evaluation: {str(e)}"

    async def _acompute(
        self,
        original_image: MLLMImage,
        instruction: str,
        edited_image: MLLMImage,
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """
        Compute image editing quality score (asynchronous)

        Args:
            original_image: Original image before editing
            instruction: Editing instruction
            edited_image: Edited image to evaluate

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """

        # Evaluate semantic consistency and perceptual quality in parallel
        (sc_scores, sc_reason), (
            pq_scores,
            pq_reason,
        ) = await asyncio.gather(
            self._aevaluate_semantic_consistency(
                original_image,
                instruction,
                edited_image,
            ),
            self._aevaluate_perceptual_quality(edited_image),
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
        original_image: Union[MLLMImage, List[MLLMImage]],
        instruction: str,
        edited_image: Union[MLLMImage, List[MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate image editing quality

        Args:
            original_image: Original image before editing (MLLMImage or list)
            instruction: Editing instruction (string)
            edited_image: Edited image to evaluate (MLLMImage or list)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with normalized quality value [0, 1]

        Example:
            >>> result = await grader.aevaluate(
            ...     original_image=MLLMImage(url="original.jpg"),
            ...     instruction="Change the sofa color to blue",
            ...     edited_image=MLLMImage(url="edited.jpg")
            ... )
        """
        # Handle if images are lists
        if isinstance(original_image, list):
            if not original_image:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No original image provided",
                    metadata={"error": "Empty original image list"},
                )
            original_image = original_image[0]

        if isinstance(edited_image, list):
            if not edited_image:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No edited image provided",
                    metadata={"error": "Empty edited image list"},
                )
            edited_image = edited_image[0]

        if not isinstance(original_image, MLLMImage) or not isinstance(
            edited_image,
            MLLMImage,
        ):
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="Invalid image type",
                metadata={"error": "Images must be MLLMImage"},
            )

        score, details = await self._acompute(
            original_image,
            instruction,
            edited_image,
            **kwargs,
        )

        # Generate comprehensive reason
        reason = f"""Image Editing Quality Score: {score:.4f}

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


__all__ = [
    "ImageEditingGrader",
]
