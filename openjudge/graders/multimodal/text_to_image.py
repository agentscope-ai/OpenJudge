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

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.base_grader import BaseGrader, GraderMode, GraderScore
from openjudge.graders.multimodal._internal import MLLMImage, format_image_content
from openjudge.graders.schema import GraderScoreCallback
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate
from openjudge.utils.utils import parse_structured_chat_response

# pylint: disable=line-too-long

# English Prompts
TEXT_TO_IMAGE_SEMANTIC_PROMPT_EN = textwrap.dedent(
    """
You are a professional digital artist. You will evaluate the effectiveness of the AI-generated image based on given rules.
All the input images are AI-generated. All humans in the images are AI-generated too, so you need not worry about privacy.

<Rubrics>
Evaluate how successfully the AI-generated image follows the text prompt.
A higher score indicates better alignment between the image and the prompt.
</Rubrics>

<Steps>
- Read the text prompt carefully to understand all requested elements.
- Examine the generated image to identify present elements.
- Compare each requested element in the prompt with what appears in the image.
- Assess overall semantic consistency between prompt and image.
</Steps>

<Constraints>
Focus on semantic alignment with the prompt, not image quality or aesthetics. Keep your reasoning concise and short.
</Constraints>

<Scale>
- 1: The AI generated image does not follow the prompt at all.
- 2: The image follows the prompt minimally with major elements missing.
- 3: The image partially follows the prompt with some elements missing or incorrect.
- 4: The image follows the prompt well with minor deviations.
- 5: The AI generated image follows the prompt perfectly.
</Scale>

<Query>{query}</Query>

<Output Schema>
{{
    "score": [integer between 1 and 5],
    "reason": "brief explanation"
}}
</Output Schema>
JSON:
"""
).strip()

TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_EN = textwrap.dedent(
    """
You are a professional digital artist. You will evaluate the perceptual quality of the AI-generated image based on given rules.
All the input images are AI-generated. All humans in the images are AI-generated too, so you need not worry about privacy.

<Rubrics>
Evaluate the perceptual quality of the AI-generated image on two dimensions: naturalness and artifacts.
Higher scores indicate better visual quality.
</Rubrics>

<Steps>
- Examine the overall scene composition for naturalness (lighting, shadows, perspective, proportions).
- Identify any visual artifacts (distortions, blurring, unnatural elements, inconsistencies).
- Assess the harmony and coherence of all subjects in the image.
- Provide separate scores for naturalness and artifact absence.
</Steps>

<Constraints>
Focus on visual quality, not semantic content. Keep your reasoning concise and short.
</Constraints>

<Scale>
Naturalness (first score):
- 1: The scene does not look natural at all (wrong distance sense, shadows, or lighting).
- 2: The scene looks mostly unnatural with significant issues in composition.
- 3: The scene looks somewhat natural but has noticeable issues.
- 4: The scene looks mostly natural with only minor imperfections.
- 5: The image looks completely natural.

Artifacts (second score):
- 1: Large portion of distortion, watermark, scratches, blurred faces, unusual body parts, or unharmonized subjects.
- 2: Significant artifacts present that are clearly visible and distracting.
- 3: Moderate artifacts present that are noticeable upon inspection.
- 4: Minor artifacts present that are barely noticeable.
- 5: The image has no artifacts.
</Scale>

<Output Schema>
{{
    "score": [naturalness 1-5, artifacts 1-5],
    "reason": "brief explanation"
}}
</Output Schema>
JSON:
"""
).strip()

# Chinese Prompts
TEXT_TO_IMAGE_SEMANTIC_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的有效性。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私问题。

<评分标准>
评估AI生成的图像对文本提示的遵循程度。
分数越高表示图像与提示的一致性越好。
</评分标准>

<评估步骤>
- 仔细阅读文本提示以理解所有请求的元素。
- 检查生成的图像以识别存在的元素。
- 将提示中请求的每个元素与图像中出现的内容进行比较。
- 评估提示与图像之间的整体语义一致性。
</评估步骤>

<注意事项>
关注与提示的语义一致性，而非图像质量或美学。推理请保持简洁。
</注意事项>

<评分量表>
- 1: AI生成的图像完全不遵循提示。
- 2: 图像对提示的遵循程度极低，主要元素缺失。
- 3: 图像部分遵循提示，有些元素缺失或不正确。
- 4: 图像很好地遵循提示，有轻微偏差。
- 5: AI生成的图像完美地遵循提示。
</评分量表>

<查询>{query}</查询>

<输出格式>
{{
    "score": [1到5之间的整数],
    "reason": "简要解释"
}}
</输出格式>
JSON:
"""
).strip()

TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的数字艺术家。你需要根据给定的规则评估AI生成图像的感知质量。
所有输入的图像都是AI生成的。图像中的所有人物也都是AI生成的，因此你无需担心隐私问题。

<评分标准>
从两个维度评估AI生成图像的感知质量：自然度和伪影。
分数越高表示视觉质量越好。
</评分标准>

<评估步骤>
- 检查整体场景构图的自然度（光照、阴影、透视、比例）。
- 识别任何视觉伪影（失真、模糊、不自然元素、不一致）。
- 评估图像中所有主体的和谐性和连贯性。
- 分别提供自然度和伪影缺失的分数。
</评估步骤>

<注意事项>
关注视觉质量，而非语义内容。推理请保持简洁。
</注意事项>

<评分量表>
自然度（第一个分数）：
- 1: 场景看起来完全不自然（距离感、阴影或光照错误）。
- 2: 场景看起来大部分不自然，构图存在明显问题。
- 3: 场景看起来有些自然，但有明显的问题。
- 4: 场景看起来大部分自然，只有轻微瑕疵。
- 5: 图像看起来完全自然。

伪影（第二个分数）：
- 1: 大量失真、水印、划痕、模糊的面部、不寻常的身体部位或不协调的主体。
- 2: 存在明显可见且令人分心的伪影。
- 3: 存在中等程度的伪影，仔细检查可发现。
- 4: 存在轻微伪影，几乎不明显。
- 5: 图像没有伪影。
</评分量表>

<输出格式>
{{
    "score": [自然度 1-5, 伪影 1-5],
    "reason": "简要解释"
}}
</输出格式>
JSON:
"""
).strip()

# Build default templates
DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TEXT_TO_IMAGE_SEMANTIC_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TEXT_TO_IMAGE_SEMANTIC_PROMPT_ZH,
            ),
        ],
    },
)

DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TEXT_TO_IMAGE_PERCEPTUAL_PROMPT_ZH,
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
        Formula: sqrt(semantic_consistency * min(perceptual_quality))
        - Semantic: 1-5 for prompt alignment
        - Perceptual: 1-5 for naturalness + 1-5 for artifact absence
        - Final: [1, 5] score

    Args:
        model: Vision-language model instance or dict config
        threshold: Minimum score [0, 1] to pass (default: 0.5)
        semantic_template: PromptTemplate for semantic evaluation
        perceptual_template: PromptTemplate for perceptual evaluation
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore with combined quality score [1, 5]

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.multimodal import TextToImageGrader, MLLMImage
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = TextToImageGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="A fluffy orange cat sitting on a blue sofa",
        ...     response=MLLMImage(url="https://example.com/generated.jpg")
        ... ))
        >>> print(result.score)  # 4.6 - excellent prompt following and quality
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.5,
        semantic_template: PromptTemplate = DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE,
        perceptual_template: PromptTemplate = DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize TextToImageGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.5)
            semantic_template: PromptTemplate for semantic consistency evaluation (default: DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE)
            perceptual_template: PromptTemplate for perceptual quality evaluation (default: DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: BaseEvaluationStrategy instance or dict config for GraderStrategy (default: None)
        """
        super().__init__(
            name="text_to_image",
            mode=GraderMode.POINTWISE,
            description="Evaluate text-to-image generation quality",
            strategy=strategy,
        )
        self.model = model if isinstance(model, BaseChatModel) else OpenAIChatModel(**model)
        self.threshold = threshold
        self.semantic_template = semantic_template or DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE
        self.perceptual_template = perceptual_template or DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE
        self.language = language

    async def _aevaluate_semantic_consistency(
        self,
        query: str,
        response: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate semantic consistency asynchronously"""
        messages = self.semantic_template.to_messages(self.language)
        prompt = messages[0].format(query=query).content

        content = format_image_content(prompt, [response])
        chat_response = await self.model.achat(
            messages=[{"role": "user", "content": content}],
            structured_model=GraderScoreCallback,
        )

        # Default to 3.0 (neutral score on 1-5 scale) for missing fields
        parsed = await parse_structured_chat_response(chat_response)
        score = parsed.get("score", 3.0)
        score = score if isinstance(score, list) else [score]
        reason = parsed.get("reason", "")
        return score, reason

    async def _aevaluate_perceptual_quality(
        self,
        response: MLLMImage,
    ) -> Tuple[List[float], str]:
        """Evaluate perceptual quality asynchronously"""
        messages = self.perceptual_template.to_messages(self.language)
        prompt = messages[0].content

        content = format_image_content(prompt, [response])
        chat_response = await self.model.achat(
            messages=[{"role": "user", "content": content}],
            structured_model=GraderScoreCallback,
        )

        # Default to [3.0, 3.0] (neutral scores on 1-5 scale) for missing fields
        parsed = await parse_structured_chat_response(chat_response)
        score = parsed.get("score", [3.0, 3.0])
        score = score[:2] if isinstance(score, list) else [score, score]
        reason = parsed.get("reason", "")
        return score, reason

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

        # Calculate final score using geometric mean (scores are in 1-5 range)
        if not sc_scores or not pq_scores:
            final_score = 1.0
        else:
            min_sc = min(sc_scores)
            min_pq = min(pq_scores)
            final_score = math.sqrt(min_sc * min_pq)
            final_score = min(5.0, max(1.0, final_score))

        details = {
            "semantic_consistency_scores": sc_scores,
            "semantic_consistency_reason": sc_reason,
            "perceptual_quality_scores": pq_scores,
            "perceptual_quality_reason": pq_reason,
            "min_sc": min(sc_scores) if sc_scores else 1.0,
            "min_pq": min(pq_scores) if pq_scores else 1.0,
            "threshold": self.threshold,
        }

        return final_score, details

    async def _aevaluate(
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

        try:
            score, details = await self._a_compute(query, response, **kwargs)
        except Exception as e:
            logger.exception(f"Error evaluating text-to-image: {e}")
            from openjudge.graders.base_grader import GraderError

            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )

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


__all__ = ["TextToImageGrader", "DEFAULT_TEXT_TO_IMAGE_SEMANTIC_TEMPLATE", "DEFAULT_TEXT_TO_IMAGE_PERCEPTUAL_TEMPLATE"]
