# -*- coding: utf-8 -*-
"""
Image Coherence Grader

Evaluates the coherence between images and their surrounding text context.
Restructured to work with Grader framework.
"""

import asyncio
import textwrap
from typing import Any, List, Optional, Tuple, Union

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.multimodal._internal import (
    MLLMImage,
    get_image_context,
    get_image_indices,
)
from openjudge.graders.schema import GraderScoreCallback
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate
from openjudge.utils.utils import parse_structured_chat_response

# pylint: disable=line-too-long

# English Prompt
IMAGE_COHERENCE_PROMPT_EN = textwrap.dedent(
    """
You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
Your task is to evaluate the coherence between the image and the text (context above and below) it accompanies.

<Rubrics>
Assess how coherent the image is in relation to its accompanying text, assigning a score from 1 to 5.
A higher score indicates stronger coherence between the image and the text. Be precise when assigning the score.
</Rubrics>

<Steps>
- Analyze the context above and below to understand the textual content and topic.
- Examine the image to identify its main subjects and visual elements.
- Compare the image content with the surrounding text to assess semantic alignment.
- Evaluate whether the image placement is logical and enhances the text.
</Steps>

<Constraints>
Be rigorous and discerning when assigning your score. Focus on the semantic relationship between image and text, not image quality.
</Constraints>

<Scale>
- 1: The image is not at all coherent with the text.
- 2: The image is minimally coherent with the text.
- 3: The image shows some coherence with the text but may include unrelated elements.
- 4: The image is highly coherent with the text.
- 5: Perfect coherence, where the image completely corresponds with and enhances the text.
</Scale>

<Context Above>{context_above}</Context Above>
<Context Below>{context_below}</Context Below>
<Image>[The image is provided below this section.]</Image>

<Output Schema>
{{
    "reason": "brief explanation for the assigned score",
    "score": integer between 1 and 5
}}
</Output Schema>
JSON:
"""
).strip()

# Chinese Prompt
IMAGE_COHERENCE_PROMPT_ZH = textwrap.dedent(
    """
你是一名多模态文档评估助手。你将收到一张图片及其文本背景。
你的任务是评估图片与其伴随文本（上下文）之间的连贯性。

<评分标准>
评估图片与其伴随文本的连贯性，给出1到5的分数。
分数越高表示图片与文本之间的连贯性越强。请精确地给出分数。
</评分标准>

<评估步骤>
- 分析上下文以理解文本内容和主题。
- 检查图片以识别其主要对象和视觉元素。
- 将图片内容与周围文本进行比较，评估语义一致性。
- 评估图片位置是否合理，是否增强了文本内容。
</评估步骤>

<注意事项>
请严格审慎地评分。关注图片与文本之间的语义关系，而非图片质量本身。
</注意事项>

<评分量表>
- 1: 图片与文本完全不连贯。
- 2: 图片与文本的连贯性极低。
- 3: 图片与文本有一定连贯性，但可能包含无关元素。
- 4: 图片与文本高度连贯。
- 5: 完美连贯，图片完全对应并增强文本内容。
</评分量表>

<上文>{context_above}</上文>
<下文>{context_below}</下文>
<图片>[图片将在本节下方提供。]</图片>

<输出格式>
{{
    "reason": "对所给分数的简要解释",
    "score": 1到5之间的整数
}}
</输出格式>
JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_IMAGE_COHERENCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=IMAGE_COHERENCE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=IMAGE_COHERENCE_PROMPT_ZH,
            ),
        ],
    },
)


class ImageCoherenceGrader(LLMGrader):
    """
    Image Coherence Grader

    Purpose:
        Evaluates how well images match and relate to their surrounding text context.
        Assesses whether images are appropriately placed and meaningfully connected
        to the text above and below them.

    What it evaluates:
        - Semantic Alignment: Image content matches surrounding text topic
        - Contextual Relevance: Image relates to both preceding and following text
        - Visual-Text Consistency: Image illustrates concepts mentioned in text
        - Placement Appropriateness: Image positioned at logical point in content

    When to use:
        - Document generation with embedded images
        - Multimodal content quality assurance
        - Educational material evaluation
        - Technical documentation review
        - Marketing content assessment

    Scoring:
        - 5: Perfect coherence, image perfectly illustrates text
        - 4: Strong coherence with clear relationship
        - 3: Some coherence but connection could be clearer
        - 2: Weak coherence, image seems somewhat misplaced
        - 1: No coherence, image is completely unrelated
        Note: For multiple images, returns average score

    Args:
        model: Vision-language model instance or dict config
        max_context_size: Max characters from text context (default: 500)
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_IMAGE_COHERENCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore with coherence score [1, 5]

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.multimodal import ImageCoherenceGrader, MLLMImage
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = ImageCoherenceGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     response=[
        ...         "Q3 sales increased 25%.",
        ...         MLLMImage(url="https://example.com/sales_chart.jpg"),
        ...         "Growth driven by new products."
        ...     ]
        ... ))
        >>> print(result.score)  # 4.8 - image coherent with sales context
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        max_context_size: int = 500,
        threshold: float = 0.7,
        template: PromptTemplate = DEFAULT_IMAGE_COHERENCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ImageCoherenceGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            max_context_size: Maximum characters to extract from context (default: 500)
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_IMAGE_COHERENCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="image_coherence",
            mode=GraderMode.POINTWISE,
            description="Evaluate image-text coherence",
            model=model,
            template=template or DEFAULT_IMAGE_COHERENCE_TEMPLATE,
            language=language,
            strategy=strategy,
        )
        self.max_context_size = max_context_size
        self.threshold = threshold

    async def _aevaluate_single_image(
        self,
        image: MLLMImage,
        context_above: Optional[str],
        context_below: Optional[str],
    ) -> Tuple[float, str]:
        """Async evaluation of single image coherence"""
        messages = self.template.to_messages(self.language)
        prompt = messages[0].content.format(
            context_above=context_above or "",
            context_below=context_below or "",
        )

        # Format image content for OpenAI API
        content = [{"type": "text", "text": prompt}]

        if image.url:
            content.append({"type": "image_url", "image_url": {"url": image.url}})
        elif image.base64:
            # Format base64 image with data URL scheme
            image_format = image.format or "jpeg"
            data_url = f"data:image/{image_format};base64,{image.base64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        chat_response = await self.model.achat(
            messages=[{"role": "user", "content": content}],
            structured_model=GraderScoreCallback,
        )

        # Default to 3.0 (neutral score on 1-5 scale) for missing fields
        parsed = await parse_structured_chat_response(chat_response)
        score = parsed.get("score", 3.0)
        reason = parsed.get("reason", "")
        return score, reason

    async def _acompute(
        self,
        response: List[Union[str, MLLMImage]],
        **_kwargs: Any,
    ) -> Tuple[float, dict]:
        """
        Compute image coherence score (asynchronous)

        Args:
            response: List containing text and images

        Returns:
            tuple[float, dict]: (normalized_score [0,1], details)
        """

        # Find all images
        image_indices = get_image_indices(response)

        if not image_indices:
            return 0.0, {
                "error": "No images found in response",
                "num_images": 0,
            }

        # Prepare evaluation tasks
        tasks = []
        for image_index in image_indices:
            context_above, context_below = get_image_context(
                image_index,
                response,
                self.max_context_size,
            )
            image = response[image_index]
            tasks.append(
                self._aevaluate_single_image(image, context_above, context_below),
            )

        # Evaluate all images in parallel
        results = await asyncio.gather(*tasks)

        # Process results (scores are already in 1-5 range)
        scores = []
        reasons = []
        for raw_score, reason in results:
            scores.append(raw_score)
            reasons.append(reason)

        # Compute average score
        final_score = sum(scores) / len(scores) if scores else 0.0

        details = {
            "num_images": len(image_indices),
            "individual_scores": scores,
            "individual_reasons": reasons,
            "threshold": self.threshold,
        }

        return final_score, details

    async def _aevaluate(
        self,
        response: List[Union[str, MLLMImage]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate image coherence

        Args:
            response: List containing text and images (mixed)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with coherence value [1, 5]

        Example:
            >>> result = await grader.aevaluate(
            ...     response=[
            ...         "Sales data for Q3:",
            ...         MLLMImage(url="chart.jpg"),
            ...         "Shows 20% growth"
            ...     ]
            ... )
        """
        try:
            score, details = await self._acompute(response, **kwargs)
        except Exception as e:
            logger.exception(f"Error evaluating image coherence: {e}")
            from openjudge.graders.base_grader import GraderError

            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )

        if "error" in details:
            return GraderScore(
                name=self.name,
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
                zip(details["individual_scores"], details["individual_reasons"]),
                1,
            ):
                reason_parts.append(f"Image {i} (score: {s:.2f}): {r}")
            reason = "\n".join(reason_parts)

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Image coherence score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageCoherenceGrader", "DEFAULT_IMAGE_COHERENCE_TEMPLATE"]
