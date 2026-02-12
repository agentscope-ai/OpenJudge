# -*- coding: utf-8 -*-
"""
Image Helpfulness Grader

Evaluates how helpful images are in understanding surrounding text.
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
    format_image_content,
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
IMAGE_HELPFULNESS_PROMPT_EN = textwrap.dedent(
    """
You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
Your task is to evaluate the helpfulness of the image in enabling human readers to comprehend the text (context above and below) it accompanies.

<Rubrics>
Evaluate how well the image helps human readers understand the content of its accompanying text, assigning a score from 1 to 5.
A higher score indicates that the image significantly enhances comprehension of the text. Be precise when assigning the score.
</Rubrics>

<Steps>
- Read the context above and below to understand what information is being conveyed.
- Examine the image to identify what information or concepts it illustrates.
- Assess whether the image adds value beyond what the text alone provides.
- Evaluate the clarity and educational value of the image for readers.
</Steps>

<Constraints>
Be rigorous and discerning when assigning your score. Focus on how much the image aids comprehension, not just its relevance.
</Constraints>

<Scale>
- 1: The image is not at all helpful for comprehension.
- 2: The image is minimally helpful for comprehension.
- 3: The image provides some helpful context or information but may contain extraneous or less relevant details.
- 4: The image is highly helpful in enabling comprehension of the text.
- 5: The image perfectly enhances and clarifies the information provided in the text.
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
IMAGE_HELPFULNESS_PROMPT_ZH = textwrap.dedent(
    """
你是一名多模态文档评估助手。你将收到一张图片及其文本背景。
你的任务是评估图片对于帮助人类读者理解其伴随文本（上下文）的有用性。

<评分标准>
评估图片对于帮助人类读者理解伴随文本内容的有用程度，给出1到5的分数。
分数越高表示图片越能显著增强对文本的理解。请精确地给出分数。
</评分标准>

<评估步骤>
- 阅读上下文以了解正在传达的信息。
- 检查图片以识别它所说明的信息或概念。
- 评估图片是否在文本之外提供了额外价值。
- 评估图片对读者的清晰度和教育价值。
</评估步骤>

<注意事项>
请严格审慎地评分。关注图片对理解的帮助程度，而不仅仅是相关性。
</注意事项>

<评分量表>
- 1: 图片对理解文本完全没有帮助。
- 2: 图片对理解文本的帮助极小。
- 3: 图片提供了一些有用的背景或信息，但可能包含多余或关联性较弱的细节。
- 4: 图片对理解文本非常有帮助。
- 5: 图片完美地增强并澄清了文本中提供的信息。
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
DEFAULT_IMAGE_HELPFULNESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=IMAGE_HELPFULNESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=IMAGE_HELPFULNESS_PROMPT_ZH,
            ),
        ],
    },
)


class ImageHelpfulnessGrader(LLMGrader):
    """
    Image Helpfulness Grader

    Purpose:
        Evaluates how helpful images are in aiding readers' understanding of text.
        Goes beyond simple coherence to assess whether images provide genuine value,
        clarify concepts, and enhance comprehension.

    What it evaluates:
        - Information Enhancement: Does image add new understanding beyond text?
        - Concept Clarification: Does it make complex ideas easier to grasp?
        - Practical Utility: Is it genuinely useful vs. merely decorative?
        - Educational Value: Does it aid learning or task completion?
        - Comprehension Support: Does it help readers grasp the content faster?

    When to use:
        - Educational content evaluation
        - Technical documentation quality assurance
        - Tutorial and how-to guide assessment
        - Instructional design evaluation
        - User manual and help documentation review

    Scoring:
        - 5: Extremely helpful, significantly enhances understanding
        - 4: Very helpful, provides clear value
        - 3: Somewhat helpful but limited value
        - 2: Minimally helpful
        - 1: Not helpful or redundant with text
        Note: For multiple images, returns average score

    Args:
        model: Vision-language model instance or dict config
        max_context_size: Max characters from text context (default: 500)
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom template (default: DEFAULT_IMAGE_HELPFULNESS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore with helpfulness score [1, 5]

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.multimodal import ImageHelpfulnessGrader, MLLMImage
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = ImageHelpfulnessGrader(model=model)
        >>>
        >>> result = asyncio.run(grader.aevaluate(
        ...     response=[
        ...         "The system architecture has three layers.",
        ...         MLLMImage(url="https://example.com/arch_diagram.jpg"),
        ...         "Each layer handles specific functions."
        ...     ]
        ... )
        >>> print(result.score)  # 4.5 - diagram very helpful for understanding
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        max_context_size: int = 500,
        threshold: float = 0.7,
        template: PromptTemplate = DEFAULT_IMAGE_HELPFULNESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize ImageHelpfulnessGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            max_context_size: Maximum characters to extract from context (default: 500)
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_IMAGE_HELPFULNESS_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
        """
        super().__init__(
            name="image_helpfulness",
            mode=GraderMode.POINTWISE,
            description="Evaluate image helpfulness for understanding text",
            model=model,
            template=template or DEFAULT_IMAGE_HELPFULNESS_TEMPLATE,
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
        """Async evaluation of single image helpfulness"""
        messages = self.template.to_messages(self.language)
        prompt = (
            messages[0]
            .format(
                context_above=context_above or "",
                context_below=context_below or "",
            )
            .content
        )

        content = format_image_content(prompt, [image])
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
        """Compute image helpfulness score (asynchronous)"""

        image_indices = get_image_indices(response)

        if not image_indices:
            return 0.0, {
                "error": "No images found in response",
                "num_images": 0,
            }

        tasks = []
        for image_index in image_indices:
            context_above, context_below = get_image_context(
                image_index,
                response,
                self.max_context_size,
            )
            image = response[image_index]
            tasks.append(
                self._aevaluate_single_image(
                    image,
                    context_above,
                    context_below,
                ),
            )

        results = await asyncio.gather(*tasks)

        # Scores are already in 1-5 range
        scores = []
        reasons = []
        for raw_score, reason in results:
            scores.append(raw_score)
            reasons.append(reason)

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
        Evaluate image helpfulness

        Args:
            response: List containing text and images (mixed)
            **kwargs: Additional arguments (ignored)

        Returns:
            GraderScore: Score with helpfulness value [1, 5]

        Example:
            >>> result = await grader.aevaluate(
            ...     response=[
            ...         "The system architecture:",
            ...         MLLMImage(url="diagram.jpg"),
            ...         "shows the component interactions"
            ...     ]
            ... )
        """
        try:
            score, details = await self._acompute(response, **kwargs)
        except Exception as e:
            logger.exception(f"Error evaluating image helpfulness: {e}")
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
                zip(
                    details["individual_scores"],
                    details["individual_reasons"],
                ),
                1,
            ):
                reason_parts.append(f"Image {i} (score: {s:.2f}): {r}")
            reason = "\n".join(reason_parts)

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Image helpfulness score: {score:.4f}\n{reason}",
            metadata=details,
        )


__all__ = ["ImageHelpfulnessGrader", "DEFAULT_IMAGE_HELPFULNESS_TEMPLATE"]
