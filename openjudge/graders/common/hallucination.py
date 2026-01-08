# -*- coding: utf-8 -*-
"""
Hallucination Grader

Evaluates whether model response contain hallucinations (fabricated information not
supported by the context).
"""

import textwrap
from typing import Any, Dict, Optional

from loguru import logger

from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
HALLUCINATION_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model response contains hallucinations. Your task is to score according to the following criteria:

<Scoring Criteria>
A hallucination-free response should:
- Contain only verifiable facts (If context is provided, verify support from the context. If no context is provided or if the context is inconsistent with facts/common knowledge, verify factual correctness based on common knowledge).
- Not make unsupported claims or assumptions.
- Not add speculative or imagined details.
- Be completely accurate regarding dates, numbers, and specific details.
- Appropriately indicate uncertainty when information is incomplete.
</Scoring Criteria>

<Guidance>
- Thoroughly read the query and response.
- Identify all claims made in the response.
- If context is provided: Cross-check each claim with the context.
- If no context is provided, or the context is inconsistent with facts/common knowledge: verify the statement based on common knowledge and logical consistency.
- Note any unsupported, contradictory, or factually incorrect information.
- Consider the severity and number of hallucinations.
</Guidance>

<Reminder>
Focus only on factual accuracy. If context is provided, verify support from the context. If no context is provided or if the context is inconsistent with facts/common knowledge, verify factual correctness based on common knowledge. Do not consider style, grammar, or presentation when scoring. A short but factual response should score higher than a longer response containing unsupported claims.
</Reminder>

<query>
{query}
</query>

<response>
{response}
</response>

Use the following context to help you evaluate whether there are hallucinations in the response (ignore if empty):
<context>
{context}
</context>

If available, you may also use the following reference response to help you identify hallucinations in the response (ignore if empty):
<reference_response>
{reference_response}
</reference_response>


# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5, where 5 means no hallucinations and 1 means severe hallucinations>,
    "reason": "<brief explanation for the assigned score, specifically mentioning any hallucinations found or confirming factual accuracy>"
}}

Scoring Scale:
- 5: Response is not hallucinatory
- 4: Response has slight deviation
- 3: Response is partially fabricated
- 2: Response is seriously fabricated
- 1: Response is completely fabricated

JSON:
"""

# Chinese Prompt
HALLUCINATION_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否包含幻觉（虚构信息）。你的任务是根据以下标准进行评分：

<评分标准>
无幻觉的回答应该：
- 仅包含可验证事实（如果提供了上下文，则需要参考上下文。如果未提供上下文或者上下文与事实/常识不一致，则基于事实/常识验证事实正确性）。
- 不做出无依据的声明或假设。
- 不添加推测性或想象的细节。
- 在日期、数字和具体细节方面完全准确。
- 在信息不完整时适当地表示不确定性。
</评分标准>

<指导>
- 仔细阅读输入问题和输出回答。
- 识别输出中的所有声明。
- 如果提供了上下文：需要参考上下文。
- 如果未提供上下文或者上下文与事实/常识不一致：根据常识和逻辑一致性验证声明。
- 注意任何无依据、矛盾或事实错误的信息。
- 考虑幻觉的严重程度和数量。
</指导>

<提醒>
仅关注事实准确性。如果提供了上下文，则需要参考上下文。如果未提供上下文或者上下文与事实/常识不一致，则基于事实/常识验证事实正确性。评分时不要考虑风格、语法或呈现方式。简短但真实的回答应该比包含无依据声明的较长回答得分更高。
</提醒>

<查询>
{query}
</查询>

<回答>
{response}
</回答>

使用以下上下文帮助你评估输出中是否存在幻觉（如为空则忽略）:
<上下文>
{context}
</上下文>

如有需要，你也可以使用以下参考输出来帮助识别回答中的幻觉（如为空则忽略）：
<参考回答>
{reference_response}
</参考回答>

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1到5之间的整数，其中5表示无幻觉，1表示完全捏造>,
    "reason": "<对所给分数的简要解释，特别提到发现的任何幻觉或确认事实准确性>"
}}

评分标尺：
- 5: 输出回答无幻觉
- 4: 输出回答轻微偏差
- 3: 输出回答局部虚构
- 2: 输出回答严重虚构
- 1: 输出回答完全捏造

JSON:
"""


# Build default template from prompts
DEFAULT_HALLUCINATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(HALLUCINATION_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(HALLUCINATION_PROMPT_ZH),
            ),
        ],
    },
)


class HallucinationGrader(LLMGrader):
    """
    Hallucination Grader

    Purpose:
        Detects hallucinations in model outputs by verifying that all claims are properly
        grounded in the provided context. A hallucination occurs when the model generates
        information that is not supported by, or contradicts, the given context.

    What it evaluates:
        - Factual Grounding: Every claim must be supported by the input context
        - Claim Verification: All statements must have evidence in provided materials
        - Speculation Detection: Model should not add imagined or assumed details
        - Numerical Accuracy: Dates, numbers, and statistics must be exact
        - Contradiction Avoidance: Output must not contradict the context

    When to use:
        - RAG (Retrieval-Augmented Generation) systems where context is provided
        - Question-answering systems that must stay grounded in given documents
        - Summarization tasks where fidelity to source is critical
        - Fact-checking generated content against reference materials
        - General factual accuracy evaluation (without context, based on common knowledge)

    Scoring:
        - 5: Response is not hallucinatory
        - 4: Response has slight deviation
        - 3: Response is partially fabricated
        - 2: Response is seriously fabricated
        - 1: Response is completely fabricated

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_HALLUCINATION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = no hallucinations, 1 = severe hallucinations
            - reason: Detailed explanation of any hallucinations found
            - metadata: Threshold and evaluation details

    Example:
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.llm_judge import HallucinationGrader
        >>>
        >>> # Initialize model
        >>> model = OpenAIChatModel(
        ...     api_key="sk-...",
        ...     model="qwen3-max",
        ...     temperature=0.1
        ... )
        >>>
        >>> # Create grader
        >>> grader = HallucinationGrader(model=model, threshold=0.7)
        >>>
        >>> # With context: Good output (grounded in context)
        >>> result = await grader.aevaluate(
        ...     query="When was the company founded?",
        ...     response="The company was founded in 2020 in San Francisco.",
        ...     context="The company was founded in 2020 in San Francisco."
        ... )
        >>> print(result.score)  # 5 - no hallucinations
        >>> print(result.reason)  # "Output is fully supported by context"
        >>>
        >>> # With context: Bad output (contains hallucination)
        >>> result = await grader.aevaluate(
        ...     query="When was the company founded?",
        ...     response="The company was founded in 2020 with 100 employees.",
        ...     context="The company was founded in 2020 in San Francisco."
        ... )
        >>> print(result.score)  # 3 - contains unsupported claim about employees
        >>> print(result.reason)  # "Output contains hallucination: '100 employees' not mentioned"
        >>>
        >>> # Without context: Factual verification
        >>> result = await grader.aevaluate(
        ...     query="What is the capital of France?",
        ...     response="The capital of France is Paris."
        ... )
        >>> print(result.score)  # 5 - factually correct
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_HALLUCINATION_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize HallucinationGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_HALLUCINATION_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="hallucination",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether response contains hallucinations",
            model=model,
            template=template or DEFAULT_HALLUCINATION_TEMPLATE,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        query: str,
        response: str,
        context: str = "",
        reference_response: str = "",
    ) -> GraderScore:
        """
        Evaluate hallucination in response

        Args:
            query: Input question or prompt
            response: Model response to evaluate
            context: Context information to verify against. If empty string (default),
                    evaluation will be based on general factual consistency and common knowledge.
            reference_response: Reference response for comparison. Defaults to empty string.

        Returns:
            GraderScore: Score with hallucination value [1, 5]
                        where 5 means no hallucinations, 1 means severe hallucinations

        Example:
            >>> # With context
            >>> result = await grader.aevaluate(
            ...     query="When did the product launch?",
            ...     response="The product launched in 2023 with great success.",
            ...     context="The product launched in 2023.",
            ...     reference_response="The product launched in 2023."
            ... )
            >>> # Without context
            >>> result = await grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     response="The capital of France is Paris."
            ... )
        """

        try:
            result = await super().aevaluate(
                query=query,
                response=response,
                context=context,
                reference_response=reference_response,
            )
            score = result.score
            reason = result.reason

        except Exception as e:
            logger.error(f"Error evaluating hallucination: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        prompt = DEFAULT_HALLUCINATION_TEMPLATE.get_prompt()
        return {"aevaluate": HallucinationGrader.aevaluate.__doc__, "prompt": prompt}


__all__ = ["HallucinationGrader", "DEFAULT_HALLUCINATION_TEMPLATE"]
