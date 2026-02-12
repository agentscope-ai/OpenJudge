# -*- coding: utf-8 -*-
"""
Instruction Clarification Grader for Multi-turn Conversations (IC)

Evaluates whether the assistant can appropriately ask for clarification
when faced with vague, ambiguous, or incomplete user queries.

This is an advanced interaction ability that tests the model's capacity to:
- Recognize when a query lacks necessary details
- Ask targeted clarifying questions
- Avoid making assumptions when information is insufficient
- Guide users to provide the information needed for a helpful response
"""

import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate
from openjudge.utils.utils import format_conversation_history

# ============================================================================
# Prompt Templates
# ============================================================================

INSTRUCTION_CLARIFICATION_PROMPT_ZH = textwrap.dedent(
    """
您是一位专业的指令澄清能力评估专家。您的任务是评估多轮对话中，当用户提出模糊或不完整的问题时，助手是否能够恰当地追问细节以提供更准确的帮助。

<评分标准>
完美的指令澄清应该：
- 识别出用户问题中的模糊或缺失信息。
- 追问针对性强，能够获取关键缺失信息。
- 在追问时保持礼貌和专业的态度。
- 只在真正需要时才追问，而不是过度追问。
- 追问能有效引导用户提供所需信息。
</评分标准>

<评估步骤>
1. 分析用户的问题，确定问题是否存在模糊或歧义。
2. 判断缺少哪些关键信息才能给出准确回答。
3. 评估助手是否识别出了问题的模糊之处。
4. 检查追问的问题是否针对关键缺失信息。
5. 评估追问方式是否礼貌且专业。
</评估步骤>

<注意事项>
如果问题本身足够清晰，不需要追问，助手直接回答是正确的。过度追问（问题已经清晰还追问）应该扣分。追问应该具体而非泛泛而问。
</注意事项>

<评分量表>
- 5分：完美澄清，助手准确识别模糊点，追问针对性强且礼貌，有效引导用户。
- 4分：良好澄清，助手识别了主要模糊点，追问基本恰当，但可能略有不足。
- 3分：基本澄清，助手进行了追问，但追问不够精准或遗漏了重要方面。
- 2分：澄清不足，助手未能有效识别模糊点，追问过于宽泛或不够相关。
- 1分：未能澄清，助手完全忽略问题的模糊性，直接给出可能错误的回答，或过度追问。
</评分量表>

<对话历史>
{history}
</对话历史>

<当前助手回复>
{response}
</当前助手回复>

<输出格式>
请按以下JSON格式输出评估结果：
{{
    "reason": "详细说明评估理由，包括模糊点识别和追问质量分析",
    "score": 1-5的整数
}}
</输出格式>

JSON:
"""
).strip()


INSTRUCTION_CLARIFICATION_PROMPT_EN = textwrap.dedent(
    """
You are a professional Instruction Clarification Evaluation Expert. Your task is to
evaluate whether the assistant can appropriately ask for clarification when users pose
vague or incomplete questions.

<Rubrics>
Perfect instruction clarification should:
- Identify vague or missing information in user queries.
- Ask targeted clarifying questions to obtain key missing information.
- Maintain a polite and professional attitude when asking.
- Only ask when truly needed, not over-questioning.
- Effectively guide users to provide needed information.
</Rubrics>

<Steps>
1. Analyze the user's question to determine if there is ambiguity or vagueness.
2. Determine what key information is missing for an accurate answer.
3. Evaluate if the assistant identified the ambiguous aspects.
4. Check if the clarifying questions target key missing information.
5. Evaluate if the questioning manner is polite and professional.
</Steps>

<Constraints>
If the question is clear enough, directly answering is correct. Over-questioning (when
question is already clear) should be penalized. Questions should be specific, not vague.
</Constraints>

<Scale>
- 5: Perfect clarification - accurately identified ambiguity, targeted and polite questions, effectively guided user.
- 4: Good clarification - identified main ambiguities, mostly appropriate questions, but slightly lacking.
- 3: Basic clarification - asked questions but not precise enough or missed important aspects.
- 2: Insufficient clarification - failed to effectively identify ambiguity, questions too broad or irrelevant.
- 1: Failed clarification - completely ignored ambiguity, gave potentially wrong answer, or over-questioned.
</Scale>

<Conversation History>
{history}
</Conversation History>

<Current Assistant Response>
{response}
</Current Assistant Response>

<Output Schema>
Output your evaluation in the following JSON format:
{{
    "reason": "detailed explanation including ambiguity identification and question quality analysis",
    "score": integer from 1-5
}}
</Output Schema>

JSON:
"""
).strip()


DEFAULT_INSTRUCTION_CLARIFICATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=INSTRUCTION_CLARIFICATION_PROMPT_ZH),
        ],
        LanguageEnum.EN: [
            ChatMessage(role="user", content=INSTRUCTION_CLARIFICATION_PROMPT_EN),
        ],
    },
)


# ============================================================================
# Grader Implementation
# ============================================================================


class InstructionClarificationGrader(LLMGrader):
    """
    Instruction Clarification Grader for Multi-turn Conversations.

    Purpose:
        Evaluates whether the assistant can appropriately ask for clarification
        when faced with vague, ambiguous, or incomplete user queries, ensuring
        accurate and helpful responses.

    What it evaluates:
        - Ambiguity Detection: Recognizing vague or incomplete queries
        - Targeted Questions: Asking specific clarifying questions
        - Professional Manner: Maintaining polite and helpful tone
        - Appropriate Timing: Only asking when truly necessary
        - User Guidance: Effectively guiding users to provide needed info

    When to use:
        - Evaluating clarification behavior in chatbots
        - Testing ambiguity handling capabilities
        - Assessing user guidance quality
        - Quality assurance for customer service bots
        - Measuring conversational intelligence

    Scoring:
        - 5: Perfect clarification - accurate detection, targeted questions
        - 4: Good clarification - mostly appropriate with minor gaps
        - 3: Basic clarification - questions asked but not precise
        - 2: Insufficient clarification - failed to identify ambiguity
        - 1: Failed clarification - ignored ambiguity or over-questioned

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        template: Custom evaluation template (default: DEFAULT_INSTRUCTION_CLARIFICATION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect clarification, 1 = failed
            - reason: Explanation of clarification quality analysis
            - metadata: Evaluation details including history_turns

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.multi_turn import InstructionClarificationGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = InstructionClarificationGrader(model=model)
        >>>
        >>> # Evaluate clarification behavior
        >>> history = [
        ...     {"role": "user", "content": "Book me a flight."},  # Vague query
        ... ]
        >>> response = (
        ...     "I'd be happy to help! Could you tell me your departure city, "
        ...     "destination, and preferred travel dates?"
        ... )
        >>> result = asyncio.run(grader.aevaluate(response=response, history=history))
        >>> print(result.score)  # Expected: high score for appropriate clarification
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
        **kwargs: Any,
    ):
        """
        Initialize InstructionClarificationGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel.
            template: Custom PromptTemplate. Defaults to DEFAULT_INSTRUCTION_CLARIFICATION_TEMPLATE.
            language: Language for prompts (ZH or EN). Defaults to EN.
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.
            **kwargs: Additional arguments passed to LLMGrader.
        """
        super().__init__(
            model=model,
            name="instruction_clarification",
            mode=GraderMode.POINTWISE,
            description="Evaluate instruction clarification ability in multi-turn conversations",
            template=template or DEFAULT_INSTRUCTION_CLARIFICATION_TEMPLATE,
            language=language,
            strategy=strategy,
            **kwargs,
        )

    async def _aevaluate(
        self,
        response: str | Dict[str, Any],
        history: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate instruction clarification of the assistant's response.

        Args:
            response: The current assistant response to evaluate.
                      Can be a string or a dict with 'content' field.
            history: List of previous conversation messages in ChatMessage format.
                     Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            **kwargs: Additional arguments.

        Returns:
            GraderScore with:
                - score: 1-5 instruction clarification score
                - reason: Detailed evaluation explanation
                - metadata: Contains evaluation type and history info
        """
        try:
            # Handle response as string or dict
            if isinstance(response, dict):
                response_str = response.get("content", str(response))
            else:
                response_str = response

            history_str = format_conversation_history(history)

            result = await super()._aevaluate(
                response=response_str,
                history=history_str,
                **kwargs,
            )

            metadata = {
                "history_turns": len(history),
                "evaluation_type": "instruction_clarification",
            }

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating instruction clarification: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = [
    "InstructionClarificationGrader",
    "DEFAULT_INSTRUCTION_CLARIFICATION_TEMPLATE",
]
