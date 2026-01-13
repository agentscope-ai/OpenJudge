# -*- coding: utf-8 -*-
"""
Instruction Following Grader

Evaluates whether model response correctly follows the given instructions, including
content requirements, format constraints, style guidelines, and other specified criteria.
"""

import textwrap
from typing import Optional

from loguru import logger

from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# English Prompt
INSTRUCTION_FOLLOWING_PROMPT_EN = textwrap.dedent(
    """
You are a professional data annotator responsible for evaluating whether the model response follows the given
instructions. Your task is to score according to the following criteria:

<Scoring Criteria>
A response that perfectly follows instructions should:
- Address all required topics, questions, or tasks mentioned in the instruction.
- Follow the specified format exactly (e.g., JSON, bullet points, numbered list, essay format).
- Adhere to all constraints specified (e.g., word/sentence count, tone, style, vocabulary level).
- Include all required elements (e.g., introduction, conclusion, specific sections).
- Avoid adding information not requested by the instruction.
- Meet quality requirements if specified (e.g., "detailed", "concise", "professional").

Points should be deducted for:
- Missing required content or topics.
- Incorrect format or structure.
- Violating specified constraints (e.g., too long/short, wrong tone).
- Omitting required elements.
- Adding excessive unrequested information.
- Misinterpreting the instruction's intent.
</Scoring Criteria>

<Guidance>
- Carefully parse the instruction to identify ALL requirements and constraints.
- Break down complex instructions into individual requirements.
- Check each requirement systematically against the response.
- Consider both explicit requirements (stated clearly) and implicit requirements (strongly implied).
- Evaluate format, structure, and style requirements separately from content requirements.
- Be strict: partial fulfillment should result in lower scores.
</Guidance>

<Reminder>
The goal is to evaluate instruction-following capability, not content quality per se. A response can be well-written but
 score low if it doesn't follow instructions. Conversely, a simple response that perfectly follows all instructions
 should score high.
</Reminder>

Evaluate the following:

<instruction>
{instruction}
</instruction>

<query>
{query}
</query>

<response>
{response}
</response>

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5, where 5 means perfect instruction adherence and 1 means complete failure to
    follow instructions>,
    "reason": "<brief explanation for the assigned score, specifically mentioning which instruction requirements were
    met or violated>"
}}

Scoring Scale:
- 5: Perfect adherence to all instruction aspects
- 4: Follows most instructions with minor deviations
- 3: Partial adherence, misses some requirements
- 2: Significant instruction violations, misses major requirements
- 1: Complete failure to follow instructions or major misunderstanding

JSON:
"""
).strip()

# Chinese Prompt
INSTRUCTION_FOLLOWING_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的数据标注员，负责评估模型输出是否遵循给定的指令。你的任务是根据以下标准进行评分：

<评分标准>
完美遵循指令的回答应该：
- 涵盖指令中提到的所有必需主题、问题或任务。
- 完全遵循指定的格式（例如，JSON、项目符号、编号列表、论文格式）。
- 遵守所有指定的约束（例如，字数/句子数、语气、风格、词汇水平）。
- 包含所有必需的元素（例如，引言、结论、特定部分）。
- 避免添加指令未要求的信息。
- 满足指定的质量要求（例如，"详细"、"简洁"、"专业"）。

以下情况应扣分：
- 缺少必需的内容或主题。
- 格式或结构不正确。
- 违反指定的约束（例如，太长/太短、错误的语气）。
- 遗漏必需的元素。
- 添加过多未要求的信息。
- 误解指令的意图。
</评分标准>

<指导>
- 仔细解析指令以识别所有要求和约束。
- 将复杂的指令分解为单个要求。
- 系统地根据输出检查每个要求。
- 考虑明确的要求（清楚陈述的）和隐含的要求（强烈暗示的）。
- 将格式、结构和风格要求与内容要求分开评估。
- 严格要求：部分满足应导致较低的分数。
</指导>

<提醒>
目标是评估指令遵循能力，而不是内容质量本身。一个回答可能写得很好，但如果不遵循指令就会得分低。相反，一个简单但完美遵循所有指令的回答应该得到高分。
</提醒>

评估以下内容：

<指令>
{instruction}
</指令>

<查询>
{query}
</查询>

<回答>
{response}
</回答>

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1到5之间的整数，其中5表示完美遵循指令，1表示完全未能遵循指令>,
    "reason": "<对所给分数的简要解释，特别提到满足或违反了哪些指令要求>"
}}

评分标尺：
- 5: 完美遵循指令的所有方面
- 4: 遵循大部分指令，有轻微偏离
- 3: 部分遵循，遗漏一些要求
- 2: 明显违反指令，遗漏主要要求
- 1: 完全未能遵循指令或严重误解

JSON:
"""
).strip()


# Build default template from prompts
DEFAULT_INSTRUCTION_FOLLOWING_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=INSTRUCTION_FOLLOWING_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=INSTRUCTION_FOLLOWING_PROMPT_ZH,
            ),
        ],
    },
)


class InstructionFollowingGrader(LLMGrader):
    """
    Instruction Following Grader

    Purpose:
        Evaluates how precisely model outputs follow given instructions across content,
        format, style, and constraints. Essential for ensuring AI systems execute tasks
        as specified without deviation.

    What it evaluates:
        - Content Relevance: Addresses all required topics and questions
        - Format Compliance: Follows specified format (JSON, bullet points, essay, etc.)
        - Constraint Adherence: Satisfies length, tone, style requirements
        - Completeness: Fulfills all instruction requirements
        - Precision: Avoids adding unrequested information
        - Structural Accuracy: Maintains requested organization

    Distinction from RelevanceGrader:
        - InstructionFollowingGrader: Evaluates **adherence to instructions** - Did the model
          do what it was asked to do? Focuses on following specific requirements, formats,
          and constraints.
        - RelevanceGrader: Evaluates **relevance to the query** - Did the model address
          the user's question? Focuses on whether the response is on-topic and appropriately
          addresses the user's needs.

        Example: If asked "Write 3 bullet points about AI", a response with 5 well-written
        paragraphs would score low on instruction following (wrong format, wrong count) but
        could score high on relevance (addresses the AI topic).

    When to use:
        - Structured output generation (JSON, XML, specific formats)
        - Task completion verification in agent systems
        - Evaluating instruction-tuned models
        - Quality control for templated content generation
        - Testing AI assistants' ability to follow complex instructions

    Scoring:
        - 5: Perfect adherence to all instruction aspects
        - 4: Follows most instructions with minor deviations
        - 3: Partial adherence, misses some requirements
        - 2: Significant instruction violations, misses major requirements
        - 1: Complete failure to follow instructions or major misunderstanding

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_INSTRUCTION_FOLLOWING_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect adherence, 1 = complete failure
            - reason: Detailed analysis of adherence and violations
            - metadata: Threshold and evaluation details

    Example:
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.common.instruction_following import InstructionFollowingGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = InstructionFollowingGrader(model=model, threshold=0.7)
        >>>
        >>> # Good adherence
        >>> result = asyncio.run(grader.aevaluate(
        ...     instruction="Write exactly 3 sentences in formal academic tone.",
        ...     response="Climate change poses serious risks. Research shows rising temperatures. "
        ...              "Action is urgently needed."
        ... ))
        >>> print(result.score)  # 5 - follows all requirements
        >>>
        >>> # Poor adherence
        >>> result = asyncio.run(grader.aevaluate(
        ...     instruction="Write a 3-sentence summary in formal tone about climate change.",
        ...     response="Climate change is a big problem. It's getting hotter. We need to act now!",
        ...     query="Summarize the climate situation."
        ... ))
        >>> print(result.score)  # 2 - informal tone, poor structure
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_INSTRUCTION_FOLLOWING_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize InstructionFollowingGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_INSTRUCTION_FOLLOWING_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="instruction_following",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether response follows the given instructions",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        instruction: str,
        response: str,
        query: str = "",
    ) -> GraderScore:
        """
        Evaluate instruction following in response

        Args:
            instruction: The instruction or prompt given to the model
            response: Model response to evaluate
            query: Original user query or question. Defaults to empty string.

        Returns:
            GraderScore: Score with instruction following value [1, 5]
                        where 5 means perfect adherence, 1 means complete failure

        Example:
            >>> result = await grader.aevaluate(
            ...     instruction="Write exactly 3 bullet points about AI safety.",
            ...     response="• AI safety is important\\n• We need alignment research\\n• Testing is crucial",
            ... )
        """
        try:
            result = await super().aevaluate(
                instruction=instruction,
                response=response,
                query=query,
            )
            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata={"threshold": self.threshold},
            )

        except Exception as e:
            logger.error(f"Error evaluating instruction following: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["InstructionFollowingGrader", "DEFAULT_INSTRUCTION_FOLLOWING_TEMPLATE"]
