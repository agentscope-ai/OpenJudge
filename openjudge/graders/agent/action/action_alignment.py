# -*- coding: utf-8 -*-
"""
Action Alignment Grader

Evaluates whether the agent executes an action that aligns with its stated plan or reasoning.
"""

import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.graders.agent.utils import format_history
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
ACTION_ALIGNMENT_PROMPT_EN = textwrap.dedent(
    """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent executes an action that aligns with its stated plan or reasoning.

<Evaluation Type: Action Alignment>
The agent should execute actions that are consistent with its stated plan or reasoning. The action should follow logically from the plan and reflection, demonstrating good alignment between the agent's planning and execution modules.
</Evaluation Type>

<Rubrics for Evaluation>
1. The action directly implements the stated plan (e.g., plan says "open" and action is "open")
2. The action targets the correct object specified in the plan
3. The action contributes to achieving the goal stated in the plan
4. The action sequence follows the logical order outlined in the plan
5. The action respects important preconditions or constraints mentioned in the plan
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good alignment patterns described in each rubric
2. Focus on relevant modules: Only consider plan and action modules
3. Provide evidence-based reasoning: Explain how the step demonstrates alignment and why
4. Assess confidence: Rate your confidence based on how clearly the alignment is exhibited
</Evaluation Criteria>

{context}

{history}

<Current Step>
Plan: {plan}
Action: {action}
</Current Step>

# Scoring Instructions
- If the action aligns well with the plan: score = 1.0 (good alignment)
- If the action does not align with the plan: score = 0.0 (poor alignment)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of action-plan alignment and confidence level>"
}}

JSON:
"""
).strip()

# Chinese Prompt
ACTION_ALIGNMENT_PROMPT_ZH = textwrap.dedent(
    """
你是一名分析智能体行为的专家。你的任务是评估智能体是否执行了与其声明的计划或推理一致的动作。

<评估类型：动作对齐>
智能体应该执行与其声明的计划或推理一致的动作。该动作应该符合计划和反思的逻辑，表明智能体的计划和执行模块之间有良好的对齐。
</评估类型>

<评估准则>
1. 动作直接实现了声明的计划（例如，计划说"打开"动作就是"打开"）
2. 动作针对计划中指定的正确对象
3. 动作有助于实现计划中声明的目标
4. 动作序列遵循计划中概述的逻辑顺序
5. 动作尊重计划中提到的重要前提条件或约束
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好对齐模式
2. 关注相关模块：仅考虑计划和动作模块
3. 提供基于证据的推理：解释步骤如何展示对齐以及原因
4. 评估置信度：根据对齐表现的清晰程度评估你的置信度
</评估标准>

{context}

{history}

<当前步骤>
计划：{plan}
动作：{action}
</当前步骤>

# 评分指令
- 如果动作与计划很好地对齐：score = 1.0（良好对齐）
- 如果动作与计划不对齐：score = 0.0（对齐不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于动作-计划对齐的详细解释和置信度水平>"
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_ACTION_ALIGNMENT_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=ACTION_ALIGNMENT_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=ACTION_ALIGNMENT_PROMPT_ZH,
            ),
        ],
    },
)


class ActionAlignmentGrader(LLMGrader):
    """
    Action Alignment Grader

    Evaluates whether the agent executes an action that aligns with its stated plan or reasoning.

    Required modules: plan, action

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> import asyncio
        >>> from openjudge.model.openai_llm import OpenAIChatModel
        >>> from openjudge.models.schema.prompt_template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = ActionAlignmentGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     plan="I will open drawer 1 to find the key.",
        ...     action="open drawer 1"
        ... )
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_ACTION_ALIGNMENT_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize ActionAlignmentGrader.

        Args:
            model: The chat model to use for evaluation, either as a BaseChatModel instance or config dict
            template: The prompt template for action alignment evaluation.
                     Defaults to DEFAULT_ACTION_ALIGNMENT_TEMPLATE.
            language: The language for the evaluation prompt. Defaults to LanguageEnum.EN.
        """
        super().__init__(
            name="action_alignment",
            mode=GraderMode.POINTWISE,
            description="Evaluate action alignment with plan",
            model=model,
            template=template or DEFAULT_ACTION_ALIGNMENT_TEMPLATE,
            language=language,
        )

    async def aevaluate(
        self,
        plan: str,
        action: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> GraderScore:
        """
        Evaluate action alignment with plan

        Args:
            plan: Agent's planning/reasoning
            action: Agent's chosen action
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)

        Returns:
            GraderScore: Score with binary value (1.0 = good alignment, 0.0 = poor alignment)

        Example:
            >>> result = await grader.aevaluate(
            ...     plan="I will open drawer 1 to find the key.",
            ...     action="open drawer 1",
            ...     context="Task: Find the key"
            ... )
        """
        # Format context section
        context_str = f"<context>\n{context}\n</context>" if context else ""

        # Format history
        history_str = format_history(history)

        try:
            result = await super().aevaluate(
                plan=plan,
                action=action,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating action alignment: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "action_alignment",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )
