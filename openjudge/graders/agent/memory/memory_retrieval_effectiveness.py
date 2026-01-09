# -*- coding: utf-8 -*-
"""
Memory Retrieval Effectiveness Grader

Evaluates whether the agent effectively retrieves relevant information from memory when needed.
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
MEMORY_RETRIEVAL_EFFECTIVENESS_PROMPT_EN = textwrap.dedent(
    """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent effectively retrieves relevant information from memory when needed.

<Evaluation Type: Memory Retrieval Effectiveness>
The agent should effectively retrieve information from memory when needed, access information that is present, and use current and correct information. This leads to the agent making well-informed plans and avoiding repetition of past actions.
</Evaluation Type>

<Rubrics for Evaluation>
1. The agent's plan incorporates relevant information from memory based on previous observations
2. The agent avoids repeating actions that were already tried, showing awareness of past attempts
3. The agent utilizes information that was already discovered and should be in memory
4. The agent's plan is consistent with facts that are stored in memory from earlier steps
5. The agent retrieves current and correct information rather than outdated or incorrect memory state
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good retrieval effectiveness patterns described in each rubric
2. Focus on relevant modules: Only consider plan, observation, and memory modules
3. Provide evidence-based reasoning: Explain how memory retrieval is effective and why
4. Assess confidence: Rate your confidence based on how clearly the effectiveness is exhibited
</Evaluation Criteria>

{context}

{history}

<Current Step>
Plan: {plan}
Observation: {observation}
Memory: {memory}
</Current Step>

# Scoring Instructions
- If memory retrieval is effective: score = 1.0 (good retrieval)
- If memory retrieval is ineffective: score = 0.0 (poor retrieval)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of memory retrieval effectiveness and confidence level>"
}}

JSON:
"""
).strip()

# Chinese Prompt
MEMORY_RETRIEVAL_EFFECTIVENESS_PROMPT_ZH = textwrap.dedent(
    """
你是一名分析智能体行为的专家。你的任务是评估智能体在需要时是否有效地从记忆中检索相关信息。

<评估类型：记忆检索有效性>
智能体应该在需要时有效地从记忆中检索信息，访问存在的信息，并使用当前且正确的信息。这导致智能体制定明智的计划并避免重复过去的动作。
</评估类型>

<评估准则>
1. 智能体的计划结合了基于先前观察的记忆中的相关信息
2. 智能体避免重复已经尝试过的动作，显示对过去尝试的意识
3. 智能体利用了已经发现并应该在记忆中的信息
4. 智能体的计划与早期步骤中存储在记忆中的事实一致
5. 智能体检索当前且正确的信息，而不是过时或错误的记忆状态
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好检索有效性模式
2. 关注相关模块：仅考虑计划、观察和记忆模块
3. 提供基于证据的推理：解释记忆检索如何有效以及原因
4. 评估置信度：根据有效性表现的清晰程度评估你的置信度
</评估标准>

{context}

{history}

<当前步骤>
计划：{plan}
观察：{observation}
记忆：{memory}
</当前步骤>

# 评分指令
- 如果记忆检索有效：score = 1.0（良好检索）
- 如果记忆检索无效：score = 0.0（检索不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于记忆检索有效性的详细解释和置信度水平>"
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_MEMORY_RETRIEVAL_EFFECTIVENESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=MEMORY_RETRIEVAL_EFFECTIVENESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=MEMORY_RETRIEVAL_EFFECTIVENESS_PROMPT_ZH,
            ),
        ],
    },
)


class MemoryRetrievalEffectivenessGrader(LLMGrader):
    """
    Memory Retrieval Effectiveness Grader

    Evaluates whether the agent effectively retrieves relevant information from memory when needed.

    Required modules: plan, observation, memory

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
        >>> grader = MemoryRetrievalEffectivenessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="You see a closed cabinet.",
        ...     memory="The cabinet is closed."
        ... ))
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_MEMORY_RETRIEVAL_EFFECTIVENESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="memory_retrieval_effectiveness",
            mode=GraderMode.POINTWISE,
            description="Evaluate memory retrieval effectiveness",
            model=model,
            template=template or DEFAULT_MEMORY_RETRIEVAL_EFFECTIVENESS_TEMPLATE,
            language=language,
        )

    async def aevaluate(
        self,
        plan: str,
        observation: str,
        memory: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate memory retrieval effectiveness

        Args:
            plan: Agent's planning/reasoning
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = effective, 0.0 = ineffective)

        Example:
            >>> result = await grader.aevaluate(
            ...     plan="I will use the key from drawer 1.",
            ...     observation="You are standing in the room.",
            ...     memory="The key was found in drawer 1 in step 3.",
            ...     context="Task: Find and use the key"
            ... )
        """
        # Format context section
        context_str = f"<context>\n{context}\n</context>" if context else ""

        # Format history
        history_str = format_history(history)

        try:
            result = await super().aevaluate(
                plan=plan,
                observation=observation,
                memory=memory,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating memory retrieval effectiveness: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "memory_retrieval_effectiveness",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "MemoryRetrievalEffectivenessGrader",
    "DEFAULT_MEMORY_RETRIEVAL_EFFECTIVENESS_TEMPLATE",
]
