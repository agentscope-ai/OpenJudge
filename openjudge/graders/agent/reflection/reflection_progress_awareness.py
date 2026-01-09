# -*- coding: utf-8 -*-
"""
Reflection Progress Awareness Grader

Evaluates whether the agent demonstrates accurate awareness of progress toward completing the task
in its reflection.
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
REFLECTION_PROGRESS_AWARENESS_PROMPT_EN = textwrap.dedent(
    """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent demonstrates accurate awareness of progress toward completing the task in its reflection.

<Evaluation Type: Reflection Progress Awareness>
The agent should demonstrate accurate awareness of progress toward completing the task in its reflection. This includes correctly estimating progress and recognizing whether the agent is making forward progress or stuck in a loop.
</Evaluation Type>

<Rubrics for Evaluation>
1. The agent accurately assesses progress when making forward advancement
2. The agent correctly recognizes when it is stuck in a loop or repeating actions
3. The agent realistically estimates how close it is to task completion based on current state
4. The agent appropriately evaluates the progress made when observations show advancement
5. The agent demonstrates awareness of whether it is repeating failed attempts
6. The agent correctly identifies all required sub-goals from the task description
7. The agent tracks progress for every mandatory step or constraint specified in the task
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good progress awareness patterns described in each rubric
2. Focus on relevant modules: Only consider observation and reflection modules
3. Provide evidence-based reasoning: Explain how the reflection demonstrates progress awareness and why
4. Assess confidence: Rate your confidence based on how clearly the awareness is exhibited
5. Task Decomposition Check: Verify the agent identifies all sub-goals from the task description
6. Keyword Sensitivity: Check if the agent recognizes key action verbs and object requirements in the task
7. Progress Overestimation Check: Detect if the agent claims to be "close" or "almost done" while ignoring critical unfinished sub-goals
</Evaluation Criteria>

{context}

{history}

<Current Step>
Observation: {observation}
Reflection: {reflection}
</Current Step>

# Scoring Instructions
- If progress awareness is accurate: score = 1.0 (good awareness)
  * Agent correctly identifies what has been accomplished
  * Agent accurately assesses distance to goal completion
  * Agent recognizes all required sub-goals from task description
  * Agent does not overestimate progress while ignoring critical unfinished steps

- If progress awareness is inaccurate: score = 0.0 (poor awareness)
  * Agent misjudges current progress or proximity to completion
  * Agent overlooks critical sub-goals mentioned in the task
  * Agent claims to be "close" or "almost done" while major requirements remain unmet
  * Agent overestimates completion by ignoring mandatory task constraints

Critical Note on Task Requirements:
- If the task specifies particular objects, the agent MUST use exactly those objects. Considering alternatives indicates poor progress awareness and failure to understand task constraints.
- If the task contains multiple sub-goals, the agent must track progress toward ALL sub-goals. Omitting any sub-goal leads to score = 0.0.
- Agent must respect exact task specifications. Substituting similar items or skipping mentioned steps indicates fundamental misunderstanding of task progress.

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of progress awareness quality and confidence level>"
}}

JSON:
"""
).strip()

# Chinese Prompt
REFLECTION_PROGRESS_AWARENESS_PROMPT_ZH = textwrap.dedent(
    """
你是一名分析智能体行为的专家。你的任务是评估智能体是否在其反思中展示了对完成任务进度的准确意识。

<评估类型：反思进度意识>
智能体应该在其反思中展示对完成任务进度的准确意识。这包括正确估计进度并识别智能体是在向前推进还是陷入循环。
</评估类型>

<评估准则>
1. 智能体在向前推进时准确评估进度
2. 智能体在陷入循环或重复动作时正确识别
3. 智能体根据当前状态现实地估计距离任务完成的接近程度
4. 智能体在观察显示进展时适当评估所取得的进度
5. 智能体展示了对是否正在重复失败尝试的意识
6. 智能体正确识别出任务描述中所有要求的子目标
7. 智能体跟踪任务中指定的每个强制步骤或约束的进度
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好进度意识模式
2. 关注相关模块：仅考虑观察和反思模块
3. 提供基于证据的推理：解释反思如何展示进度意识以及原因
4. 评估置信度：根据意识表现的清晰程度评估你的置信度
5. 任务分解检查：验证智能体是否识别出任务描述中的所有子目标
6. 关键词敏感性：检查智能体是否识别任务中的关键动作动词和对象要求
7. 进度高估检查：检测智能体是否在忽略关键未完成子目标的情况下声称"接近"或"几乎完成"
</评估标准>

{context}

{history}

<当前步骤>
观察：{observation}
反思：{reflection}
</当前步骤>

# 评分指令
- 如果进度意识准确：score = 1.0（良好意识）
  * 智能体正确识别已完成的内容
  * 智能体准确评估距离目标完成的距离
  * 智能体识别出任务描述中所有要求的子目标
  * 智能体在忽略关键未完成步骤的情况下不会高估进度

- 如果进度意识不准确：score = 0.0（意识不佳）
  * 智能体误判当前进度或接近完成的程度
  * 智能体忽略任务中提到的关键子目标
  * 智能体在主要要求仍未满足时声称"接近"或"几乎完成"
  * 智能体通过忽略强制任务约束来高估完成度

关于任务要求的重要说明：
- 如果任务指定了特定对象，智能体必须使用完全相同的对象。考虑替代品表明进度意识不佳且未能理解任务约束。
- 如果任务包含多个子目标，智能体必须跟踪所有子目标的进度。遗漏任何子目标导致评分 = 0.0。
- 智能体必须尊重确切的任务规范。替换类似物品或跳过提到的步骤表明对任务进度的根本误解。

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于进度意识质量的详细解释和置信度水平>"
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=REFLECTION_PROGRESS_AWARENESS_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=REFLECTION_PROGRESS_AWARENESS_PROMPT_ZH,
            ),
        ],
    },
)


class ReflectionProgressAwarenessGrader(LLMGrader):
    """
    Reflection Progress Awareness Grader

    Evaluates whether the agent demonstrates accurate awareness of progress toward completing the task
    in its reflection.

    Required modules: observation, reflection

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
        >>> grader = ReflectionProgressAwarenessGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="Cabinet 1 now has apples. Task complete.",
        ...     reflection="Good progress! I've successfully found the apples.",
        ...     context="Task: Find apples in cabinets"
        ... ))
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="reflection_progress_awareness",
            mode=GraderMode.POINTWISE,
            description="Evaluate reflection progress awareness",
            model=model,
            template=template or DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE,
            language=language,
        )

    async def aevaluate(
        self,
        observation: str,
        reflection: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate reflection progress awareness

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good awareness, 0.0 = poor awareness)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="Cabinet 1 now has apples. Task complete.",
            ...     reflection="Good progress! I've successfully found the apples.",
            ...     context="Task: Find apples in cabinets"
            ... )
        """
        # Format context section
        context_str = f"<context>\n{context}\n</context>" if context else ""

        # Format history
        history_str = format_history(history)

        try:
            result = await super().aevaluate(
                observation=observation,
                reflection=reflection,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating reflection progress awareness: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "reflection_progress_awareness",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReflectionProgressAwarenessGrader",
    "DEFAULT_REFLECTION_PROGRESS_AWARENESS_TEMPLATE",
]
