# -*- coding: utf-8 -*-
"""
Plan Feasibility Grader

Evaluates whether the agent creates a plan that is logically sound and feasible.
"""

import textwrap
from typing import Any, Optional

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
PLAN_FEASIBILITY_PROMPT_EN = """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent creates a plan that is logically sound and feasible.

<Evaluation Type: Plan Feasibility>
The agent should create plans that are logically sound and feasible, respecting causal logic, specifying actions in correct order, and proposing executable actions. The plan should respect physical constraints, logical prerequisites, and the current state of the environment.
</Evaluation Type>

<Rubrics for Evaluation>
1. The plan respects causal logic (e.g., obtaining an object before using it)
2. The plan specifies actions in a feasible order (e.g., opening before closing)
3. The plan proposes actions that can be executed given the current environment state
4. The plan accounts for necessary preconditions or prerequisites for actions
5. The plan contains logically consistent steps and goals
</Rubrics>

<Evaluation Criteria>
For your analysis:
1. Apply each rubric: Check if the step demonstrates good feasibility patterns described in each rubric
2. Focus on relevant modules: Only consider plan, observation, and memory modules
3. Provide evidence-based reasoning: Explain how the plan demonstrates feasibility and why
4. Assess confidence: Rate your confidence based on how clearly the feasibility is exhibited
</Evaluation Criteria>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# Scoring Instructions
- If the plan is feasible and logically sound: score = 1.0 (good feasibility)
- If the plan has feasibility issues: score = 0.0 (poor feasibility)

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of plan feasibility and confidence level>"
}}

JSON:
"""

# Chinese Prompt
PLAN_FEASIBILITY_PROMPT_ZH = """
你是一名分析智能体行为的专家。你的任务是评估智能体是否创建了逻辑上合理且可行的计划。

<评估类型：计划可行性>
智能体应该创建逻辑上合理且可行的计划，尊重因果逻辑、以正确的顺序指定动作，并提出可执行的动作。该计划应该尊重物理约束、逻辑前提和环境的当前状态。
</评估类型>

<评估准则>
1. 计划尊重因果逻辑（例如，在使用对象之前获得它）
2. 计划以可行的顺序指定动作（例如，在关闭之前打开）
3. 计划提出了在当前环境状态下可以执行的动作
4. 计划考虑了动作的必要前提条件或先决条件
5. 计划包含逻辑上一致的步骤和目标
</评估准则>

<评估标准>
进行分析时：
1. 应用每个准则：检查步骤是否展示了每个准则中描述的良好可行性模式
2. 关注相关模块：仅考虑计划、观察和记忆模块
3. 提供基于证据的推理：解释计划如何展示可行性以及原因
4. 评估置信度：根据可行性表现的清晰程度评估你的置信度
</评估标准>

{context_section}

<trajectory_steps>
{trajectory_steps}
</trajectory_steps>

# 评分指令
- 如果计划可行且逻辑合理：score = 1.0（良好可行性）
- 如果计划存在可行性问题：score = 0.0（可行性不佳）

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于计划可行性的详细解释和置信度水平>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_PLAN_FEASIBILITY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(PLAN_FEASIBILITY_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(PLAN_FEASIBILITY_PROMPT_ZH),
            ),
        ],
    },
)


class PlanFeasibilityGrader(LLMGrader):
    """
    Plan Feasibility Grader

    Evaluates whether the agent creates a plan that is logically sound and feasible.

    Required modules: plan, observation, memory

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.schema.template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = PlanFeasibilityGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     plan="I will first open the drawer to get the key, then use it to unlock the door.",
        ...     observation="The drawer is closed. You don't have any items.",
        ...     memory="The key is inside the drawer."
        ... )
        >>> print(f"Score: {result.score}")  # 1.0 (feasible plan)
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_PLAN_FEASIBILITY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="plan_feasibility",
            mode=GraderMode.POINTWISE,
            description="Evaluate plan feasibility",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_PLAN_FEASIBILITY_TEMPLATE

    def _format_trajectory_steps(
        self,
        plan: str,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
    ) -> str:
        """Format trajectory steps for evaluation.

        Args:
            plan: Agent's planning/reasoning
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history_steps: Optional list of previous step dictionaries

        Returns:
            Formatted trajectory string
        """
        lines = []

        # Add history steps if provided
        if history_steps:
            for i, hist_step in enumerate(history_steps):
                lines.append(f"Step {i + 1}:")
                for key, value in hist_step.items():
                    if value:
                        lines.append(f"{key.capitalize()}: {value}")
                lines.append("")

        # Add current step
        step_number = len(history_steps) + 1 if history_steps else 1
        lines.append(f"Step {step_number}:")
        lines.append(f"Plan: {plan}")
        lines.append(f"Observation: {observation}")
        lines.append(f"Memory: {memory}")

        return "\n".join(lines)

    async def aevaluate(
        self,
        plan: str,
        observation: str,
        memory: str,
        history_steps: Optional[list] = None,
        task_context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate plan feasibility

        Args:
            plan: Agent's planning/reasoning
            observation: Agent's observation from the environment
            memory: Agent's memory content
            history_steps: Optional list of previous step dictionaries for context
            task_context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = feasible, 0.0 = infeasible)

        Example:
            >>> result = await grader.aevaluate(
            ...     plan="I will first open the drawer to get the key, then use it.",
            ...     observation="The drawer is closed. You don't have any items.",
            ...     memory="The key is inside the drawer.",
            ...     task_context="Task: Unlock the door"
            ... )
        """
        # Format trajectory steps
        trajectory_steps = self._format_trajectory_steps(
            plan=plan,
            observation=observation,
            memory=memory,
            history_steps=history_steps,
        )

        # Prepare context section
        context_section = ""
        if task_context:
            context_section = f"""<task_context>
{task_context}
</task_context>"""

        try:
            result = await super().aevaluate(
                trajectory_steps=trajectory_steps,
                context_section=context_section,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating plan feasibility: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "plan_feasibility",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "PlanFeasibilityGrader",
    "DEFAULT_PLAN_FEASIBILITY_TEMPLATE",
]
