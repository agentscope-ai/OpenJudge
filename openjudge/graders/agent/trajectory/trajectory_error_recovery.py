# -*- coding: utf-8 -*-
"""
Trajectory Error Recovery Grader

Evaluates whether the agent properly recovers from errors in its trajectory —
recognizing failures, diagnosing causes, and adapting its strategy rather than
repeating the same failed approach.
"""

import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
TRAJECTORY_ERROR_RECOVERY_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing agent trajectory error recovery behavior. Your task is to evaluate whether the agent properly recovers from errors encountered during task execution. Good error recovery means the agent recognizes failures, diagnoses root causes, and adapts its strategy rather than blindly repeating the same failed approach.

<Rubrics>
1. The agent recognizes when an error or failure has occurred (does not ignore error signals)
2. The agent correctly identifies the type and cause of the error (e.g., wrong parameters, missing prerequisites, tool unavailability)
3. The agent adapts its strategy after encountering an error (changes approach instead of repeating the same action)
4. The agent's recovery plan addresses the root cause of the error (not just symptoms)
5. The agent does not repeat the exact same failed action without modification
6. The agent shows progressive recovery — each retry attempts a different approach or fixes a different aspect
7. The agent knows when to give up on a failed approach and try an alternative path
</Rubrics>

<Steps>
1. Identify errors: Find all instances where tool calls or actions resulted in errors or failures
2. Check error recognition: Did the agent acknowledge each error in its reasoning?
3. Assess diagnosis: Did the agent correctly identify why the error occurred?
4. Evaluate adaptation: Did the agent change its approach after the error?
5. Check for repeated failures: Did the agent repeat the same failed action?
6. Assess recovery success: Did the adapted approach ultimately resolve the issue?
</Steps>

<Scale>
- **Score 1.0**: Good error recovery — The agent recognizes errors, diagnoses causes, and adapts strategy effectively
- **Score 0.0**: Poor error recovery — The agent ignores errors, repeats failed actions, or fails to adapt
</Scale>

<Context (Optional)>
{context}
</Context>

<Agent Trajectory>
{messages}
</Agent Trajectory>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of error recovery behavior, including which errors were encountered and how the agent responded>",
    "score": <0.0 or 1.0>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
TRAJECTORY_ERROR_RECOVERY_PROMPT_ZH = textwrap.dedent(
    """你是一名分析智能体轨迹错误恢复行为的专家。你的任务是评估智能体是否正确地从任务执行过程中遇到的错误中恢复。良好的错误恢复意味着智能体能够识别失败、诊断根本原因，并调整策略，而不是盲目重复相同的失败方法。

<评分标准>
1. 智能体识别到错误或失败的发生（不会忽略错误信号）
2. 智能体正确识别了错误的类型和原因（例如，参数错误、缺少前提条件、工具不可用）
3. 智能体在遇到错误后调整了策略（改变了方法而不是重复相同的动作）
4. 智能体的恢复计划针对了错误的根本原因（而不仅仅是症状）
5. 智能体没有未经修改地重复完全相同的失败动作
6. 智能体展示了渐进式恢复——每次重试都尝试不同的方法或修复不同的方面
7. 智能体知道何时放弃失败的方法并尝试替代路径
</评分标准>

<评估步骤>
1. 识别错误：找出所有工具调用或动作导致错误或失败的实例
2. 检查错误识别：智能体是否在推理中承认了每个错误？
3. 评估诊断：智能体是否正确识别了错误发生的原因？
4. 评估适应：智能体在错误后是否改变了方法？
5. 检查重复失败：智能体是否重复了相同的失败动作？
6. 评估恢复成功：调整后的方法是否最终解决了问题？
</评估步骤>

<评分量表>
- **分数 1.0**：良好的错误恢复 — 智能体识别错误、诊断原因并有效调整策略
- **分数 0.0**：糟糕的错误恢复 — 智能体忽略错误、重复失败动作或未能调整
</评分量表>

<上下文（可选）>
{context}
</上下文>

<智能体轨迹>
{messages}
</智能体轨迹>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于错误恢复行为的详细解释，包括遇到的错误和智能体的应对方式>",
    "score": <0.0 或 1.0>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_TRAJECTORY_ERROR_RECOVERY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=TRAJECTORY_ERROR_RECOVERY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=TRAJECTORY_ERROR_RECOVERY_PROMPT_ZH,
            ),
        ],
    },
)


class TrajectoryErrorRecoveryGrader(LLMGrader):
    """
    Trajectory Error Recovery Grader

    Evaluates whether the agent properly recovers from errors in its trajectory —
    recognizing failures, diagnosing causes, and adapting its strategy.

    Required modules: messages (full trajectory)

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.EN)

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.models.schema.prompt_template import LanguageEnum
        >>>
        >>> api = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>> grader = TrajectoryErrorRecoveryGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=[
        ...         {"role": "user", "content": "Search for the file"},
        ...         {"role": "assistant", "tool_calls": [{"function": {"name": "search", "arguments": '{"path": "/wrong"}'}}]},
        ...         {"role": "tool", "content": "Error: path not found"},
        ...         {"role": "assistant", "content": "The path was incorrect. Let me try the home directory.", "tool_calls": [{"function": {"name": "search", "arguments": '{"path": "/home"}'}}]},
        ...     ]
        ... ))
        >>> print(f"Score: {result.score}")
    """

    DEFAULT_TEMPLATE = DEFAULT_TRAJECTORY_ERROR_RECOVERY_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize TrajectoryErrorRecoveryGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectStrategy.
        """
        super().__init__(
            name="trajectory_error_recovery",
            mode=GraderMode.POINTWISE,
            description="Evaluate agent trajectory error recovery behavior",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Format messages into a readable string for evaluation."""
        import json

        messages = [msg.get("message", msg) for msg in messages]
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if isinstance(content, str):
                content_str = content
            elif isinstance(content, list):
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content_str = " ".join(text_parts) if text_parts else ""
            else:
                content_str = str(content)

            msg_str = f"[{role}]"
            if content_str:
                msg_str += f" {content_str}"
            if tool_calls:
                tool_calls_str = json.dumps(tool_calls, indent=2, ensure_ascii=False)
                msg_str += f"\nTool Calls: {tool_calls_str}"

            formatted_parts.append(msg_str)

        return "\n\n".join(formatted_parts)

    async def _aevaluate(
        self,
        messages: List[Dict[str, Any]],
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate trajectory error recovery behavior.

        Args:
            messages: Full trajectory of agent interactions
            context: Optional task context

        Returns:
            GraderScore: Score indicating error recovery quality
        """
        if not messages:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No messages provided to evaluate",
                metadata={"error": "No messages provided"},
            )

        formatted_messages = self._format_messages(messages)
        context_str = context if context else ""

        try:
            result = await super()._aevaluate(
                messages=formatted_messages,
                context=context_str,
            )
            score = result.score
            reason = result.reason

            # Ensure score is binary (0.0 or 1.0)
            normalized_score = 1.0 if score > 0.5 else 0.0

        except Exception as e:
            logger.error(f"Error evaluating trajectory error recovery: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        metadata = {
            "raw_score": score,
            "evaluation_type": "trajectory_error_recovery",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "TrajectoryErrorRecoveryGrader",
    "DEFAULT_TRAJECTORY_ERROR_RECOVERY_TEMPLATE",
]
