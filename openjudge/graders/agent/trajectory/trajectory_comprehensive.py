# -*- coding: utf-8 -*-
"""
Trajectory Comprehensive Grader for Multi-Step Tool Call Evaluation

This module provides comprehensive evaluation for agent trajectories,
assessing each step's contribution and the overall problem-solving capability.
"""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.oai.response import ChatResponse
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long,too-many-statements

# Chinese Prompt
TRAJECTORY_COMPREHENSIVE_PROMPT_ZH = textwrap.dedent(
    """# 任务描述

你是一位专业的评估专家，负责评估智能体轨迹中每个工具调用步骤对问题解决的贡献度。

你需要为轨迹中的每个工具调用步骤提供独立且一致的评估，考虑它们各自对问题解决的贡献度以及相对重要性。

## 步骤级评估（每个步骤的详细评估）

对于每个工具调用步骤，请提供以下分数（1-5的整数）：

- **step_reason**: 详细的评估理由
- **contribution_score**: 对解决问题的整体贡献
- **relevance_score**: 与用户查询的相关性
- **accuracy_score**: 获取信息的准确性
- **efficiency_score**: 此步骤的效率

# 用户查询

{query}

# 完整轨迹

{messages}

# 最终答案

{response}

# 评估指南

## 各维度评分框架

### 贡献度评分 (contribution_score)
评估此步骤对解决用户问题的整体贡献程度：
- **5分**: 关键贡献，解决用户问题不可或缺的核心步骤，若缺失则任务必然失败
- **4分**: 重要贡献，显著推进问题解决，提供关键中间结果，虽非唯一路径，但极大提升成功率或准确性
- **3分**: 一般贡献，对问题解决有辅助作用，但非关键，其信息可被省略或替代
- **2分**: 轻微贡献，几乎不影响问题解决，属于冗余、重复或低价值操作，可安全移除
- **1分**: 无效或负贡献，与用户问题完全无关，或者导致错误，阻碍问题解决

### 相关性评分 (relevance_score)
评估此步骤与用户查询的相关程度：
- **5分**: 高度相关，直接、必要地服务于用户查询的核心意图，不可或缺的关键步骤
- **4分**: 较为相关，服务于用户查询的合理子目标，虽非最直接路径，但属于有效步骤
- **3分**: 部分相关，与用户查询存在间接或边缘关联，但非必要
- **2分**: 基本不相关，与用户查询无实质性关联，属于明显偏离或误判意图的行为
- **1分**: 完全无关，与用户查询毫无逻辑联系

### 准确性评分 (accuracy_score)
评估此步骤获取或处理信息的准确程度：
- **5分**: 完全准确，完整、精确、无失真地获取并保留了最终答案需要的关键信息
- **4分**: 基本准确，核心信息准确，但存在非关键性省略或格式归一化，不影响信息本质
- **3分**: 部分准确，存在关键遗漏或失真，信息包含正确部分，但遗漏关键限定条件、状态或字段，可能导致误解
- **2分**: 基本错误，信息严重失真，关键信息被错误获取或颠倒，与最终答案明显矛盾
- **1分**: 完全虚构，信息不真实或与最终答案不相关

### 效率评分 (efficiency_score)
评估此步骤的执行效率和必要性：
- **5分**: 高效必要，步骤直接、精准且不可省略
- **4分**: 较为高效，步骤必须执行，但存在轻微低效（如非最优工具调用，参数冗余等）
- **3分**: 效率一般，步骤非任务所需，存在冗余，不影响结果
- **2分**: 低效非必要，步骤存在冗余，可以优化或者移除
- **1分**: 完全多余，步骤完全不必要或严重浪费资源，甚至导致任务失败

# 重要说明

- 识别任何冗余或不必要的工具调用，同时注意可能改善答案质量的缺失工具调用
- 确保评分既反映对解决用户问题的实际贡献，也反映在支持事实准确性方面的作用
- 使用步骤索引将您的评估与正确的步骤匹配

# 输出要求

请以JSON格式输出评估结果：
{{
    "step_evaluations": [
        {{
            "step_index": <int - 步骤索引，从0开始>,
            "step_reason": "<此步骤的详细评估理由>",
            "contribution_score": <int (1-5)>,
            "relevance_score": <int (1-5)>,
            "accuracy_score": <int (1-5)>,
            "efficiency_score": <int (1-5)>
        }}
    ]
}}

JSON:
"""
).strip()

# English Prompt
TRAJECTORY_COMPREHENSIVE_PROMPT_EN = textwrap.dedent(
    """# Task Description

You are a professional evaluation expert responsible for assessing the contribution of each tool call step in an agent trajectory.

You need to provide independent and consistent evaluation for each tool call step in the trajectory, considering their respective contributions and relative importance.

## Step-Level Evaluation (Detailed evaluation for each step)

For each tool call step, please provide the following scores (integer from 1-5):

- **step_reason**: Detailed evaluation reasoning
- **contribution_score**: Overall contribution to solving the problem
- **relevance_score**: Relevance to the user query
- **accuracy_score**: Accuracy of information obtained
- **efficiency_score**: Efficiency of this step

# User Query

{query}

# Complete Trajectory

{messages}

# Final Answer

{response}

# Evaluation Guidelines

## Scoring Framework for Each Dimension

### Contribution Score (contribution_score)
Evaluate how much this step contributes to solving the user's problem:
- **5**: Critical contribution, an indispensable core step in solving the user's problem, its absence would inevitably lead to task failure.
- **4**: Significant contribution, significantly advancing problem-solving and providing crucial intermediate results, while not the only path, it greatly improves the success rate or accuracy.
- **3**: Moderate contribution, providing auxiliary support to problem-solving, but not critical, the information can be omitted or replaced.
- **2**: Minor contribution, having almost no impact on problem-solving, it is redundant, repetitive, or of low value, and can be safely removed.
- **1**: Ineffective contribution or negative contribution, completely unrelated to the user's problem, or leading to errors and hindering problem-solving.

### Relevance Score (relevance_score)
Evaluate how relevant this step is to the user query:
- **5**: Highly relevant, directly and necessarily serves the core intent of the user's query, an indispensable key step.
- **4**: Fairly relevant, serves a reasonable sub-goal of the user's query; although not the most direct path, it is an effective step.
- **3**: Partially relevant, has an indirect or marginal connection to the user's query, but is not necessary.
- **2**: Largely irrelevant, has no substantial connection to the user's query, representing a clear deviation or misinterpretation of intent.
- **1**: Completely irrelevant, has no logical connection to the user's query.

### Accuracy Score (accuracy_score)
Evaluate the accuracy of information obtained or processed in this step:
- **5**: Completely accurate, the key information needed for the final answer is acquired and preserved completely, accurately, and without distortion.
- **4**: Mostly accurate, core information is accurate, but there are non-critical omissions or format inconsistencies that do not affect the essence of the information.
- **3**: Partially accurate, there are critical omissions or distortions; the information contains correct parts, but omits key qualifying conditions, states, or fields, which may lead to misunderstanding.
- **2**: Mostly incorrect, the information is severely distorted, key information is incorrectly acquired or reversed, and is clearly contradictory to the final answer.
- **1**: Completely fabricated, the information is untrue or irrelevant to the final answer.

### Efficiency Score (efficiency_score)
Evaluate the efficiency and necessity of this step:
- **5**: Highly efficient and necessary, steps are direct, precise, and indispensable.
- **4**: Fairly efficient, steps are necessary but slightly inefficient (e.g., non-optimal tool usage, redundant parameters, etc.).
- **3**: Moderately efficient, steps are not required for the task, contain redundancy, but do not affect the result.
- **2**: Inefficient and unnecessary, steps contain redundancy and can be optimized or removed.
- **1**: Completely unnecessary, steps are entirely unnecessary or severely waste resources, potentially even leading to task failure.


# Important Notes

- Identify any redundant or unnecessary tool calls, while noting missing tool calls that could improve answer quality
- Ensure scoring reflects both actual contribution to solving the user's problem and role in supporting factual accuracy
- Use step_index to match your evaluations with the correct steps

# Output Requirement

Please output your evaluation in JSON format:
{{
    "step_evaluations": [
        {{
            "step_index": <int - step index starting from 0>,
            "step_reason": "<detailed evaluation reasoning for this step>",
            "contribution_score": <int (1-5)>,
            "relevance_score": <int (1-5)>,
            "accuracy_score": <int (1-5)>,
            "efficiency_score": <int (1-5)>
        }}
    ]
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_TRAJECTORY_COMPREHENSIVE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TRAJECTORY_COMPREHENSIVE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TRAJECTORY_COMPREHENSIVE_PROMPT_ZH,
            ),
        ],
    },
)


def _normalize_score(score: Union[int, float]) -> float:
    """
    Normalize a 1-5 integer score to 0-1 continuous scale.

    Mapping:
    - 1 -> 0.0
    - 2 -> 0.25
    - 3 -> 0.5
    - 4 -> 0.75
    - 5 -> 1.0

    Args:
        score: Integer score from 1 to 5

    Returns:
        float: Normalized score from 0.0 to 1.0
    """
    # Clamp score to valid range [1, 5]
    score = max(1, min(5, float(score)))
    # Normalize: (score - 1) / 4
    return (score - 1) / 4.0


# Pydantic models for structured LLM output
class StepEvaluation(BaseModel):
    """Single step evaluation from LLM."""

    step_index: int = Field(description="Step index starting from 0")
    step_reason: str = Field(default="", description="Detailed evaluation reasoning for this step")
    contribution_score: int = Field(ge=1, le=5, description="Contribution score (1-5)")
    relevance_score: int = Field(ge=1, le=5, description="Relevance score (1-5)")
    accuracy_score: int = Field(ge=1, le=5, description="Accuracy score (1-5)")
    efficiency_score: int = Field(ge=1, le=5, description="Efficiency score (1-5)")


class TrajectoryEvaluationOutput(BaseModel):
    """Structured output model for trajectory evaluation LLM response."""

    step_evaluations: List[StepEvaluation] = Field(
        default_factory=list,
        description="List of step-level evaluations",
    )


class TrajectoryComprehensiveGrader(LLMGrader):
    """
    Comprehensive evaluation grader for agent trajectories.

    This grader evaluates agent trajectories by assessing each step independently:
    - Step-level evaluation: contribution, relevance, accuracy, efficiency (per step)
    - Overall score is computed by averaging all step scores (not from LLM output)

    The grader uses a 1-5 integer scoring system in prompts to avoid ambiguous boundary
    definitions, then normalizes scores to 0-1 range (1->0.0, 2->0.25, 3->0.5, 4->0.75, 5->1.0).

    The overall score is computed as:
    1. For each step: average of (contribution, relevance, accuracy, efficiency)
    2. Overall score: average of all step scores

    The grader accepts standard messages format and automatically extracts
    the trajectory after removing system prompts.

    Attributes:
        name: Grader name
        model: ChatModelBase instance for evaluation
        language: Language for evaluation prompts
        resolution_threshold: Threshold for determining if the trajectory is resolved (default: 0.8, on normalized 0-1 scale)

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> api = OpenAIChatModel(api_key="...", model="qwen3-32b")
        >>> grader = TrajectoryComprehensiveGrader(model=api, resolution_threshold=0.75)
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=[
        ...         {"role": "system", "content": "..."},
        ...         {"role": "user", "content": "帮我找投资建议"},
        ...         {"role": "assistant", "content": "...", "tool_calls": [...]},
        ...         ...
        ...     ]
        ... ))
        >>> print(f"Score: {result.score}")  # computed from step averages
    """

    @staticmethod
    def _create_trajectory_callback(
        language: LanguageEnum = LanguageEnum.ZH,
    ) -> Callable[[ChatResponse], Dict[str, Any]]:
        """
        Create a callback function to process step-level evaluations into final score and reason.

        This callback:
        1. Extracts step_evaluations from ChatResponse.metadata (which contains the model_dump of TrajectoryEvaluationOutput)
        2. Calculates average raw scores (1-5) across all steps for each dimension
        3. Normalizes the final average to 0-1 scale (more efficient than normalizing each step)
        4. Generates aggregated reason from step evaluations

        Args:
            language: Language for generating the aggregated reason

        Returns:
            Callable that processes ChatResponse into metadata dict with score and reason
        """

        def callback(response: ChatResponse) -> Dict[str, Any]:
            # Extract step_evaluations from ChatResponse.parsed
            # parsed contains the model_dump() of TrajectoryEvaluationOutput
            parsed = response.parsed or {}
            step_evaluations_raw = parsed.get("step_evaluations", [])

            # Convert dict representations to StepEvaluation objects
            # Note: structured_model ensures all items are dicts from model_dump()
            try:
                step_evaluations: List[StepEvaluation] = [
                    StepEvaluation(**s) if isinstance(s, dict) else s for s in step_evaluations_raw
                ]
            except Exception as e:
                logger.warning(f"Failed to parse step evaluations: {e}")
                step_evaluations = []

            if not step_evaluations:
                return {
                    "score": 0.0,
                    "reason": "No steps to evaluate." if language == LanguageEnum.EN else "没有可评估的步骤。",
                    "step_evaluations": [],
                    "avg_contribution": 0.0,
                    "avg_relevance": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_efficiency": 0.0,
                }

            num_steps = len(step_evaluations)

            # Calculate average raw scores (1-5) first - more efficient than normalizing each step
            total_contribution = sum(s.contribution_score for s in step_evaluations)
            total_relevance = sum(s.relevance_score for s in step_evaluations)
            total_accuracy = sum(s.accuracy_score for s in step_evaluations)
            total_efficiency = sum(s.efficiency_score for s in step_evaluations)

            avg_contribution_raw = total_contribution / num_steps
            avg_relevance_raw = total_relevance / num_steps
            avg_accuracy_raw = total_accuracy / num_steps
            avg_efficiency_raw = total_efficiency / num_steps

            # Normalize dimension averages for metadata
            avg_contribution = _normalize_score(avg_contribution_raw)
            avg_relevance = _normalize_score(avg_relevance_raw)
            avg_accuracy = _normalize_score(avg_accuracy_raw)
            avg_efficiency = _normalize_score(avg_efficiency_raw)

            # Calculate overall average in raw scale, then normalize once
            overall_raw = (avg_contribution_raw + avg_relevance_raw + avg_accuracy_raw + avg_efficiency_raw) / 4.0
            score = _normalize_score(overall_raw)
            reason = "\n".join([f"Step {s.step_index}: {s.step_reason}" for s in step_evaluations])

            # Convert step_evaluations to dicts for JSON serialization
            step_evaluations_dicts = [s.model_dump() for s in step_evaluations]

            return {
                "score": score,
                "reason": reason,
                "avg_contribution": avg_contribution,
                "avg_relevance": avg_relevance,
                "avg_accuracy": avg_accuracy,
                "avg_efficiency": avg_efficiency,
                "step_evaluations": step_evaluations_dicts,
            }

        return callback

    def __init__(
        self,
        model: Union[BaseChatModel, dict],
        template: Optional[PromptTemplate] = DEFAULT_TRAJECTORY_COMPREHENSIVE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        resolution_threshold: float = 0.8,
    ):
        """
        Initialize the TrajectoryComprehensiveGrader.

        Args:
            model (Union[BaseChatModel, dict]): The chat model to use for evaluation.
                Can be either a BaseChatModel instance or a dictionary configuration.
            template (Optional[PromptTemplate]): The prompt template for trajectory evaluation.
                Defaults to DEFAULT_TRAJECTORY_COMPREHENSIVE_TEMPLATE.
            language (LanguageEnum): Language for the evaluation prompt.
                Defaults to LanguageEnum.ZH (Chinese).
            resolution_threshold (float): Threshold for determining if the trajectory is resolved.
                Scores greater than or equal to this value are considered resolved.
                Defaults to 0.8 (80%).

        Example:
            >>> from openjudge.models.openai_chat_model import OpenAIChatModel
            >>> model = OpenAIChatModel(api_key="...", model="qwen3-32b")
            >>> grader = TrajectoryComprehensiveGrader(model=model, resolution_threshold=0.75)
        """
        super().__init__(
            name="trajectory_comprehensive",
            mode=GraderMode.POINTWISE,
            description="Comprehensive evaluation for agent trajectories including step-level and overall problem-solving assessment",
            model=model,
            template=template,
            language=language,
            structured_model=TrajectoryEvaluationOutput,
            callback=self._create_trajectory_callback(language=language),
        )
        self.resolution_threshold = resolution_threshold

    def _extract_trajectory_from_messages(
        self,
        messages: List[Dict[str, Any]],
        language: LanguageEnum = LanguageEnum.ZH,
    ) -> tuple[str, str, str]:
        """
        Extract user query, trajectory, and final answer from messages.

        Args:
            messages: List of message dicts (standard format).
            language: Language for formatting trajectory messages (ZH or EN)

        Returns:
            Tuple of (user_query, trajectory_messages, final_answer)
        """
        # Filter out system messages
        messages = [msg.get("message", msg) for msg in messages]
        non_system_messages = [msg for msg in messages if msg.get("role", "") != "system"]

        if not non_system_messages:
            return "", "", ""

        # Extract user query (first non-system user message)
        user_query = ""
        if non_system_messages[0].get("role", "") == "user":
            user_query = non_system_messages[0].get("content", "")

        # Extract final answer (last assistant message content)
        final_answer = ""
        for msg in reversed(non_system_messages):
            if msg.get("role", "") == "assistant" and msg.get("content", ""):
                final_answer = msg.get("content", "")
                break

        # Language-specific labels
        if language == LanguageEnum.ZH:
            step_label = "步骤"
            assistant_label = "助手"
            tool_calls_label = "工具调用"
            tool_response_label = "工具响应"
            user_label = "用户"
        else:
            step_label = "Step"
            assistant_label = "Assistant"
            tool_calls_label = "Tool Calls"
            tool_response_label = "Tool Response"
            user_label = "User"

        # Format trajectory: exclude first user query and last assistant response
        trajectory_parts = []
        step_index = 0

        for msg in non_system_messages[1:]:  # Skip first user query
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # Skip the final answer in trajectory
            if role == "assistant" and content == final_answer and not tool_calls:
                continue

            if role == "assistant" and tool_calls:
                # Format tool calls with step index
                tool_calls_formatted = []
                for tc in tool_calls:
                    tc = tc.get("tool_call", tc)
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    tool_args = func.get("arguments", "{}")

                    try:
                        args = json.dumps(json.loads(tool_args), ensure_ascii=False)
                        tool_calls_formatted.append(f"  - {tool_name}({args})")
                    except json.JSONDecodeError:
                        tool_calls_formatted.append(f"  - {tool_name}({tool_args})")

                step = (
                    f"**{step_label} {step_index} - {assistant_label} {tool_calls_label}**:\n\n{tool_calls_formatted}"
                )
                trajectory_parts.append(step)
                step_index += 1

            elif role == "tool":
                tool_name = msg.get("name", "unknown_tool")
                trajectory_parts.append(f"**{tool_response_label} ({tool_name})**: {content}")

            elif role == "assistant" and content:
                # Intermediate assistant responses (not final answer)
                trajectory_parts.append(f"**{assistant_label}**: {content}")

            elif role == "user":
                # Additional user messages in multi-turn conversations
                trajectory_parts.append(f"**{user_label}**: {content}")

        trajectory_messages = "\n\n".join(trajectory_parts)

        return user_query, trajectory_messages, final_answer

    async def aevaluate(
        self,
        messages: List[Dict[str, Any]],
        query: Optional[str] = None,
        response: Optional[str | Dict[str, Any]] = None,
    ) -> GraderScore | GraderError:
        """
        Evaluate complete agent trajectory comprehensively.

        The evaluation uses 1-5 integer scores in LLM prompts for each step, then normalizes to 0-1 scale:
        - 1 -> 0.0, 2 -> 0.25, 3 -> 0.5, 4 -> 0.75, 5 -> 1.0

        The overall score is computed as the average of all step scores (each step's score is the average
        of its four dimensions: contribution, relevance, accuracy, efficiency).

        The callback function handles step-level to final score/reason conversion efficiently:
        - Calculates average raw scores (1-5) first
        - Then normalizes the final result (avoiding redundant per-step normalization for aggregation)

        Args:
            messages: List of messages (standard format, including system, user, assistant, tool)
                "message" key for message, and "tool_call" key for tool call can be optional.
                example without "message" and "tool_call"
                ```
                [
                  {"role": "system", "content": "..."},
                  {"role": "user", "content": "Plan travel from Shanghai to Hangzhou."},
                  {"role": "assistant", "tool_calls": [{"function": {"arguments": "{\"city\": \"Hangzhou\"}","name": "weather"}}]}
                ]
                ```
                or with "message" and "tool_call"
                ```
                [
                  {"message":{"role": "system", "content": "..."}},
                  {"message":{"role": "user", "content": "Plan travel from Shanghai to Hangzhou."}},
                  {"role": "assistant", "tool_calls": [{"tool_call":{"function": {"arguments": "{\"city\": \"Hangzhou\"}","name": "weather"}}}]}
                ]
                ```
            query:    optional, user query, will use the first message with role=user as query if not provided.
            response: optional, final response, will use the last non-emptry message 'content' with role=assistant if not provided.

        Returns:
            GraderScore: Comprehensive evaluation score for the trajectory (normalized 0.0-1.0)
                - score: Overall score computed from step averages (normalized 0.0-1.0)
                - reason: Aggregated evaluation summary generated from step evaluations
                - metadata: Contains step_evaluations list with normalized (0-1) scores

        Example:
            >>> result = await grader.aevaluate(
            ...     messages=[
            ...         {"role": "user", "content": "帮我找投资建议"},
            ...         {"role": "assistant", "content": "...", "tool_calls": [...]},
            ...         ...
            ...     ]
            ... )
            >>> print(f"Overall Score: {result.score}")  # normalized 0-1, computed from step averages
            >>> for step in result.metadata["step_evaluations"]:
            ...     print(f"Step {step['step_index']}: contribution={step['contribution_score']}")
        """
        # Extract trajectory from messages
        user_query, trajectory_messages, final_answer = self._extract_trajectory_from_messages(
            messages,
            language=self.language,
        )
        user_query = query or user_query
        final_answer = response or final_answer

        if not user_query or not trajectory_messages:
            logger.warning("Empty user query or trajectory, returning zero score")
            return GraderError(name=self.name, error="Empty user query or trajectory")

        try:
            # Call parent evaluation with formatted parameters
            # The callback handles step-level to final score/reason conversion
            result = await super().aevaluate(
                query=user_query,
                messages=trajectory_messages,
                response=final_answer,
            )

            # Determine resolution status using the specified threshold
            is_resolved = result.score >= self.resolution_threshold

            # Add additional metadata
            metadata = result.metadata or {}
            metadata["is_resolved"] = is_resolved
            metadata["resolution_threshold"] = self.resolution_threshold
            metadata["evaluation_type"] = "trajectory_comprehensive"

            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {e}")
            return GraderError(name=self.name, error=f"Evaluation error: {str(e)}")
