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

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate
from rm_gallery.core.models.schema.response import ChatResponse

# pylint: disable=line-too-long,too-many-statements

# Chinese Prompt
TRAJECTORY_COMPREHENSIVE_PROMPT_ZH = """# 任务描述

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

{user_query}

# 完整轨迹

{trajectory_messages}

# 最终答案

{final_answer}

# 评估指南

## 各维度评分框架

### 贡献度评分 (contribution_score)
评估此步骤对解决用户问题的整体贡献程度：
- **5分**: 关键贡献，解决问题必不可少，直接推动问题解决
- **4分**: 重要贡献，提供关键信息或执行重要操作
- **3分**: 中等贡献，有一定帮助但非核心步骤
- **2分**: 轻微贡献，提供一些辅助信息但价值有限
- **1分**: 无效贡献，不相关、冗余或对问题解决无帮助

### 相关性评分 (relevance_score)
评估此步骤与用户查询的相关程度：
- **5分**: 高度相关，直接针对用户需求
- **4分**: 较为相关，与用户需求密切关联
- **3分**: 部分相关，与用户需求有一定关联
- **2分**: 略微相关，与用户需求关联较弱
- **1分**: 不相关，偏离用户需求或完全无关

### 准确性评分 (accuracy_score)
评估此步骤获取或处理信息的准确程度：
- **5分**: 完全准确，信息可验证且无误
- **4分**: 基本准确，信息可靠，可能有极小偏差
- **3分**: 部分准确，信息大体正确但有一些不确定性
- **2分**: 准确性存疑，包含可疑或未验证的信息
- **1分**: 不准确，包含明显错误或误导性信息

### 效率评分 (efficiency_score)
评估此步骤的执行效率和必要性：
- **5分**: 高效必要，步骤精准且不可省略
- **4分**: 较为高效，步骤合理且有明确目的
- **3分**: 效率一般，步骤可接受但可能有更优方案
- **2分**: 效率较低，步骤冗余或可以合并优化
- **1分**: 低效或多余，步骤完全不必要或严重浪费资源

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

# English Prompt
TRAJECTORY_COMPREHENSIVE_PROMPT_EN = """# Task Description

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

{user_query}

# Complete Trajectory

{trajectory_messages}

# Final Answer

{final_answer}

# Evaluation Guidelines

## Scoring Framework for Each Dimension

### Contribution Score (contribution_score)
Evaluate how much this step contributes to solving the user's problem:
- **5**: Critical contribution, essential for solving the problem, directly drives solution
- **4**: Significant contribution, provides key information or performs important operations
- **3**: Moderate contribution, somewhat helpful but not a core step
- **2**: Minor contribution, provides auxiliary information with limited value
- **1**: Ineffective contribution, irrelevant, redundant, or unhelpful for problem solving

### Relevance Score (relevance_score)
Evaluate how relevant this step is to the user query:
- **5**: Highly relevant, directly addresses user needs
- **4**: Fairly relevant, closely related to user needs
- **3**: Partially relevant, has some connection to user needs
- **2**: Slightly relevant, weakly connected to user needs
- **1**: Not relevant, deviates from or completely unrelated to user needs

### Accuracy Score (accuracy_score)
Evaluate the accuracy of information obtained or processed in this step:
- **5**: Completely accurate, information is verifiable and error-free
- **4**: Mostly accurate, information is reliable with minimal deviation
- **3**: Partially accurate, information is generally correct with some uncertainty
- **2**: Questionable accuracy, contains suspicious or unverified information
- **1**: Inaccurate, contains obvious errors or misleading information

### Efficiency Score (efficiency_score)
Evaluate the efficiency and necessity of this step:
- **5**: Highly efficient and necessary, precise and indispensable step
- **4**: Fairly efficient, reasonable step with clear purpose
- **3**: Moderately efficient, acceptable but could potentially be optimized
- **2**: Low efficiency, redundant or could be merged/optimized
- **1**: Inefficient or unnecessary, completely redundant or wastes resources

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

# Build default template from prompts
DEFAULT_TRAJECTORY_COMPREHENSIVE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TRAJECTORY_COMPREHENSIVE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(TRAJECTORY_COMPREHENSIVE_PROMPT_ZH),
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
        >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
        >>> api = OpenAIChatModel(api_key="...", model="gpt-4o")
        >>> grader = TrajectoryComprehensiveGrader(model=api, resolution_threshold=0.75)
        >>> result = await grader.aevaluate(
        ...     messages=[
        ...         {"role": "system", "content": "..."},
        ...         {"role": "user", "content": "帮我找投资建议"},
        ...         {"role": "assistant", "content": "...", "tool_calls": [...]},
        ...         ...
        ...     ]
        ... )
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
            # Extract step_evaluations from ChatResponse.metadata
            # metadata contains the model_dump() of TrajectoryEvaluationOutput
            metadata = response.metadata or {}
            step_evaluations_raw = metadata.get("step_evaluations", [])

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
        language: LanguageEnum = LanguageEnum.ZH,
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
            >>> from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
            >>> model = OpenAIChatModel(api_key="...", model="gpt-4o")
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
    ) -> GraderScore:
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

        if not user_query or not trajectory_messages:
            logger.warning("Empty user query or trajectory, returning zero score")
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="Empty user query or trajectory",
                metadata={
                    "evaluation_type": "trajectory_comprehensive",
                    "error": "Empty input",
                    "step_evaluations": [],
                    "is_resolved": False,
                    "resolution_threshold": self.resolution_threshold,
                },
            )

        try:
            # Call parent evaluation with formatted parameters
            # The callback handles step-level to final score/reason conversion
            result = await super().aevaluate(
                user_query=user_query,
                trajectory_messages=trajectory_messages,
                final_answer=final_answer,
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
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"Evaluation error: {str(e)}",
                metadata={
                    "evaluation_type": "trajectory_comprehensive",
                    "error": str(e),
                    "step_evaluations": [],
                    "is_resolved": False,
                    "resolution_threshold": self.resolution_threshold,
                    "avg_contribution": 0.0,
                    "avg_relevance": 0.0,
                    "avg_accuracy": 0.0,
                    "avg_efficiency": 0.0,
                },
            )
