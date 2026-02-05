# -*- coding: utf-8 -*-
"""
Trajectory Accuracy Grader

Evaluates the accuracy of agent trajectories in solving user queries.
"""

import json
import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy.base_evaluation_strategy import (
    BaseEvaluationStrategy,
)
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
TRAJECTORY_ACCURACY_PROMPT_EN = textwrap.dedent(
    """
You are a professional data labeling expert specializing in trajectory evaluation. \
Your goal is to assess the accuracy of an AI agent's internal trajectory based on the \
provided conversation data.
**Trajectory Accuracy** refers to the overall correctness and effectiveness of an agent's \
trajectory (sequence of actions, tool calls, and responses) in achieving the user's goal.

<Rubrics>
Evaluate based on these factors:
1. **Logical Coherence**: Does the trajectory maintain logical coherence between consecutive steps?
2. **Goal Progression**: Does the trajectory show clear progression toward the objective?
3. **Efficiency**: Is the trajectory reasonably efficient (perfection is not required, but it \
    should not be unnecessarily inefficient)?
</Rubrics>

<Steps>
First, identify the goal of the trajectory by examining the input (if the input is not explicitly \
provided, infer it from the content of the first message), as well as the output of the final message. \
Once you understand the goal, evaluate the trajectory based on how effectively it achieves that goal.
</Steps>


<Scale>
- Score 3: Successfully achieves the task goal without any steps totally unrelated to the task \
    (reasonable extensions to improve task quality are acceptable).
- Score 2: Successfully achieves the task goal, but contains obvious unnecessary steps \
    unrelated to the task.
- Score 1: Fails to achieve the task goal.
</Scale>


<trajectory>
TRAJECTORY: {messages}
</trajectory>

<Output Schema>
Your output should be a JSON object with the following format:
```json
{{
    "score": [Trajectory Accuracy Score],
    "reason": [Reason for the score],
}}
```
</Output Schema>
JSON:
"""
).strip()

# Chinese Prompt
TRAJECTORY_ACCURACY_PROMPT_ZH = textwrap.dedent(
    """
你是一位专业的数据标注专家，专注于轨迹评估。\
你的目标是根据提供的对话数据，评估AI智能体内部轨迹的准确性。
**轨迹准确性**是指智能体的轨迹（动作序列、工具调用和响应）在实现用户目标方面的整体正确性和有效性。

<评分标准>
根据以下因素进行评估：
1. **逻辑连贯性**：轨迹在连续步骤之间是否保持逻辑连贯？
2. **目标推进**：轨迹是否显示出朝向目标的清晰进展？
3. **效率**：轨迹是否合理高效（不要求完美，但不应存在不必要的低效）？
</评分标准>

<评估步骤>
首先，通过检查输入（如果输入未明确提供，则从第一条消息的内容中推断）以及最终消息的输出来\
确定轨迹的目标。一旦理解了目标，就根据轨迹实现该目标的有效程度进行评估。
</评估步骤>


<评分量表>
- 3分：成功实现任务目标，且没有任何与任务完全无关的步骤（为提高任务质量而进行的合理扩展是可以接受的）。
- 2分：成功实现任务目标，但包含与任务明显无关的不必要步骤。
- 1分：未能实现任务目标。
</评分量表>


<轨迹>
轨迹内容：{messages}
</轨迹>

<输出格式>
你的输出应该是一个JSON对象，格式如下：
```json
{{
    "score": [轨迹准确性评分],
    "reason": [评分原因],
}}
```
</输出格式>
JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_TRAJECTORY_ACCURACY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=TRAJECTORY_ACCURACY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=TRAJECTORY_ACCURACY_PROMPT_ZH,
            ),
        ],
    },
)


class TrajectoryAccuracyGrader(LLMGrader):
    """
    Trajectory Accuracy Grader

    Evaluates the accuracy of agent trajectories in solving user queries.

    The TrajectoryAccuracyGrader assesses how accurately an agent's trajectory addresses
    and solves a user's query by examining:
    - Relevance to the user query
    - Logical flow of actions
    - Correctness of tool calls
    - Accuracy of the final solution

    The evaluator uses a scoring rubric of 1, 2, and 3:
    - Score 3: Successfully achieves the task goal without unnecessary steps
    - Score 2: Successfully achieves the task goal but contains unnecessary steps
    - Score 1: Fails to achieve the task goal

    This evaluation focuses on measuring whether the trajectory meaningfully contributes
    to solving the user's query with correct logic and accurate results.

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
        ...     api_key="your-key",  # pragma: allowlist secret
        ...     model="qwen3-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = TrajectoryAccuracyGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>>
        >>> messages = [
        ...     {"role": "user", "content": "What's the weather like in New York?"},
        ...     {"role": "assistant", "content": "...", "tool_calls": [...]},
        ...     {"role": "tool", "content": "..."},
        ...     {"role": "assistant", "content": "The weather in New York is sunny, 72°F."}
        ... ]
        >>> result = asyncio.run(grader.aevaluate(
        ...     messages=messages
        ... ))
        >>> print(result.score)
        3.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_TRAJECTORY_ACCURACY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """Initialize the TrajectoryAccuracyGrader.

        Args:
            model: The language model used for evaluation. Can be either a BaseChatModel
                   instance or a dictionary configuration. If a dict is provided, it will
                   be used to initialize an OpenAIChatModel.
            template: Evaluation template. Defaults to DEFAULT_TRAJECTORY_ACCURACY_TEMPLATE.
            language: Language for evaluation prompts (default: LanguageEnum.EN).
            strategy: Evaluation strategy. Defaults to None.
        """
        super().__init__(
            name="trajectory_accuracy",
            mode=GraderMode.POINTWISE,
            description="Evaluates the accuracy of agent trajectories in solving user queries",
            model=model,
            template=template or DEFAULT_TRAJECTORY_ACCURACY_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Format messages into a readable string representation.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            Formatted string representation of messages.
        """
        # Handle messages that might be wrapped in "message" key
        messages = [msg.get("message", msg) for msg in messages]

        formatted_parts = []
        for _, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            # Format content
            if isinstance(content, str):
                content_str = content
            elif isinstance(content, list):
                # Handle multimodal content
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                content_str = " ".join(text_parts) if text_parts else ""
            else:
                content_str = str(content)

            # Format message
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
    ) -> GraderScore | GraderError:
        """
        Evaluate trajectory accuracy

        This method evaluates the accuracy of agent trajectories based on multiple criteria including
        relevance to user query, logical flow, tool call correctness, and solution accuracy.
        It assigns a score of 1, 2, or 3 based on how well the trajectory achieves the task goal.

        Args:
            messages: List of messages in OpenAI format representing the conversation trajectory.
                     Each message should have 'role' and 'content' fields.
                     Example:
                     [
                         {"role": "user", "content": "What's the weather?"},
                         {"role": "assistant", "content": "...", "tool_calls": [...]},
                         {"role": "tool", "content": "..."},
                         {"role": "assistant", "content": "The weather is sunny."}
                     ]

        Returns:
            GraderScore: Score of 1, 2, or 3 indicating trajectory accuracy

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What's the weather like in New York?"},
            ...     {"role": "assistant", "content": "...", "tool_calls": [...]},
            ...     {"role": "tool", "content": "..."},
            ...     {"role": "assistant", "content": "The weather in New York is sunny, 72°F."}
            ... ]
            >>> result = await grader.aevaluate(messages=messages)
        """
        # Check if we have messages to evaluate
        if not messages:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No messages provided to evaluate",
                metadata={
                    "error": "No messages provided",
                },
            )

        # Format messages for evaluation
        formatted_messages = self._format_messages(messages)

        try:
            # Call parent evaluate method with the structured data
            # The prompt will extract query and response from messages automatically
            result = await super()._aevaluate(
                messages=formatted_messages,
            )
            score = max(1.0, min(3.0, result.score))
            reason = result.reason
        except Exception as e:
            logger.error(f"Error evaluating trajectory accuracy: {e}")
            reason = f"Evaluation error: {str(e)}"
            return GraderError(
                name=self.name,
                error=reason,
            )

        # Prepare metadata
        metadata = {
            "raw_score": score,
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "TrajectoryAccuracyGrader",
    "DEFAULT_TRAJECTORY_ACCURACY_TEMPLATE",
]
