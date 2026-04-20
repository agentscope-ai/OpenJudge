# -*- coding: utf-8 -*-
"""
Plan Decomposition Grader

Evaluates whether the agent correctly decomposes a complex task into appropriate
sub-tasks in its plan — identifying all necessary sub-goals, ordering them correctly,
and recognizing dependencies.
"""

import textwrap
from typing import Any, Dict, List, Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.agent.utils import format_history
from openjudge.graders.base_grader import GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
PLAN_DECOMPOSITION_PROMPT_EN = textwrap.dedent(
    """You are an expert in analyzing agent plan decomposition. Your task is to evaluate whether the agent correctly decomposes a complex task into appropriate sub-tasks in its plan. Good plan decomposition identifies all necessary sub-goals, orders them correctly based on dependencies, and creates a plan that is both complete and efficient.

<Rubrics>
1. The agent identifies all necessary sub-goals required to complete the task (no critical sub-goals are missed)
2. The sub-goals are ordered correctly respecting dependencies (prerequisites are completed before dependent steps)
3. The decomposition is at an appropriate granularity (not too coarse, not overly detailed)
4. The agent recognizes and handles parallel vs. sequential sub-tasks appropriately
5. The decomposition does not include unnecessary or irrelevant sub-tasks
6. The agent correctly identifies constraints and requirements from the task description that affect decomposition
</Rubrics>

<Steps>
1. Parse the task: Identify the overall goal, constraints, and implicit requirements
2. Evaluate sub-goal completeness: Are all necessary steps identified?
3. Check ordering: Are dependencies respected in the sub-goal ordering?
4. Assess granularity: Is the decomposition at an appropriate level of detail?
5. Check for extraneous sub-tasks: Are there unnecessary steps?
6. Evaluate constraint awareness: Does the decomposition account for task constraints?
</Steps>

<Scale>
- **Score 5**: Excellent decomposition — All sub-goals identified, correctly ordered, appropriate granularity, no unnecessary steps, constraints fully accounted for
- **Score 4**: Good decomposition — All major sub-goals identified with minor ordering or granularity issues
- **Score 3**: Adequate decomposition — Most sub-goals identified, but some missing or ordering issues that could affect task completion
- **Score 2**: Poor decomposition — Significant sub-goals missing or incorrect ordering that would lead to task failure
- **Score 1**: Failed decomposition — The agent fails to decompose the task meaningfully or misses most critical sub-goals
</Scale>

<Context (Optional)>
{context}
</Context>

<History (Optional)>
{history}
</History>

<Task>
{query}
</Task>

<Agent's Decomposition/Plan>
{plan}
</Agent's Decomposition/Plan>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<detailed explanation of plan decomposition quality, including identified strengths and weaknesses>",
    "score": <integer between 1 and 5>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
PLAN_DECOMPOSITION_PROMPT_ZH = textwrap.dedent(
    """你是一名分析智能体计划分解的专家。你的任务是评估智能体是否在计划中正确地将复杂任务分解为适当的子任务。良好的计划分解能够识别所有必要的子目标、根据依赖关系正确排序，并创建完整且高效的计划。

<评分标准>
1. 智能体识别了完成任务所需的所有必要子目标（没有遗漏关键子目标）
2. 子目标按照依赖关系正确排序（先决条件在依赖步骤之前完成）
3. 分解处于适当的粒度（不太粗略，也不太过于详细）
4. 智能体适当识别和处理并行与顺序子任务
5. 分解不包括不必要或不相关的子任务
6. 智能体正确识别任务描述中影响分解的约束和需求
</评分标准>

<评估步骤>
1. 解析任务：识别总体目标、约束和隐含需求
2. 评估子目标完整性：是否识别了所有必要步骤？
3. 检查排序：子目标排序是否尊重了依赖关系？
4. 评估粒度：分解是否处于适当的详细程度？
5. 检查多余的子任务：是否存在不必要的步骤？
6. 评估约束意识：分解是否考虑了任务约束？
</评估步骤>

<评分量表>
- **分数 5**：优秀的分解 — 所有子目标都已识别、正确排序、粒度适当、没有不必要的步骤、完全考虑了约束
- **分数 4**：良好的分解 — 所有主要子目标已识别，有轻微的排序或粒度问题
- **分数 3**：足够的分解 — 大多数子目标已识别，但有些缺失或排序问题可能影响任务完成
- **分数 2**：较差的分解 — 缺少重要子目标或排序不正确，可能导致任务失败
- **分数 1**：失败的分解 — 智能体未能有意义地分解任务或遗漏了大多数关键子目标
</评分量表>

<上下文（可选）>
{context}
</上下文>

<历史记录（可选）>
{history}
</历史记录>

<任务>
{query}
</任务>

<智能体的分解/计划>
{plan}
</智能体的分解/计划>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<关于计划分解质量的详细解释，包括识别的优点和缺点>",
    "score": <1 到 5 之间的整数>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_PLAN_DECOMPOSITION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=PLAN_DECOMPOSITION_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=PLAN_DECOMPOSITION_PROMPT_ZH,
            ),
        ],
    },
)


class PlanDecompositionGrader(LLMGrader):
    """
    Plan Decomposition Grader

    Evaluates whether the agent correctly decomposes a complex task into appropriate
    sub-tasks in its plan.

    Required modules: query, plan

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
        >>> grader = PlanDecompositionGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Book a flight from NYC to London, find a hotel near the airport, and arrange airport transfer",
        ...     plan="1. Search flights NYC→London 2. Book flight 3. Search hotels near London airport 4. Book hotel 5. Search airport transfer options 6. Book transfer"
        ... ))
        >>> print(f"Score: {result.score}")
    """

    DEFAULT_TEMPLATE = DEFAULT_PLAN_DECOMPOSITION_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize PlanDecompositionGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            template: PromptTemplate for evaluation prompts
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectStrategy.
        """
        super().__init__(
            name="plan_decomposition",
            mode=GraderMode.POINTWISE,
            description="Evaluate plan decomposition quality",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )

    async def _aevaluate(
        self,
        query: str,
        plan: str,
        history: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        **kwargs: Any,
    ) -> GraderScore:
        """
        Evaluate plan decomposition quality.

        Args:
            query: The user's task or query
            plan: The agent's plan/decomposition of the task
            history: Optional list of previous step dictionaries for context
            context: Optional task context

        Returns:
            GraderScore: Score between 1 and 5
        """
        context_str = context if context else ""
        history_str = format_history(history, include_tags=False)

        try:
            result = await super()._aevaluate(
                query=query,
                plan=plan,
                history=history_str,
                context=context_str,
            )
            score = result.score
            reason = result.reason

        except Exception as e:
            logger.error(f"Error evaluating plan decomposition: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        metadata = {
            "raw_score": score,
            "evaluation_type": "plan_decomposition",
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "PlanDecompositionGrader",
    "DEFAULT_PLAN_DECOMPOSITION_TEMPLATE",
]
