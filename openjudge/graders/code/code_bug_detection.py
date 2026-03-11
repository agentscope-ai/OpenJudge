# -*- coding: utf-8 -*-
"""
Code Bug Detection Grader

Evaluates whether AI-generated code contains potential bugs — including logic errors,
boundary condition failures, resource leaks, race conditions, and incorrect assumptions —
without requiring pre-written test cases.

Inspired by pr-agent's `key_issues_to_review` dimension, which surfaces high-priority bugs
and correctness concerns that a human reviewer should focus on, covering issues that static
analysis and unit tests may miss.
"""

import textwrap
from typing import Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# English Prompt
CODE_BUG_DETECTION_PROMPT_EN = textwrap.dedent(
    """
You are an expert software engineer and code reviewer responsible for identifying potential
bugs in AI-generated code. Your task is to analyze the code for correctness issues and
assign a score based on the likelihood and severity of bugs found.

<Rubrics>
Bug-free code should:
- Handle all boundary and edge cases (empty inputs, zero, negative numbers, None/null values,
  empty collections, maximum values, off-by-one scenarios).
- Correctly implement the algorithm described in the task without logic errors.
- Properly manage resources (file handles, connections, locks) — open what you close,
  acquire what you release.
- Use correct data types and avoid unintended type coercions or precision loss.
- Avoid off-by-one errors in loop bounds, slice indices, and range calculations.
- Handle exceptions and error conditions without silently swallowing errors or crashing.
- Produce correct output for the base case, typical case, and extreme cases.
- Not rely on undefined behavior, uninitialized variables, or implicit assumptions about
  state that may not hold at runtime.
- Correctly handle concurrency concerns when applicable (race conditions, deadlocks, TOCTOU).
- Return or propagate results correctly through all code paths (no missing return statements).

Points should be deducted for:
- Logic errors that cause incorrect results on valid inputs.
- Missing or incorrect boundary/edge case handling.
- Off-by-one errors in loops, indices, or range computations.
- Unhandled exceptions or error paths that cause crashes.
- Resource leaks (unclosed files, connections, or unreleased locks).
- Incorrect assumptions about input types, nullability, or state.
- Infinite loops or unintended recursion without base case protection.
- Race conditions or shared-state mutation in concurrent code.
- Missing return values on some code paths.
- Incorrect use of mutable default arguments (Python-specific: `def f(x=[]):`).
</Rubrics>

<Steps>
- Carefully read the task description to understand the intended behavior and expected inputs/outputs.
- Trace through the code logic mentally for typical inputs, edge cases (empty, None, zero,
  negative, maximum), and error conditions.
- Check loop bounds, index access, and off-by-one patterns.
- Look for unhandled exception paths, missing error checks, and resource cleanup.
- Identify any assumptions the code makes that may not always hold at runtime.
- Assess the overall bug likelihood based on findings.
</Steps>

<Constraints>
Focus on correctness bugs only — not style, performance, or security (those are separate
concerns). A beautifully written but logically incorrect function should score low. Simple,
correct code should score high. Only penalize for bugs that are plausibly triggered by real
inputs, not purely hypothetical scenarios.
</Constraints>

<Scale>
- 5: No bugs detected. The code correctly handles all typical cases and visible edge cases.
- 4: Minor potential issues that are unlikely to manifest in practice (e.g., an edge case
  that almost never occurs in the expected usage context, or a very defensive missing check
  that is more style than substance).
- 3: Noticeable bugs present that would cause incorrect behavior for some valid inputs
  (e.g., an off-by-one error in a loop, missing null check for a nullable field).
- 2: Significant bugs that would cause failures or wrong results for common inputs
  (e.g., incorrect algorithm logic, unhandled exception on normal usage, resource leak
  in a frequently-called path).
- 1: Critical bugs rendering the code largely non-functional. The primary use case fails,
  or multiple severe issues exist that together make the code unreliable.
</Scale>

<Task Description>
{query}
</Task Description>

<Code>
{response}
</Code>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<concise explanation of findings. For each bug found, describe: what the bug is, which input or condition triggers it, and its likely impact. If no bugs are found, confirm correctness.>",
    "score": <integer between 1 and 5, where 5 means no bugs detected and 1 means critical bugs>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
CODE_BUG_DETECTION_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的软件工程师和代码审查员，负责识别AI生成代码中的潜在Bug。你的任务是分析代码的正确性问题，并根据发现的Bug的可能性和严重性进行评分。

<评分标准>
无Bug的代码应该：
- 处理所有边界和边缘情况（空输入、零值、负数、None/null值、空集合、最大值、差一错误场景）。
- 正确实现任务中描述的算法，不存在逻辑错误。
- 正确管理资源（文件句柄、连接、锁）——打开的要关闭，获取的要释放。
- 使用正确的数据类型，避免意外的类型强制转换或精度损失。
- 避免循环边界、切片索引和范围计算中的差一错误。
- 处理异常和错误条件，不静默吞噬错误或崩溃。
- 对基本情况、典型情况和极端情况产生正确的输出。
- 不依赖未定义行为、未初始化变量或在运行时可能不成立的状态隐式假设。
- 在适用时正确处理并发问题（竞态条件、死锁、TOCTOU）。
- 在所有代码路径上正确返回或传播结果（无缺失的返回语句）。

以下情况应扣分：
- 对有效输入产生错误结果的逻辑错误。
- 缺失或错误的边界/边缘情况处理。
- 循环、索引或范围计算中的差一错误。
- 未处理的异常或导致崩溃的错误路径。
- 资源泄漏（未关闭的文件、连接或未释放的锁）。
- 对输入类型、可空性或状态的错误假设。
- 无限循环或无基本情况保护的意外递归。
- 并发代码中的竞态条件或共享状态变更。
- 某些代码路径缺少返回值。
- 可变默认参数的不正确使用（Python特定：`def f(x=[]):`）。
</评分标准>

<评估步骤>
- 仔细阅读任务描述，了解预期行为和期望的输入/输出。
- 在脑中追踪典型输入、边缘情况（空、None、零、负值、最大值）和错误条件下的代码逻辑。
- 检查循环边界、索引访问和差一错误模式。
- 寻找未处理的异常路径、缺失的错误检查和资源清理。
- 识别代码在运行时可能不总是成立的假设。
- 根据发现结果评估整体Bug可能性。
</评估步骤>

<注意事项>
仅关注正确性Bug，不考虑风格、性能或安全性（这些是独立的关注点）。编写精美但逻辑错误的函数应获得低分。简单但正确的代码应获得高分。只针对真实输入可能触发的Bug扣分，不针对纯假设场景。
</注意事项>

<评分量表>
- 5: 未检测到Bug。代码正确处理所有典型情况和可见的边缘情况。
- 4: 存在轻微的潜在问题，在实践中不太可能出现（例如，在预期使用场景中几乎不会发生的边缘情况，或更多是风格而非实质的防御性缺失检查）。
- 3: 存在明显的Bug，会导致某些有效输入出现错误行为（例如，循环中的差一错误，可空字段缺少null检查）。
- 2: 存在重大Bug，会导致常见输入的失败或错误结果（例如，不正确的算法逻辑、正常使用时未处理的异常、频繁调用路径中的资源泄漏）。
- 1: 存在关键Bug，导致代码基本无法运行。主要用例失败，或存在多个严重问题，共同使代码不可靠。
</评分量表>

<任务描述>
{query}
</任务描述>

<代码>
{response}
</代码>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<发现结果的简要说明。对于发现的每个Bug，描述：Bug是什么，哪种输入或条件触发它，以及其可能的影响。如果没有发现Bug，确认代码的正确性。>",
    "score": <1到5之间的整数，其中5表示未检测到Bug，1表示存在关键Bug>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_CODE_BUG_DETECTION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=CODE_BUG_DETECTION_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=CODE_BUG_DETECTION_PROMPT_ZH,
            ),
        ],
    },
)


class CodeBugDetectionGrader(LLMGrader):
    """
    Code Bug Detection Grader

    Purpose:
        Detects potential bugs in AI-generated code through LLM-based reasoning, inspired by
        pr-agent's `key_issues_to_review` dimension. Unlike `CodeExecutionGrader`, this grader
        requires no pre-written test cases — it reasons about correctness from the code itself,
        covering bugs that unit tests often miss (race conditions, resource leaks, edge cases).

    What it evaluates:
        - Logic Errors: Incorrect algorithm implementation, wrong conditionals, bad state transitions
        - Boundary / Edge Cases: Empty inputs, null/None, zero, negative, max values, off-by-one
        - Resource Management: Unclosed files/connections, unreleased locks, memory leaks
        - Exception Handling: Swallowed errors, missing error propagation, crash-prone paths
        - Type Safety: Wrong type assumptions, implicit coercions, precision loss
        - Concurrency: Race conditions, deadlocks, shared mutable state issues
        - Return Value Correctness: Missing returns on some paths, incorrect propagation

    When to use:
        - Evaluating LLM code generation quality without a test suite
        - Benchmarking model bug-proneness across different tasks
        - Early-stage code review before execution testing
        - Complementing `CodeExecutionGrader` with reasoning-based bug detection
        - Identifying systematic failure patterns in a model's code output

    Scoring (higher = fewer bugs):
        - 5: No bugs detected; code handles typical and edge cases correctly
        - 4: Minor potential issues unlikely to manifest in normal usage
        - 3: Noticeable bugs for some valid inputs (off-by-one, missing null check)
        - 2: Significant bugs causing failures on common inputs
        - 1: Critical bugs; primary use case fails or multiple severe issues exist

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 5] to pass (default: 3)
        template: Custom evaluation template (default: DEFAULT_CODE_BUG_DETECTION_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: Evaluation strategy (default: DirectEvaluationStrategy)

    Returns:
        GraderScore with:
            - score: [1, 5] where 5 = no bugs, 1 = critical bugs
            - reason: Description of each bug found (trigger condition + impact)
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.code.code_bug_detection import CodeBugDetectionGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = CodeBugDetectionGrader(model=model, threshold=3)
        >>>
        >>> # Buggy code: off-by-one + missing empty list check
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Return the second largest element in a list.",
        ...     response='''
        ... def second_largest(nums):
        ...     nums.sort()
        ...     return nums[-2]
        ... ''',
        ... ))
        >>> print(result.score)   # 2 - crashes on empty list, returns wrong value for duplicates
        >>> print(result.reason)  # "Off-by-one on empty list: IndexError when len < 2. ..."
        >>>
        >>> # Correct code with edge case handling
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Return the second largest element in a list.",
        ...     response='''
        ... def second_largest(nums):
        ...     if len(nums) < 2:
        ...         raise ValueError("Need at least 2 elements")
        ...     unique = sorted(set(nums), reverse=True)
        ...     if len(unique) < 2:
        ...         raise ValueError("Need at least 2 distinct elements")
        ...     return unique[1]
        ... ''',
        ... ))
        >>> print(result.score)   # 5 - handles edge cases correctly
    """

    DEFAULT_TEMPLATE = DEFAULT_CODE_BUG_DETECTION_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 3,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize CodeBugDetectionGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [1, 5] (default: 3)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_CODE_BUG_DETECTION_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 5]
        """
        if not 1 <= threshold <= 5:
            raise ValueError(f"threshold must be in range [1, 5], got {threshold}")

        super().__init__(
            name="code_bug_detection",
            mode=GraderMode.POINTWISE,
            description="Detect potential bugs in AI-generated code without requiring test cases",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )
        self.threshold = threshold

    async def _aevaluate(
        self,
        query: str,
        response: str,
        **kwargs,
    ) -> GraderScore:
        """
        Evaluate code for potential bugs.

        Args:
            query: Task description or prompt that produced the code
            response: AI-generated code to evaluate
            **kwargs: Additional keyword arguments passed to the model

        Returns:
            GraderScore: Score [1, 5] where 5 = no bugs detected,
                        1 = critical bugs that break primary functionality

        Example:
            >>> result = await grader.aevaluate(
            ...     query="Implement a stack with push, pop, and peek.",
            ...     response="class Stack:\\n    def pop(self): return self.data.pop()",
            ... )
            >>> # score=2: pop() crashes on empty stack (no guard), missing push/peek
        """
        try:
            result = await super()._aevaluate(
                query=query,
                response=response,
            )
            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata={**result.metadata, "threshold": self.threshold},
            )
        except Exception as e:
            logger.exception(f"Error evaluating code bugs: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["CodeBugDetectionGrader", "DEFAULT_CODE_BUG_DETECTION_TEMPLATE"]
