# -*- coding: utf-8 -*-
"""
Code Complexity Grader

Evaluates whether AI-generated code is unnecessarily complex, verbose, or convoluted —
a systematic failure mode where LLMs over-engineer solutions, add redundant abstractions,
use deeply nested logic, and produce maintenance-heavy code for simple tasks.

Inspired by pr-agent's `estimated_effort_to_review_[1-5]` metric and its `maintainability`
improvement label, which together surface code that is correct but overly hard to understand,
extend, or maintain.
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
CODE_COMPLEXITY_PROMPT_EN = textwrap.dedent(
    """
You are an expert software engineer tasked with evaluating whether AI-generated code is
unnecessarily complex, verbose, or over-engineered relative to the task it solves.

AI models frequently exhibit a systematic failure mode: they produce correct code that is
far more complicated than the problem demands — adding unnecessary abstractions, excessive
class hierarchies, redundant intermediate variables, deeply nested logic, over-generalization,
and boilerplate that obscures the actual logic. Your job is to detect this failure and score
the code based on how well its complexity matches the task requirements.

<Rubrics>
Appropriately simple code should:
- Solve the task in the fewest meaningful steps without sacrificing readability.
- Use flat, linear control flow where nesting is not required.
- Avoid creating helper functions, classes, or abstractions unless they genuinely simplify
  the solution or are explicitly required by the task.
- Not introduce intermediate variables whose sole purpose is to rename an expression.
- Use built-in language features and standard library functions instead of reimplementing them.
- Avoid premature generalization (e.g., configurable parameters, plugin architectures, or
  factory patterns for a one-off task).
- Keep function length proportionate to the task — short tasks warrant short functions.
- Not repeat logic that could be expressed once more concisely.

Complexity penalties should be applied for:
- Unnecessary class or object wrappers around what should be a simple function.
- Excessive abstraction layers (base classes, interfaces, mixins) not required by the task.
- Deeply nested conditions or loops (3+ levels) when flat logic would suffice.
- Redundant intermediate variables that add lines without adding clarity.
- Over-parameterization: adding flags, modes, or config options not asked for.
- Reimplementing functionality already available in the standard library or language builtins.
- Splitting a simple algorithm across multiple functions when one would be clearer.
- Verbose error handling machinery (custom exception hierarchies, retry decorators) for a
  simple script or utility function.
- Dead code, commented-out alternatives, or speculative future features.
- Long docstrings or comment blocks that merely restate what the code already makes obvious.
</Rubrics>

<Steps>
- Read the task description carefully to understand what is actually required.
- Estimate the "ideal" implementation complexity: how many lines and concepts should a
  clean, experienced engineer need for this task?
- Compare this to the submitted code:
  * Count abstraction layers (functions, classes, decorators, mixins).
  * Note nesting depth of conditions and loops.
  * Identify redundant variables, repeated logic, and unnecessary indirection.
  * Check for standard library calls that could replace hand-rolled implementations.
- Assess the gap between required complexity and actual complexity.
- Assign a score based on how well the code's complexity matches the task.
</Steps>

<Constraints>
Evaluate complexity relative to the task, not in absolute terms. A complex task (e.g.,
implementing a full state machine) warrants complex code — penalize only when the complexity
exceeds what the task justifiably requires. Do not penalize for correct, well-named helper
functions that genuinely improve clarity. Focus on over-engineering, not under-engineering
(missing features are a correctness concern, not a complexity concern).
</Constraints>

<Scale>
- 5: The complexity perfectly matches the task requirements. The code is clean, concise,
  and no simpler solution exists without sacrificing clarity or correctness.
- 4: Minor unnecessary complexity — one or two small redundancies or slightly verbose
  patterns that could be simplified but do not significantly impact readability.
- 3: Noticeably over-engineered. Extra abstractions, unnecessary functions, or deeper nesting
  than needed, making the code harder to read than the task warrants. A reviewer would flag
  this for simplification.
- 2: Significantly over-engineered. Unnecessary class hierarchies, excessive abstraction
  layers, or a solution that is 2-3x longer than necessary, obscuring the core logic.
- 1: Extremely convoluted. The solution is unrecognizably complex for the task — multiple
  unnecessary design patterns, reimplemented standard library functions, deeply nested logic,
  or a framework-like structure built for a trivial problem.
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
    "reason": "<concise explanation. Describe the ideal complexity for this task,
    then identify specific over-engineering patterns found
    (e.g., unnecessary classes, redundant variables, deep nesting,
    reimplemented builtins).
    If the code is appropriately simple, confirm that.>",
    "score": <integer between 1 and 5, where 5 means perfectly
    appropriate complexity and 1 means extremely over-engineered>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
CODE_COMPLEXITY_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的软件工程师，负责评估AI生成的代码相对于其解决的任务是否过于复杂、冗长或过度设计。

AI模型经常表现出一种系统性失败模式：它们产生正确的代码，但远比问题所需要的复杂——
添加不必要的抽象、过多的类层次结构、冗余的中间变量、深度嵌套逻辑、过度泛化，
以及掩盖实际逻辑的样板代码。
你的工作是检测这种失败并根据代码复杂性与任务需求的匹配程度对代码进行评分。

<评分标准>
适当简洁的代码应该：
- 在不牺牲可读性的前提下，用最少的有意义步骤解决任务。
- 在不需要嵌套的情况下，使用扁平、线性的控制流。
- 避免创建辅助函数、类或抽象，除非它们真正简化了解决方案或任务明确要求。
- 不引入唯一目的是重命名表达式的中间变量。
- 使用内置语言特性和标准库函数，而不是重新实现它们。
- 避免过早泛化（例如，为一次性任务配置参数、插件架构或工厂模式）。
- 保持函数长度与任务成比例——简单的任务需要简短的函数。
- 不重复本可以更简洁地表达一次的逻辑。

以下情况应扣分（复杂性惩罚）：
- 对应该是简单函数的内容进行不必要的类或对象封装。
- 任务不需要的过多抽象层（基类、接口、混入）。
- 扁平逻辑就足够时，却使用3层以上的深度嵌套条件或循环。
- 增加行数但不增加清晰度的冗余中间变量。
- 过度参数化：添加未被要求的标志、模式或配置选项。
- 重新实现标准库或语言内置功能中已有的功能。
- 将简单算法拆分到多个函数中，而用一个函数会更清晰。
- 简单脚本或工具函数中有冗长的错误处理机制（自定义异常层次结构、重试装饰器）。
- 死代码、注释掉的替代方案或推测性的未来功能。
- 仅重述代码已明确表达内容的冗长文档字符串或注释块。
</评分标准>

<评估步骤>
- 仔细阅读任务描述，了解实际需要什么。
- 估计"理想"的实现复杂度：一位经验丰富的工程师需要多少行和概念来完成这个任务？
- 将其与提交的代码进行比较：
  * 计算抽象层数（函数、类、装饰器、混入）。
  * 注意条件和循环的嵌套深度。
  * 识别冗余变量、重复逻辑和不必要的间接层。
  * 检查是否存在可以替代手工实现的标准库调用。
- 评估所需复杂度与实际复杂度之间的差距。
- 根据代码复杂性与任务的匹配程度分配分数。
</评估步骤>

<注意事项>
相对于任务而非绝对地评估复杂性。复杂的任务（例如，实现完整的状态机）需要复杂的代码——
只有当复杂性超过任务合理要求时才扣分。
不要因真正提高清晰度的正确、命名良好的辅助函数而扣分。
专注于过度设计，而不是设计不足（缺少功能是正确性问题，不是复杂性问题）。
</注意事项>

<评分量表>
- 5: 复杂性与任务需求完美匹配。代码简洁明了，在不牺牲清晰度或正确性的情况下，不存在更简单的解决方案。
- 4: 轻微的不必要复杂性——一两个小的冗余或稍微冗长的模式，可以简化但不会显著影响可读性。
- 3: 明显过度设计。额外的抽象、不必要的函数或比所需更深的嵌套，使代码比任务所需更难阅读。审查者会标记这个问题以进行简化。
- 2: 显著过度设计。不必要的类层次结构、过多的抽象层，或比必要长2-3倍的解决方案，掩盖了核心逻辑。
- 1: 极度复杂。解决方案对于任务来说复杂到无法辨认——多种不必要的设计模式、重新实现的标准库函数、深度嵌套的逻辑，或为琐碎问题构建的框架式结构。
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
    "reason": "<简要说明。描述此任务的理想复杂度，然后识别发现的具体过度设计模式（例如，不必要的类、冗余变量、深度嵌套、重新实现的内置功能）。如果代码适当简洁，请确认这一点。>",
    "score": <1到5之间的整数，其中5表示复杂性完全适当，1表示极度过度设计>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_CODE_COMPLEXITY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=CODE_COMPLEXITY_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=CODE_COMPLEXITY_PROMPT_ZH,
            ),
        ],
    },
)


class CodeComplexityGrader(LLMGrader):
    """
    Code Complexity Grader

    Purpose:
        Evaluates whether AI-generated code is over-engineered or unnecessarily complex
        relative to the task it solves. Targets the systematic LLM failure mode of producing
        correct but bloated code — excessive abstractions, deep nesting, redundant indirection,
        and framework-like structures built for trivial problems.

        Inspired by pr-agent's `estimated_effort_to_review_[1-5]` dimension and `maintainability`
        improvement label, which together flag code that is technically correct but disproportionately
        hard to understand and maintain.

    What it evaluates:
        - Over-abstraction: Unnecessary classes, base classes, interfaces, factory patterns
        - Nesting Depth: Deeply nested conditions/loops when flat logic would suffice
        - Redundancy: Intermediate variables that rename without clarifying, duplicated logic
        - Over-parameterization: Config flags, modes, or extension points not asked for
        - Reinventing the wheel: Re-implementing stdlib/builtin functionality from scratch
        - Proportionality: Function and file length vs. task complexity
        - Dead Code: Commented-out alternatives, speculative future features

    When to use:
        - Benchmarking LLM tendency to over-engineer across tasks of varying complexity
        - Identifying models that produce maintainable vs. bloated code
        - Evaluating code generation quality beyond functional correctness
        - Code review pipelines where maintainability is a first-class concern
        - Comparing model outputs to measure conciseness and simplicity

    Scoring (higher = more appropriately simple):
        - 5: Complexity perfectly matches task requirements; clean and concise
        - 4: Minor redundancies that don't significantly impact readability
        - 3: Noticeably over-engineered; a reviewer would flag for simplification
        - 2: Significantly bloated; 2-3x longer than necessary, obscures core logic
        - 1: Extremely convoluted; unrecognizable complexity for the given task

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 5] to pass (default: 3)
        template: Custom evaluation template (default: DEFAULT_CODE_COMPLEXITY_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: Evaluation strategy (default: DirectEvaluationStrategy)

    Returns:
        GraderScore with:
            - score: [1, 5] where 5 = appropriate complexity, 1 = extremely over-engineered
            - reason: Ideal complexity estimate + specific over-engineering patterns found
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.code.code_complexity import CodeComplexityGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = CodeComplexityGrader(model=model, threshold=3)
        >>>
        >>> # Over-engineered: factory + abstract base class for a simple sum function
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Write a function that returns the sum of a list of numbers.",
        ...     response='''
        ... from abc import ABC, abstractmethod
        ... from typing import List, Union
        ...
        ... class BaseAggregator(ABC):
        ...     @abstractmethod
        ...     def aggregate(self, values: List[Union[int, float]]) -> Union[int, float]:
        ...         pass
        ...
        ... class SumAggregator(BaseAggregator):
        ...     def aggregate(self, values: List[Union[int, float]]) -> Union[int, float]:
        ...         result = 0
        ...         for value in values:
        ...             result = result + value
        ...         return result
        ...
        ... class AggregatorFactory:
        ...     @staticmethod
        ...     def create(strategy: str = "sum") -> BaseAggregator:
        ...         if strategy == "sum":
        ...             return SumAggregator()
        ...         raise ValueError(f"Unknown strategy: {strategy}")
        ...
        ... def sum_numbers(numbers: List[Union[int, float]]) -> Union[int, float]:
        ...     factory = AggregatorFactory()
        ...     aggregator = factory.create("sum")
        ...     return aggregator.aggregate(numbers)
        ... ''',
        ... ))
        >>> print(result.score)   # 1 - factory + ABC + class for `sum(numbers)`
        >>> print(result.reason)  # "Ideal: `return sum(numbers)`. Found: unnecessary ABC, factory..."
        >>>
        >>> # Appropriately simple
        >>> result = asyncio.run(grader.aevaluate(
        ...     query="Write a function that returns the sum of a list of numbers.",
        ...     response="def sum_numbers(numbers):\\n    return sum(numbers)",
        ... ))
        >>> print(result.score)   # 5 - perfectly concise
    """

    DEFAULT_TEMPLATE = DEFAULT_CODE_COMPLEXITY_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 3,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize CodeComplexityGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [1, 5] (default: 3)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_CODE_COMPLEXITY_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 5]
        """
        if not 1 <= threshold <= 5:
            raise ValueError(f"threshold must be in range [1, 5], got {threshold}")

        super().__init__(
            name="code_complexity",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether AI-generated code is unnecessarily complex or over-engineered",
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
        Evaluate code for unnecessary complexity and over-engineering.

        Args:
            query: Task description or prompt that produced the code
            response: AI-generated code to evaluate
            **kwargs: Additional keyword arguments passed to the model

        Returns:
            GraderScore: Score [1, 5] where 5 = appropriate complexity,
                        1 = extremely over-engineered relative to the task

        Example:
            >>> result = await grader.aevaluate(
            ...     query="Check if a string is a palindrome.",
            ...     response=(
            ...         "class PalindromeChecker:\\n"
            ...         "    def __init__(self, strategy='default'):\\n"
            ...         "        self.strategy = strategy\\n"
            ...         "    def check(self, s: str) -> bool:\\n"
            ...         "        cleaned = self._preprocess(s)\\n"
            ...         "        return cleaned == cleaned[::-1]\\n"
            ...         "    def _preprocess(self, s: str) -> str:\\n"
            ...         "        return s.lower().replace(' ', '')\\n"
            ...     ),
            ... )
            >>> # score=3: class wrapper unnecessary for a one-liner lambda/function
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
            logger.exception(f"Error evaluating code complexity: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["CodeComplexityGrader", "DEFAULT_CODE_COMPLEXITY_TEMPLATE"]
