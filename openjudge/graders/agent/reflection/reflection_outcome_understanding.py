# -*- coding: utf-8 -*-
"""
Reflection Outcome Understanding Grader

Evaluates whether the agent correctly understands and interprets the outcome or result of an action
in its reflection module.
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
REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_EN = textwrap.dedent(
    """
You are an expert in analyzing agent behavior. Your task is to evaluate whether the agent correctly understands and interprets the outcome or result of an action in its reflection.

<Evaluation Type: Reflection Outcome Understanding>
The agent should correctly understand and interpret the outcome or result of an action in its reflection. This occurs when the agent receives an observation indicating a specific result, and in the reflection module, the agent correctly understands what that observation means.
</Evaluation Type>

<Rubrics for Evaluation>
1. Factual Accuracy: The agent's reflection must accurately describe what the observation states, without distorting or ignoring information
2. Success/Failure Recognition: The agent correctly identifies whether an action succeeded or failed based on explicit signals in the observation
3. State Change Understanding: The agent accurately understands what state changes occurred (or did not occur) from an action
4. Evidence-Based Reasoning: The agent draws conclusions strictly supported by the observation, without unsupported inferences
5. Observation Scope Awareness: The agent recognizes when observations are partial/incomplete and does not treat them as exhaustive
6. Absence vs. Uncertainty: The agent distinguishes between "confirmed absence" and "unknown/unverified" status
7. No Premature Conclusions: The agent does not conclude something does NOT exist based solely on seeing something else
8. No Fabrication: The agent does not claim to observe information that does not actually appear in the observation text
9. Format Recognition: The agent correctly identifies when observation content is corrupted, unreadable, or in an unusable format
</Rubrics>

<Evaluation Criteria>
Critical Evaluation Principles (apply in order):

1. Truth Preservation Test: Does the reflection accurately mirror the factual content of the observation without distortion?
   - If observation says "Nothing happens", reflection CANNOT claim success
   - If observation shows "item X in location", reflection CANNOT claim "no item Y" without further verification

2. Failure Signal Detection: Does the agent correctly interpret negative outcomes?
   - "Nothing happens" = action failed
   - "You see X" (when looking for Y) ≠ "Y does not exist"

3. Completeness Awareness: Does the agent recognize partial vs. complete observations?
   - Opening a container and seeing one item ≠ seeing all items
   - Going to a location ≠ thoroughly examining the location

4. Inference Validation: Are the agent's conclusions logically supported by observations?
   - Supported: "I tried to take X but nothing happened, so I don't have X yet"
   - Unsupported: "I tried to take X → (observes 'nothing happens') → I now have X"
   - Unsupported: "I see bread → there is no lettuce"

5. Fabrication Detection: Does the reflection claim to have observed information not present in the observation?
   - If observation contains binary/encoded data or error messages, reflection CANNOT report specific parsed values
   - Agent must distinguish between "data extraction succeeded" vs "data extraction failed"

6. Format Validity Check: Does the agent recognize when observation content is corrupted or unreadable?
   - Binary data, encoding errors, or empty responses indicate extraction failure
   - Agent should acknowledge the need for alternative approaches when data cannot be parsed
</Evaluation Criteria>

{context}

{history}

<Current Step>
Observation: {observation}
Reflection: {reflection}
</Current Step>

# Scoring Instructions

Assign score = 0.0 (poor understanding) if ANY of the following occur:

1. Factual Distortion: Reflection contradicts or ignores explicit observation content
   - Observation: "Nothing happens" → Reflection: "The action was successful"
   - Observation: "You see item X" → Reflection: "Item Y is not here" (without verification)

2. Failure Misinterpretation: Agent claims success when observation indicates failure
   - Treats "Nothing happens" as successful outcome
   - Ignores negative signals like "Nothing happens", "You cannot do that", "Already there"

3. Premature Negative Conclusion: Agent concludes absence/non-existence without sufficient evidence
   - Seeing X leads to concluding "no Y" without exhaustive search
   - Opening container, seeing partial contents, concluding target is absent

4. Scope Overreach: Agent treats partial observation as complete information
   - "Open container → see item A" interpreted as "container only has item A"
   - Navigation without examination interpreted as thorough search

5. Inference Leap: Agent's conclusions not logically supported by observation
   - Drawing definitive conclusions from ambiguous or incomplete data

6. Fabricated Information: Agent claims to have extracted specific data that does not appear in observation
   - Observation contains binary/corrupted text → Reflection reports specific parsed values
   - Observation shows error or empty result → Reflection claims data was successfully retrieved

7. Format Misinterpretation: Agent fails to recognize when observation data is unusable
   - Treating binary encoding, corrupted output, or error messages as valid extractable data
   - Claiming successful data parsing when extraction clearly failed

Assign score = 1.0 (good understanding) if ALL of the following are met:
- Reflection accurately describes what the observation states (no distortion)
- Success/failure correctly identified based on observation signals
- Conclusions are evidence-based and logically supported
- Agent acknowledges uncertainty when observations are incomplete
- Agent recognizes when further actions (examine, search) are needed

Evaluation Priority: Truth preservation is paramount. Any discrepancy between observation facts and reflection interpretation → score = 0.0

Provide your evaluation in the following structured JSON format:
{{
    "score": <0.0 or 1.0>,
    "reason": "<detailed explanation of outcome understanding quality and confidence level>"
}}

JSON:
"""
).strip()

# Chinese Prompt
REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_ZH = textwrap.dedent(
    """
你是一名分析智能体行为的专家。你的任务是评估智能体是否在其反思中正确理解和解释了动作的结果或输出。

<评估类型：反思结果理解>
智能体应该在其反思中正确理解和解释动作的结果或输出。这发生在智能体收到表明特定结果的观察时，并且在反思模块中，智能体正确理解了该观察的含义。
</评估类型>

<评估准则>
1. 事实准确性：智能体的反思必须准确描述观察所陈述的内容，不能扭曲或忽略信息
2. 成功/失败识别：智能体根据观察中的明确信号正确识别动作是成功还是失败
3. 状态变化理解：智能体准确理解动作产生了哪些状态变化（或没有产生变化）
4. 基于证据的推理：智能体得出的结论必须严格基于观察支持，不能有无根据的推断
5. 观察范围意识：智能体能够识别观察何时是部分/不完整的，不会将其视为详尽信息
6. 缺失与不确定性：智能体能够区分"确认不存在"与"未知/未验证"状态
7. 无过早结论：智能体不会仅仅因为看到某物就得出其他东西不存在的结论
8. 无虚构信息：智能体不会声称观察到了实际上并未出现在观察文本中的信息
9. 格式识别：智能体能正确识别观察内容何时是损坏的、不可读的或不可用格式
</评估准则>

<评估标准>
关键评估原则（按顺序应用）：

1. 真实性保持测试：反思是否准确地镜像了观察的事实内容而没有扭曲？
   - 如果观察说"Nothing happens"，反思不能声称成功
   - 如果观察显示"位置中有物品X"，反思不能在没有进一步验证的情况下声称"没有物品Y"

2. 失败信号检测：智能体是否正确解释了负面结果？
   - "Nothing happens" = 动作失败
   - "You see X"（当寻找Y时）≠ "Y不存在"

3. 完整性意识：智能体是否识别出部分观察与完整观察的区别？
   - 打开容器看到一个物品 ≠ 看到所有物品
   - 到达某个位置 ≠ 彻底检查该位置

4. 推断验证：智能体的结论是否在逻辑上得到观察的支持？
   - 支持的："我尝试拿X但什么都没发生，所以我还没有X"
   - 不支持的："我尝试拿X → （观察到'nothing happens'）→ 我现在有X了"
   - 不支持的："我看到面包 → 没有生菜"

5. 虚构信息检测：反思是否声称观察到了实际并未出现在观察中的信息？
   - 如果观察包含二进制/编码数据或错误消息，反思不能报告具体解析的数值
   - 智能体必须区分"数据提取成功"与"数据提取失败"

6. 格式有效性检查：智能体是否识别出观察内容何时是损坏或不可读的？
   - 二进制数据、编码错误或空响应表明提取失败
   - 当数据无法解析时，智能体应该承认需要替代方法
</评估标准>

{context}

{history}

<当前步骤>
观察：{observation}
反思：{reflection}
</当前步骤>

# 评分指令

以下任一情况出现则评分 = 0.0（理解不佳）：

1. 事实扭曲：反思与观察的明确内容相矛盾或忽略观察内容
   - 观察："Nothing happens" → 反思："动作成功了"
   - 观察："你看到物品X" → 反思："物品Y不在这里"（未经验证）

2. 失败误判：智能体在观察表明失败时声称成功
   - 将"Nothing happens"视为成功结果
   - 忽略负面信号如"Nothing happens"、"You cannot do that"、"Already there"

3. 过早的否定结论：智能体在没有充分证据的情况下得出缺失/不存在的结论
   - 看到X就得出"没有Y"的结论而没有详尽搜索
   - 打开容器看到部分内容就得出目标不存在的结论

4. 范围过度扩展：智能体将部分观察视为完整信息
   - "打开容器 → 看到物品A"被解释为"容器只有物品A"
   - 导航而没有检查被解释为彻底搜索

5. 推断跳跃：智能体的结论没有得到观察的逻辑支持
   - 从模糊或不完整的数据得出明确结论

6. 虚构信息：智能体声称提取了观察中实际并未出现的具体数据
   - 观察包含二进制/损坏文本 → 反思报告具体解析的数值
   - 观察显示错误或空结果 → 反思声称数据已成功检索

7. 格式误判：智能体未能识别出观察数据不可用
   - 将二进制编码、损坏输出或错误消息视为有效的可提取数据
   - 在提取明显失败时声称数据解析成功

以下所有条件满足则评分 = 1.0（良好理解）：
- 反思准确描述了观察所陈述的内容（无扭曲）
- 基于观察信号正确识别了成功/失败
- 结论基于证据且逻辑支持充分
- 智能体在观察不完整时承认不确定性
- 智能体识别出何时需要进一步动作（examine、search）

评估优先级：真实性保持是首要的。观察事实与反思解释之间的任何差异 → 评分 = 0.0

请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <0.0 或 1.0>,
    "reason": "<关于结果理解质量的详细解释和置信度水平>"
}}

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=REFLECTION_OUTCOME_UNDERSTANDING_PROMPT_ZH,
            ),
        ],
    },
)


class ReflectionOutcomeUnderstandingGrader(LLMGrader):
    """
    Reflection Outcome Understanding Grader

    Evaluates whether the agent correctly understands and interprets the outcome or result of an action
    in its reflection module.

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
        >>> grader = ReflectionOutcomeUnderstandingGrader(
        ...     model=api,
        ...     language=LanguageEnum.EN
        ... )
        >>> result = asyncio.run(grader.aevaluate(
        ...     observation="The drawer is now open.",
        ...     reflection="I successfully opened the drawer."
        ... )
        >>> print(f"Score: {result.score}")  # Expected: 1.0
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        super().__init__(
            name="reflection_outcome_understanding",
            mode=GraderMode.POINTWISE,
            description="Evaluate reflection outcome understanding",
            model=model,
            template=template or DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE,
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
        Evaluate reflection outcome understanding

        Args:
            observation: Agent's observation from the environment
            reflection: Agent's reflection on the situation
            history: Optional list of previous step dictionaries for context
            context: Optional task context (task description, environment, available actions)
            **kwargs: Additional arguments

        Returns:
            GraderScore: Score with binary value (1.0 = good understanding, 0.0 = poor understanding)

        Example:
            >>> result = await grader.aevaluate(
            ...     observation="The drawer is now open.",
            ...     reflection="I successfully opened the drawer.",
            ...     context="Task: Open the drawer"
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
            logger.error(f"Error evaluating reflection outcome understanding: {e}")
            normalized_score = 0.0
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "raw_score": score,
            "evaluation_type": "reflection_outcome_understanding",
        }

        return GraderScore(
            name=self.name,
            score=normalized_score,
            reason=reason,
            metadata=metadata,
        )


__all__ = [
    "ReflectionOutcomeUnderstandingGrader",
    "DEFAULT_REFLECTION_OUTCOME_UNDERSTANDING_TEMPLATE",
]
