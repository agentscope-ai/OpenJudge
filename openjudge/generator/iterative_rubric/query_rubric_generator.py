# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""
Query-Specific Rubric Generator with Full Iterative Capability

This module contains the complete logic for query-specific rubric generation:
- Generate rubrics
- Evaluate data
- Validate results
- Revise rubrics
- Iterative improvement loop

All prompts are defined inline in this module.
"""

import textwrap
from typing import Any, Dict, List

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from openjudge.graders.base_grader import GraderMode, GraderRank, GraderScore
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# ========== Pointwise Generation Prompts ==========

POINTWISE_GENERATION_PROMPT_ZH = """
请基于以下样本内容和标注信息，生成**恰好 {generate_number} 个**针对性的评估rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: Pointwise (对单个回答独立评分，范围{min_score}-{max_score}，必须是整数)
- 仔细分析单个回答与评分范围的差异
- 生成能够区分单个回答与评分范围的评估标准
- 标准应该明确、具体、可操作
- 评分标准应该能够产生区间内的整数分数
- **重要：必须严格生成 {generate_number} 个评估标准，不多不少**

## 输出格式
请严格按照以下JSON格式输出：
{{
"rubrics": [
    "第一个评估标准的详细描述",
    ... (共 {generate_number} 个)
],
"reason": "生成这些评估标准的原因和依据"
}}

**关键提醒：rubrics数组必须恰好包含 {generate_number} 个元素。**

请生成评估标准
"""

POINTWISE_GENERATION_PROMPT_EN = """
Based on the following sample content and annotations, generate **exactly {generate_number}** targeted evaluation rubrics.
{task_description_section}
{sample_content}

## Task Requirements
- Evaluation Mode: Pointwise (score individual responses independently, range {min_score}-{max_score}, must be integers)
- Carefully analyze the differences between single response and score range
- Generate evaluation criteria that can distinguish single response and score range
- Criteria should be clear, specific, and actionable
- Evaluation criteria should be able to produce integer scores within the range
- **IMPORTANT: You MUST generate exactly {generate_number} evaluation criteria, no more, no less**

## Output Format
Please output strictly in the following JSON format:
{{
"rubrics": [
    "Detailed description of the first evaluation criterion",
    ... (exactly {generate_number} items)
],
"reason": "Reason and basis for generating these evaluation criteria"
}}

**CRITICAL: The rubrics array MUST contain exactly {generate_number} elements.**

Please generate evaluation criteria:
"""

# ========== Listwise Generation Prompts ==========

LISTWISE_GENERATION_PROMPT_ZH = """
请基于以下样本内容和标注信息，生成**恰好 {generate_number} 个**针对性的排序rubrics。
{task_description_section}
{sample_content}

## 任务要求
- 评估模式: Listwise (对多个回答进行整体排序)
- 仔细分析多个回答的差异，包括"优质回答"和"劣质回答"的区别
- 生成能够正确排序回答质量的标准
- 标准应该能够确定回答的相对质量顺序
- 注意：rank值越小表示质量越好（rank=1是最好的），按升序排列
- **重要：必须严格生成 {generate_number} 个排序标准，不多不少**

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "第一个排序标准的详细描述",
        ... (共 {generate_number} 个)
    ],
    "reason": "生成这些排序标准的原因和依据"
}}

**关键提醒：rubrics数组必须恰好包含 {generate_number} 个元素。**

请生成排序标准
"""

LISTWISE_GENERATION_PROMPT_EN = """
Based on the following sample content and annotations, generate **exactly {generate_number}** targeted ranking rubrics.
{task_description_section}
{sample_content}

## Task Requirements
- Evaluation Mode: Listwise (rank multiple responses holistically)
- Carefully analyze the differences between multiple responses, including "High-quality" and "Low-quality" responses
- Generate ranking criteria that can correctly order response quality
- Criteria should determine the relative quality order of responses
- Note: Smaller rank values indicate better quality (rank=1 is best), sort in ascending order
- **IMPORTANT: You MUST generate exactly {generate_number} ranking criteria, no more, no less**

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first ranking criterion",
        ... (exactly {generate_number} items)
    ],
    "reason": "Reason and basis for generating these ranking criteria"
}}

**CRITICAL: The rubrics array MUST contain exactly {generate_number} elements.**

Please generate ranking criteria
"""

# ========== Pointwise Evaluation Prompts ==========

POINTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准对回答评分。
{task_description_section}
评估标准:
{rubrics}

查询: {query}
回答: {response}

评分范围: {min_score} 到 {max_score}

## 输出格式
请严格按照以下JSON格式输出：
{{
    "score": 分数值（必须是{min_score}到{max_score}范围内的整数）,
    "reason": "评分的详细理由和依据"
}}

重要提醒：score 必须是整数，不能是小数或其他格式。

请输出评分结果
"""

POINTWISE_EVALUATION_PROMPT_EN = """
Please score the response based on the evaluation criteria.
{task_description_section}
Evaluation Criteria:
{rubrics}

Query: {query}
Response: {response}

Score Range: {min_score} to {max_score}

## Output Format
Please output strictly in the following JSON format:
{{
    "score": score_value (must be an integer between {min_score} and {max_score}),
    "reason": "Detailed reasoning and basis for the score"
}}

Important reminder: The score must be an integer, not a decimal or any other format.

Please output the scoring result
"""

# ========== Listwise Evaluation Prompts ==========

LISTWISE_EVALUATION_PROMPT_ZH = """
请根据评估标准对所有回答进行排序。
{task_description_section}
评估标准:
{rubrics}

查询: {query}

所有回答（共 {num_responses} 个回答）:
{responses}

## 任务要求
- 你正在评估 {num_responses} 个回答。你的输出必须包含恰好 {num_responses} 个rank值。
- 根据评估标准，对所有回答进行质量评估
- 为每个回答分配一个rank值，数值越小表示质量越好（rank=1是最好的）
- 保持回答的原始顺序，只输出每个回答对应的rank值
- 重要：任何两个回答的rank值都不能相同，必须严格区分质量差异，不允许平分
- 重要：rank值必须是从1到{num_responses}的连续整数
- 重要：不要有任何位置偏见 - 回答的位置（第一个还是第二个）不应该影响你的判断。只关注内容质量本身。

## 示例（针对2个回答）
示例A：如果回答2比回答1更好
- 回答1得rank=2（较差）
- 回答2得rank=1（最好）
输出：[2, 1]

示例B：如果回答1比回答2更好
- 回答1得rank=1（最好）
- 回答2得rank=2（较差）
输出：[1, 2]

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rank": [每个回答的rank值，恰好{num_responses}个值],
    "reason": "详细说明每个回答的质量评估和rank分配理由"
}}

关键提醒：你的rank数组必须恰好包含 {num_responses} 个元素，每个回答对应一个。
"""

LISTWISE_EVALUATION_PROMPT_EN = """
Please rank all responses based on the evaluation criteria.
{task_description_section}
Evaluation Criteria:
{rubrics}

Query: {query}

All Responses (Total: {num_responses} responses):
{responses}

## Task Requirements
- You are evaluating exactly {num_responses} responses. Your output must contain exactly {num_responses} rank values.
- Evaluate all responses based on the evaluation criteria
- Assign a rank value to each response, smaller values indicate better quality (rank=1 is best)
- Keep responses in original order, only output corresponding rank values
- Important: No two responses can have the same rank value, must strictly distinguish quality differences, no ties allowed
- Important: Rank values must be consecutive integers from 1 to {num_responses}
- Important: Do NOT have any position bias - the position of a response (first or second) should NOT influence your judgment. Focus ONLY on the content quality.

## Examples (for 2 responses)
Example A: If Response 2 is better than Response 1
- Response 1 gets rank=2 (worse)
- Response 2 gets rank=1 (best)
Output: [2, 1]

Example B: If Response 1 is better than Response 2
- Response 1 gets rank=1 (best)
- Response 2 gets rank=2 (worse)
Output: [1, 2]

## Output Format
Please output strictly in the following JSON format:
{{
    "rank": [rank values for each response, exactly {num_responses} values],
    "reason": "Detailed explanation of quality assessment and rank assignment for each response"
}}

CRITICAL: Your rank array MUST have exactly {num_responses} elements, one for each response.
"""

# ========== Pointwise Revision Prompts ==========

POINTWISE_REVISION_PROMPT_ZH = """
之前生成的Pointwise评分标准在验证时失败了，请生成{generate_number}个评分标准，并根据详细反馈进行改进。
{task_description_section}
{sample_content}

## 之前的评分标准
{rubrics}

## 验证失败的详细反馈
{feedback}

## Pointwise模式的改进要求
1. 分析失败原因：
   - 为什么当前标准没能给出正确的分数？
   - 是否忽略了某些关键的质量维度？
   - 标准是否过于宽松或严格？

2. 重点改进方向：
   - 仔细对比期望分数与实际分数的差异
   - 识别高分回答的核心优势（准确性、完整性、清晰度等）
   - 识别低分回答的关键缺陷（错误、遗漏、模糊等）
   - 确保标准能够精确区分不同分数档位

3. 标准制定原则：
   - 每个标准应该明确、可量化、可操作
   - 标准应该覆盖多个评估维度
   - 标准应该能够准确反映期望分数
   - 评分必须是整数，标准应该清晰定义每个整数分数档位的要求

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "改进后的第一个评分标准的详细描述",
        ... (共 {generate_number} 个)
    ],
    "reason": "改进这些评分标准的原因和依据"
}}

**关键提醒：rubrics数组必须恰好包含 {generate_number} 个元素。**

请生成改进后的Pointwise评分标准
"""

POINTWISE_REVISION_PROMPT_EN = """
The previously generated Pointwise scoring criteria failed validation. Please generate {generate_number} improved scoring criteria based on detailed feedback.
{task_description_section}
{sample_content}

## Previous Scoring Criteria
{rubrics}

## Detailed Validation Failure Feedback
{feedback}

## Improvement Requirements for Pointwise Mode
1. Analyze Failure Reasons:
   - Why didn't the current criteria produce the correct scores?
   - Were any key quality dimensions overlooked?
   - Are the criteria too lenient or too strict?

2. Key Improvement Directions:
   - Carefully compare expected scores with actual scores
   - Identify core strengths of high-scoring responses (accuracy, completeness, clarity, etc.)
   - Identify key deficiencies of low-scoring responses (errors, omissions, vagueness, etc.)
   - Ensure criteria can precisely distinguish different score levels

3. Criteria Development Principles:
   - Each criterion should be clear, quantifiable, and actionable
   - Criteria should cover multiple evaluation dimensions
   - Criteria should accurately reflect expected scores
   - Scores must be integers, criteria should clearly define requirements for each integer score level

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first improved scoring criterion",
        ... (exactly {generate_number} items)
    ],
    "reason": "Reason and basis for improving these scoring criteria"
}}

**CRITICAL: The rubrics array MUST contain exactly {generate_number} elements.**

Please generate improved Pointwise scoring criteria:
"""

# ========== Listwise Revision Prompts ==========

LISTWISE_REVISION_PROMPT_ZH = """
之前生成的Listwise排序标准在验证时失败了，请生成{generate_number}个改进的排序标准。
{task_description_section}
{sample_content}

## 之前的排序标准
{rubrics}

## 验证失败的详细反馈
{feedback}

## Listwise模式的改进要求
1. 分析失败原因：
   - 为什么当前标准没能产生正确的排序或比较结果？
   - 排序结果中哪些位置出现了错误？
   - 是否混淆了不同质量层次的回答？

2. 重点改进方向：
   - 仔细分析期望排序与实际排序的差异
   - 识别每个回答的相对质量水平
   - 确保标准能够准确区分所有质量层次

3. 标准制定原则：
   - 每个标准应该能够建立清晰的质量梯度
   - 标准应该能够一致性地对所有回答排序
   - 标准应该覆盖从最优到最差的完整质量范围
   - 标准应该具有判别性，能够明确区分质量差异

## 重要提醒：Listwise评估输出格式
- 为每个回答分配一个rank值，数值越小表示质量越好（rank=1是最好的）
- rank值必须是正整数
- 保持回答的原始顺序，只输出每个回答对应的rank值
- 任何两个回答的rank值都不能相同，必须严格区分质量差异，不允许平分

## 输出格式
请严格按照以下JSON格式输出：
{{
    "rubrics": [
        "改进后的第一个排序标准的详细描述",
        ... (共 {generate_number} 个)
    ],
    "reason": "改进这些排序标准的原因和依据"
}}

**关键提醒：rubrics数组必须恰好包含 {generate_number} 个元素。**

请生成改进后的Listwise排序标准：
"""

LISTWISE_REVISION_PROMPT_EN = """
The previously generated Listwise ranking criteria failed validation. Please generate {generate_number} improved ranking criteria based on detailed feedback.
{task_description_section}
{sample_content}

## Previous Ranking Criteria
{rubrics}

## Detailed Validation Failure Feedback
{feedback}

## Improvement Requirements for Listwise Mode
1. Analyze Failure Reasons:
   - Why didn't the current criteria produce the correct ranking or comparison results?
   - Which positions in the ranking were incorrect?
   - Were different quality levels of responses confused?

2. Key Improvement Directions:
   - Carefully analyze differences between expected and actual rankings
   - Identify relative quality levels of each response
   - Ensure criteria can accurately distinguish all quality levels

3. Criteria Development Principles:
   - Each criterion should establish a clear quality gradient
   - Criteria should consistently rank all responses
   - Criteria should cover the full quality range from best to worst
   - Criteria should be discriminative, clearly distinguishing quality differences

## Important Reminder: Listwise Evaluation Output Format
- Assign a rank value to each response, smaller values indicate better quality (rank=1 is best)
- Rank values must be positive integers
- Keep responses in original order, only output corresponding rank values
- Important: No two responses can have the same rank value, must strictly distinguish quality differences, no ties allowed

## Output Format
Please output strictly in the following JSON format:
{{
    "rubrics": [
        "Detailed description of the first improved ranking criterion",
        ... (exactly {generate_number} items)
    ],
    "reason": "Reason and basis for improving these ranking criteria"
}}

**CRITICAL: The rubrics array MUST contain exactly {generate_number} elements.**

Please generate improved Listwise ranking criteria:
"""

# ========== Build PromptTemplates ==========

POINTWISE_GENERATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
            ChatMessage(
                role="user",
                content=textwrap.dedent(POINTWISE_GENERATION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are a professional evaluation criteria expert.",
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(POINTWISE_GENERATION_PROMPT_EN),
            ),
        ],
    },
)

LISTWISE_GENERATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
            ChatMessage(
                role="user",
                content=textwrap.dedent(LISTWISE_GENERATION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are a professional evaluation criteria expert.",
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(LISTWISE_GENERATION_PROMPT_EN),
            ),
        ],
    },
)

POINTWISE_EVALUATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(POINTWISE_EVALUATION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(POINTWISE_EVALUATION_PROMPT_EN),
            ),
        ],
    },
)

LISTWISE_EVALUATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(LISTWISE_EVALUATION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(LISTWISE_EVALUATION_PROMPT_EN),
            ),
        ],
    },
)

POINTWISE_REVISION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
            ChatMessage(
                role="user",
                content=textwrap.dedent(POINTWISE_REVISION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are a professional evaluation criteria expert.",
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(POINTWISE_REVISION_PROMPT_EN),
            ),
        ],
    },
)

LISTWISE_REVISION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.ZH: [
            ChatMessage(role="system", content="你是一个专业的评估标准制定专家。"),
            ChatMessage(
                role="user",
                content=textwrap.dedent(LISTWISE_REVISION_PROMPT_ZH),
            ),
        ],
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are a professional evaluation criteria expert.",
            ),
            ChatMessage(
                role="user",
                content=textwrap.dedent(LISTWISE_REVISION_PROMPT_EN),
            ),
        ],
    },
)


class RubricGenerationOutput(BaseModel):
    """Output model for rubric generation.

    Represents the structured output from rubric generation operations.

    Attributes:
        rubrics (List[str]): List of generated rubrics.
        reason (str): Reasoning for the generated rubrics.
    """

    rubrics: List[str] = Field(description="List of generated rubrics")
    reason: str = Field(description="Reasoning for the generated rubrics")


class QuerySpecificRubricGenerator:
    """
    Complete query-specific rubric generator with iterative improvement.

    Workflow for each data:
    1. Generate initial rubrics based on data + annotations
    2. Evaluate data using rubrics
    3. Validate evaluation against ground truth
    4. If incorrect, revise rubrics and repeat
    5. Stop when valid or max_epochs reached

    This generator supports both pointwise and listwise evaluation modes,
    and includes built-in validation and revision capabilities for iterative
    improvement of the generated rubrics.
    """

    def __init__(
        self,
        model: BaseChatModel,
        grader_mode: GraderMode | str = GraderMode.POINTWISE,
        generate_number: int = 3,
        max_retries: int = 5,
        max_epochs: int = 3,
        min_score: int = 0,
        max_score: int = 4,
        language: LanguageEnum | str = LanguageEnum.ZH,
        task_description: str | None = None,
    ):
        """
        Initialize generator.

        Args:
            model: Language model for generation and evaluation.
                  Used for all LLM interactions during rubric generation.
            grader_mode: GraderMode enum or string ("pointwise" or "listwise").
                        Determines whether to generate pointwise scoring rubrics
                        or listwise ranking rubrics.
            generate_number: Number of rubrics to generate.
                            Controls how many evaluation criteria are created.
            max_retries: Maximum retry attempts for LLM calls.
                        Used for handling transient failures in LLM interactions.
            max_epochs: Maximum iteration epochs for improvement.
                       Limits how many times rubrics can be revised.
            min_score: Minimum score for pointwise evaluation.
                      Defines the lower bound of the scoring range.
            max_score: Maximum score for pointwise evaluation.
                      Defines the upper bound of the scoring range.
            language: LanguageEnum or string ("zh" or "en").
                     Determines which language to use for prompts.
            task_description: Optional task description to guide rubric generation.
                            Provides context about the evaluation task (e.g.,
                            "Evaluate summaries focusing on coherence and fluency").
        """
        self.model = model

        if isinstance(grader_mode, str):
            self.grader_mode = GraderMode(grader_mode)
        else:
            self.grader_mode = grader_mode

        if isinstance(language, str):
            self.language = (
                LanguageEnum(language) if language in [item.value for item in LanguageEnum] else LanguageEnum.ZH
            )
        else:
            self.language = language

        self.generate_number = generate_number
        self.max_retries = max_retries
        self.max_epochs = max_epochs
        self.min_score = min_score
        self.max_score = max_score
        self.task_description = task_description

        self.generation_template = (
            POINTWISE_GENERATION_TEMPLATE if self.grader_mode == "pointwise" else LISTWISE_GENERATION_TEMPLATE
        )
        self.evaluation_template = (
            POINTWISE_EVALUATION_TEMPLATE if self.grader_mode == "pointwise" else LISTWISE_EVALUATION_TEMPLATE
        )
        self.revision_template = (
            POINTWISE_REVISION_TEMPLATE if self.grader_mode == "pointwise" else LISTWISE_REVISION_TEMPLATE
        )

        logger.info(
            f"QuerySpecificRubricGenerator initialized: mode={grader_mode}, language={language}",
        )

    async def generate_iterative(self, data: dict) -> Dict[str, Any]:
        """
        Complete iterative generation and improvement for a single data.

        Returns:
            Dict with:
            - rubrics: List[str]
            - rubric_valid: True or False
            - rubric_epoch: str (convergence epoch)
            - evaluation_result: Dict
        """
        # Step 1: Initial rubric generation
        rubrics = await self.generate(data)
        if not rubrics:
            return {
                "rubrics": [],
                "rubric_valid": False,
                "rubric_epoch": "0",
                "evaluation_result": {},
            }

        # Step 2: Iterative improvement
        evaluation_result = {}
        for epoch in range(self.max_epochs):
            # Evaluate current rubrics
            evaluation_result = await self.aevaluate(data, rubrics)

            # Validate
            is_correct = self.validate(data, evaluation_result)
            logger.debug(f"Epoch {epoch}: correctness = {is_correct}")

            if is_correct:
                return {
                    "rubrics": rubrics,
                    "rubric_valid": True,
                    "rubric_epoch": str(epoch),
                    "evaluation_result": evaluation_result,
                }

            # Failed, try to revise
            feedback = self.generate_feedback(data, evaluation_result)
            revised_rubrics = await self.revise(
                data,
                rubrics,
                feedback,
            )

            if not revised_rubrics:
                break

            rubrics = revised_rubrics

        # Max epochs reached, still not converged
        return {
            "rubrics": rubrics,
            "rubric_valid": False,
            "rubric_epoch": str(self.max_epochs),
            "evaluation_result": evaluation_result,
        }

    async def generate(self, data: dict) -> List[str]:
        """Generate rubrics for a single data using ChatTemplate"""
        sample_content = self._format_data_context(data)
        task_description_section = self._format_task_description_section()

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def generate_rubrics():
            # Prepare parameters for generation
            if self.grader_mode == "pointwise":
                params = {
                    "language": self.language,
                    "sample_content": sample_content,
                    "generate_number": self.generate_number,
                    "min_score": self.min_score,
                    "max_score": self.max_score,
                    "task_description_section": task_description_section,
                }
            else:  # listwise
                params = {
                    "language": self.language,
                    "sample_content": sample_content,
                    "generate_number": self.generate_number,
                    "task_description_section": task_description_section,
                }

            # Use ChatTemplate with structured output
            chat_response = await self.model.achat(
                messages=self.generation_template.format(
                    **params,
                ),
                structured_model=RubricGenerationOutput,
            )

            # Get structured data from metadata
            if not chat_response.parsed:
                raise ValueError("No metadata in response")

            if "rubrics" not in chat_response.parsed:
                logger.error(
                    f"Missing 'rubrics' key in metadata. Available keys: {list(chat_response.parsed.keys())}",
                )
                raise KeyError(
                    f"'rubrics' key not found in response.parsed. Available keys: {list(chat_response.parsed.keys())}",
                )

            rubrics = chat_response.parsed["rubrics"]

            if not rubrics or len(rubrics) == 0:
                raise ValueError("No rubrics generated")

            return rubrics

        try:
            rubrics = await generate_rubrics()
            logger.debug(f"Generated {len(rubrics)} rubrics")
            return rubrics
        except Exception as e:
            logger.error(
                f"Failed to generate rubrics after {self.max_retries} attempts: {str(e)}",
            )
        return []

    async def aevaluate(
        self,
        data: dict,
        rubrics: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate data using rubrics

        Returns:
            - pointwise: {"scores": [int, ...]}
            - listwise: {"rank_values": [int, ...]}
        """
        if self.grader_mode == "pointwise":
            return await self._evaluate_pointwise(data, rubrics)
        else:  # listwise
            return await self._evaluate_listwise(data, rubrics)

    def validate(
        self,
        data: dict,
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """Validate evaluation result against ground truth"""
        if self.grader_mode == "pointwise":
            return self._validate_pointwise(data, evaluation_result)
        else:  # listwise
            return self._validate_listwise(data, evaluation_result)

    async def revise(
        self,
        data: dict,
        rubrics: List[str],
        feedback: str,
    ) -> List[str]:
        """Revise rubrics based on feedback using ChatTemplate"""
        sample_content = self._format_data_context(data)
        rubrics_text = self._format_rubrics_text(rubrics)
        task_description_section = self._format_task_description_section()

        @retry(stop=stop_after_attempt(self.max_retries), wait=wait_fixed(1.0))
        async def revise_rubrics():
            # Prepare parameters for revision
            params = {
                "language": self.language,
                "sample_content": sample_content,
                "rubrics": rubrics_text,
                "feedback": feedback,
                "generate_number": self.generate_number,
                "task_description_section": task_description_section,
            }

            chat_response = await self.model.achat(
                messages=self.revision_template.format(**params),
                structured_model=RubricGenerationOutput,
            )

            # Get structured data from metadata
            if not chat_response.parsed:
                raise ValueError("No metadata in response")

            if "rubrics" not in chat_response.parsed:
                logger.error(
                    f"Missing 'rubrics' key in metadata. Available keys: {list(chat_response.parsed.keys())}",
                )
                raise KeyError(
                    f"'rubrics' key not found in response.parsed. Available keys: {list(chat_response.parsed.keys())}",
                )

            revised_rubrics = chat_response.parsed["rubrics"]

            if not revised_rubrics or len(revised_rubrics) == 0:
                raise ValueError("No revised rubrics generated")

            return revised_rubrics

        try:
            revised_rubrics = await revise_rubrics()
            logger.debug(f"Revised {len(revised_rubrics)} rubrics")
            return revised_rubrics
        except Exception as e:
            logger.error(
                f"Failed to revise rubrics after {self.max_retries} attempts: {str(e)}",
            )
        return []

    def generate_feedback(
        self,
        data: dict,
        evaluation_result: Dict[str, Any],
    ) -> str:
        """Generate simple feedback for revision"""
        if self.grader_mode == "pointwise":
            return self._generate_pointwise_feedback(data, evaluation_result)
        else:  # listwise
            return self._generate_listwise_feedback(data, evaluation_result)

    # ========== Evaluation Methods ==========

    async def _evaluate_pointwise(
        self,
        data: dict,
        rubrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate in pointwise mode using ChatTemplate.

        Expects data format: {"query": "...", "response": "...", "label_score": 1}
        """
        rubrics_text = self._format_rubrics_text(rubrics)
        query = data.get("query", "")
        response = data.get("response", "")
        task_description_section = self._format_task_description_section()

        try:
            # Prepare parameters for pointwise evaluation
            params = {
                "language": self.language,
                "rubrics": rubrics_text,
                "query": query,
                "response": response,
                "min_score": self.min_score,
                "max_score": self.max_score,
                "task_description_section": task_description_section,
            }

            # Use ChatTemplate with structured output
            response_obj = await self.model.achat(
                messages=self.evaluation_template.format(**params),
                structured_model=GraderScore,
            )
            logger.debug(f"Pointwise evaluation response: {response_obj}")

            # Get structured data from parsed
            if response_obj.parsed and "score" in response_obj.parsed:
                score = response_obj.parsed["score"]
                score = max(self.min_score, min(self.max_score, score))
            else:
                score = self.min_score

            return {"scores": [score]}

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"scores": [self.min_score]}

    async def _evaluate_listwise(
        self,
        data: dict,
        rubrics: List[str],
    ) -> Dict[str, Any]:
        """Evaluate in listwise mode - model gives complete ranking at once.

        Expects data format: {"query": "...", "responses": [...], "label_rank": [1, 2, 3, ...]}
        Note: Smaller rank values indicate better quality (rank=1 is best).
        """
        rubrics_text = self._format_rubrics_text(rubrics)
        query = data.get("query", "")
        responses = data.get("responses", [])

        # Format responses for evaluation
        responses_text = "\n\n".join(
            [f"Response {i + 1}:\n{resp}" for i, resp in enumerate(responses)],
        )

        task_description_section = self._format_task_description_section()

        try:
            # Prepare parameters for listwise evaluation
            params = {
                "language": self.language,
                "rubrics": rubrics_text,
                "query": query,
                "responses": responses_text,
                "num_responses": len(responses),  # Explicitly tell LLM how many responses
                "task_description_section": task_description_section,
            }

            # Use ChatTemplate with structured output
            response_obj = await self.model.achat(
                messages=self.evaluation_template.format(**params),
                structured_model=GraderRank,
            )
            logger.debug(f"Listwise evaluation response: {response_obj}")

            # Get structured data from parsed
            if response_obj.parsed and "rank" in response_obj.parsed:
                rank_values = response_obj.parsed["rank"]

                # Validate rank values
                if len(rank_values) == len(responses):
                    if len(set(rank_values)) != len(rank_values):
                        logger.warning(
                            f"Duplicate rank values detected (ties not allowed): {rank_values}",
                        )
                    return {"rank_values": rank_values}
                else:
                    logger.warning(
                        f"Invalid rank values: got {len(rank_values)} ranks {rank_values} "
                        f"for {len(responses)} responses. Query: {query[:100]}...",
                    )

            return {"rank_values": []}

        except Exception as e:
            logger.error(f"Listwise evaluation failed: {e}")
            return {"rank_values": []}

    # ========== Validation Methods ==========

    def _validate_pointwise(
        self,
        data: dict,
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """Strict score match for pointwise evaluation.

        Expects data format: {"query": "...", "response": "...", "label_score": 1}
        """
        scores = evaluation_result.get("scores", [])
        if not scores or len(scores) != 1:
            return False

        # Check against expected score
        expected_score = data.get("label_score")
        if expected_score is not None:
            actual_score = scores[0]
            return actual_score == expected_score

        return False

    def _validate_listwise(
        self,
        data: dict,
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """Validate listwise results - supports rank value comparison by relative order.

        Expects data format: {"query": "...", "responses": [...], "label_rank": [1, 2, 3, ...]}
        Note: Smaller label_rank values indicate better quality (label_rank=1 is best).
        """
        rank_values = evaluation_result.get("rank_values", [])
        expected_ranks = data.get("label_rank", [])

        if not rank_values or not expected_ranks:
            return False

        if len(rank_values) != len(expected_ranks):
            return False

        # Compare relative order
        # Both expected and predicted ranks: smaller = better
        expected_order = self._get_relative_order(expected_ranks)
        logger.debug(f"Expected ranks: {expected_ranks}, order: {expected_order}")
        predicted_order = self._get_relative_order(rank_values)
        logger.debug(f"Predicted ranks: {rank_values}, order: {predicted_order}")

        # Must match relative order exactly
        return expected_order == predicted_order

    def _get_relative_order(self, values: List[float]) -> List[int]:
        """Convert rank values to relative order rankings (0-based indices sorted by rank asc).

        Note: In ranking, smaller values indicate better quality (label_rank=1 is best).
        Returns indices sorted from best to worst.
        """
        # Create (index, value) pairs
        indexed_values = list(enumerate(values))
        # Sort by label_rank value in ascending order (smaller label_rank = better quality)
        indexed_values.sort(key=lambda x: x[1])
        # Return the indices in order from best to worst
        return [idx for idx, _ in indexed_values]

    # ========== Feedback Generation ==========

    def _generate_pointwise_feedback(
        self,
        data: dict,
        evaluation_result: Dict[str, Any],
    ) -> str:
        """Generate simple pointwise feedback.

        Expects data format: {"query": "...", "response": "...", "label_score": 1}
        """
        expected_score = data.get("label_score")
        actual_scores = evaluation_result.get("scores", [])
        actual_score = actual_scores[0] if actual_scores else None

        return f"Expected score: {expected_score}\nActual score: {actual_score}"

    def _generate_listwise_feedback(
        self,
        data: dict,
        evaluation_result: Dict[str, Any],
    ) -> str:
        """Generate simple listwise feedback.

        Expects data format: {"query": "...", "responses": [...], "label_rank": [1, 2, 3, ...]}
        Note: Smaller rank values indicate better quality (rank=1 is best).
        """
        expected_ranks = data.get("label_rank", [])
        actual_ranks = evaluation_result.get("rank_values", [])

        # Format as simple comparison
        expected_ranks_str = str(expected_ranks)
        actual_ranks_str = str(actual_ranks)

        return f"Expected ranks: {expected_ranks_str}\nActual ranks: {actual_ranks_str}"

    def _format_data_context(self, data: dict) -> str:
        """Format data context for reference - language-neutral data formatting.

        Supports two data formats:
        - Pointwise: {"query": "...", "response": "...", "label_score": 1}
        - Listwise: {"query": "...", "responses": [...], "label_rank": [1, 2, 3, ...]}

        Note: In ranking, smaller rank values indicate better quality (rank=1 is best).
        """
        lines = []

        # Extract query
        query = data.get("query", "")

        # Check if this is pointwise or listwise data
        if "response" in data:
            # Pointwise mode: single response with score
            lines.append(f"Query: {query}")
            lines.append(f"Response: {data.get('response', '')}")

            score = data.get("label_score")
            if score is not None:
                lines.append(f"Expected score: {score}")

        elif "responses" in data:
            # Listwise mode: multiple responses with rankings
            lines.append(f"Query: {query}")
            lines.append("")

            responses = data.get("responses", [])
            ranks = data.get("label_rank", [])

            for i, response in enumerate(responses):
                lines.append(f"Response {i + 1}:")
                lines.append(f"{response}")

                if i < len(ranks):
                    # Smaller rank = better quality (rank=1 is best)
                    lines.append(f"Expected rank: {ranks[i]}")

                lines.append("")

        return "\n".join(lines)

    # ========== Utility Methods ==========

    def _format_rubrics_text(self, rubrics: List[str]) -> str:
        """Format rubrics list into numbered text"""
        return "\n".join(
            [f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics)],
        )

    def _format_task_description_section(self) -> str:
        """Format task description section for prompts.

        Returns:
            Formatted task description section, or empty string if not set.
        """
        if not self.task_description:
            return ""

        if self.language == LanguageEnum.ZH:
            return f"\n## 任务场景描述\n{self.task_description}\n"
        else:
            return f"\n## Task Description\n{self.task_description}\n"
