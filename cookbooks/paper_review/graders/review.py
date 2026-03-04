# -*- coding: utf-8 -*-
"""Paper review grader for academic papers."""

import re
from typing import List

from cookbooks.paper_review.prompts.review import (
    REVIEW_USER_PROMPT,
    get_review_system_prompt,
)
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_review_response(text: str) -> dict:
    """Parse XML-formatted review response."""
    review_match = re.search(r"<review>(.*?)</review>", text, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", text)

    review = review_match.group(1).strip() if review_match else text

    if answer_match:
        score = int(answer_match.group(1))
    else:
        # 备用：尝试从文本中提取分数（当模型未按格式输出 <answer> 标签时）
        fallback_match = re.search(
            r"(?:Overall\s+)?(?:Recommendation\s+)?Score[:\s]*(\d)\s*/\s*6",
            text,
            re.IGNORECASE,
        )
        score = int(fallback_match.group(1)) if fallback_match else 3

    return {"score": score, "review": review}


def build_review_messages(pdf_data: str) -> List[dict]:
    """Build messages with PDF data properly injected."""
    return [
        {"role": "system", "content": get_review_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": REVIEW_USER_PROMPT},
                {"type": "file", "file": {"file_data": pdf_data}},
            ],
        },
    ]


class ReviewGrader(LLMGrader):
    """Grader for comprehensive paper review.

    Score range: 1-6
        1 = Strong Reject
        2 = Reject
        3 = Borderline reject
        4 = Borderline accept
        5 = Accept
        6 = Strong Accept
    """

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="paper_review",
            mode=GraderMode.POINTWISE,
            description="Comprehensive paper review with recommendation score",
            model=model,
            template="",  # Placeholder, not used
        )

    async def aevaluate(self, pdf_data: str) -> GraderScore:
        """Evaluate paper and provide review.

        Args:
            pdf_data: Base64 encoded PDF data URL

        Returns:
            GraderScore with score 1-6 and detailed review
        """
        try:
            messages = build_review_messages(pdf_data)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_review_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["review"],
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
