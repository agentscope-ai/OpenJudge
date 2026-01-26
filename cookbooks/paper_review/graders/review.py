# -*- coding: utf-8 -*-
"""Paper review grader for academic papers."""

import re
from typing import List

from cookbooks.paper_review.prompts.review import (
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT,
)
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_review_response(text: str) -> dict:
    """Parse XML-formatted review response."""
    review_match = re.search(r"<review>(.*?)</review>", text, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", text)

    review = review_match.group(1).strip() if review_match else text
    score = int(answer_match.group(1)) if answer_match else 3

    return {"score": score, "review": review}


def build_review_messages(pdf_data: str) -> List[dict]:
    """Build messages with PDF data properly injected."""
    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
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
            template=REVIEW_SYSTEM_PROMPT,  # Placeholder, not used
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

            if hasattr(response, "__aiter__"):
                async for chunk in response:
                    response = chunk

            content = response.content if hasattr(response, "content") else str(response)
            parsed = parse_review_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["review"],
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
