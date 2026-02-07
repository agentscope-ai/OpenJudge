# -*- coding: utf-8 -*-
"""Correctness detection grader for academic papers."""

import re
from typing import List

from cookbooks.paper_review.prompts.correctness import (
    CORRECTNESS_USER_PROMPT,
    get_correctness_system_prompt,
)
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_correctness_response(text: str) -> dict:
    """Parse XML-formatted correctness response."""
    score_match = re.search(r"<score>\s*(\d+)\s*</score>", text)
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    issues_match = re.search(r"<key_issues>(.*?)</key_issues>", text, re.DOTALL)

    score = int(score_match.group(1)) if score_match else 1
    reasoning = reasoning_match.group(1).strip() if reasoning_match else text

    key_issues = []
    if issues_match:
        issues_text = issues_match.group(1)
        key_issues = [line.strip().lstrip("- ") for line in issues_text.strip().split("\n") if line.strip()]

    return {"score": score, "reason": reasoning, "key_issues": key_issues}


def build_correctness_messages(pdf_data: str) -> List[dict]:
    """Build messages with PDF data properly injected."""
    return [
        {"role": "system", "content": get_correctness_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CORRECTNESS_USER_PROMPT},
                {"type": "file", "file": {"file_data": pdf_data}},
            ],
        },
    ]


class CorrectnessGrader(LLMGrader):
    """Grader for detecting objective errors in academic papers.

    Score range: 1-3
        1 = No objective errors detected
        2 = Minor errors present
        3 = Major errors present
    """

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="paper_correctness",
            mode=GraderMode.POINTWISE,
            description="Detect objective errors in academic papers",
            model=model,
            template="",  # Placeholder, not used
        )

    async def aevaluate(self, pdf_data: str) -> GraderScore:
        """Evaluate paper correctness.

        Args:
            pdf_data: Base64 encoded PDF data URL

        Returns:
            GraderScore with score 1-3 and identified issues
        """
        try:
            messages = build_correctness_messages(pdf_data)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_correctness_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["reason"],
                metadata={"key_issues": parsed["key_issues"]},
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
