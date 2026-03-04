# -*- coding: utf-8 -*-
"""Format compliance grader for academic papers."""

import re
from typing import List

from cookbooks.paper_review.prompts.format import (
    FORMAT_SYSTEM_PROMPT,
    FORMAT_USER_PROMPT,
)
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_format_response(text: str) -> dict:
    """Parse XML-formatted format check response."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    result_match = re.search(r"<result>\s*(\d+)\s*</result>", text)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else text
    result = int(result_match.group(1)) if result_match else 0

    violations = []
    violations_match = re.search(r"Violations:\s*(.+?)(?:\n|$)", reasoning)
    if violations_match:
        violations = [v.strip() for v in violations_match.group(1).split(",")]

    return {"score": result, "reasoning": reasoning, "violations": violations}


def build_format_messages(pdf_data: str) -> List[dict]:
    """Build messages with PDF data properly injected."""
    return [
        {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FORMAT_USER_PROMPT},
                {"type": "file", "file": {"file_data": pdf_data}},
            ],
        },
    ]


class FormatGrader(LLMGrader):
    """Grader for checking format compliance.

    Score range: 0-1
        0 = Format is OK (compliant)
        1 = Format violations found
    """

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="format_compliance",
            mode=GraderMode.POINTWISE,
            description="Check paper format compliance",
            model=model,
            template=FORMAT_SYSTEM_PROMPT,  # Placeholder, not used
        )

    async def aevaluate(self, pdf_data: str) -> GraderScore:
        """Check format compliance.

        Args:
            pdf_data: Base64 encoded PDF data URL

        Returns:
            GraderScore with compliance status
        """
        try:
            messages = build_format_messages(pdf_data)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_format_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["reasoning"],
                metadata={"violations": parsed["violations"]},
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
