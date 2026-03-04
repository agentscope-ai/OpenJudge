# -*- coding: utf-8 -*-
"""Jailbreaking detection grader for academic papers."""

import re
from typing import List

from cookbooks.paper_review.prompts.jailbreaking import (
    JAILBREAKING_SYSTEM_PROMPT,
    JAILBREAKING_USER_PROMPT,
)
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_jailbreaking_response(text: str) -> dict:
    """Parse XML-formatted jailbreaking detection response."""
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    answer_match = re.search(r"<answer>\s*(abuse|ok)\s*</answer>", text, re.IGNORECASE)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else text
    answer = answer_match.group(1).lower() if answer_match else "ok"

    return {"is_abuse": answer == "abuse", "reasoning": reasoning}


def build_jailbreaking_messages(pdf_data: str) -> List[dict]:
    """Build messages with PDF data properly injected."""
    return [
        {"role": "system", "content": JAILBREAKING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": JAILBREAKING_USER_PROMPT},
                {"type": "file", "file": {"file_data": pdf_data}},
            ],
        },
    ]


class JailbreakingGrader(LLMGrader):
    """Grader for detecting jailbreaking attempts in papers.

    Score range: 0-1
        0 = OK (no jailbreaking detected)
        1 = Abuse detected
    """

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="jailbreaking_detector",
            mode=GraderMode.POINTWISE,
            description="Detect jailbreaking attempts in papers",
            model=model,
            template=JAILBREAKING_SYSTEM_PROMPT,  # Placeholder, not used
        )

    async def aevaluate(self, pdf_data: str) -> GraderScore:
        """Detect jailbreaking attempts.

        Args:
            pdf_data: Base64 encoded PDF data URL

        Returns:
            GraderScore with detection result
        """
        try:
            messages = build_jailbreaking_messages(pdf_data)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_jailbreaking_response(content)

            return GraderScore(
                name=self.name,
                score=1 if parsed["is_abuse"] else 0,
                reason=parsed["reasoning"],
                metadata={"is_abuse": parsed["is_abuse"]},
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
