# -*- coding: utf-8 -*-
"""Criticality verification grader for academic papers."""

import json
import re
from typing import List

from cookbooks.paper_review.prompts.criticality import get_criticality_system_prompt
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_criticality_response(text: str) -> dict:
    """Parse JSON-formatted criticality response."""
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return {
                "score": data.get("score", 1),
                "reasoning": data.get("reasoning", ""),
                "issues": data.get("issues", {"major": [], "minor": [], "false_positives": []}),
            }
        except json.JSONDecodeError:
            pass

    return {
        "score": 1,
        "reasoning": text,
        "issues": {"major": [], "minor": [], "false_positives": []},
    }


def build_criticality_messages(pdf_data: str, findings: str) -> List[dict]:
    """Build messages with PDF data and findings properly injected."""
    user_prompt = f"""The correctness detector identified the following issues in this paper:

{findings}

Carefully verify each issue against the actual paper content. Classify each issue by severity and assign an overall score from 1 to 3."""

    return [
        {"role": "system", "content": get_criticality_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "file", "file": {"file_data": pdf_data}},
            ],
        },
    ]


class CriticalityGrader(LLMGrader):
    """Grader for verifying criticality of identified issues.

    Score range: 1-3
        1 = No genuine errors (all false positives)
        2 = Minor errors present but main contributions valid
        3 = Major error that compromises paper validity
    """

    def __init__(self, model: BaseChatModel | dict):
        super().__init__(
            name="criticality_verifier",
            mode=GraderMode.POINTWISE,
            description="Verify criticality of identified issues",
            model=model,
            template="",  # Placeholder, not used
        )

    async def aevaluate(self, pdf_data: str, findings: str) -> GraderScore:
        """Verify criticality of correctness findings.

        Args:
            pdf_data: Base64 encoded PDF data URL
            findings: Formatted findings from correctness detector

        Returns:
            GraderScore with classified issues
        """
        try:
            messages = build_criticality_messages(pdf_data, findings)
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_criticality_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["score"],
                reason=parsed["reasoning"],
                metadata={"issues": parsed["issues"]},
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
