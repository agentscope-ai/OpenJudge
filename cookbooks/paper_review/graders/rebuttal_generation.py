# -*- coding: utf-8 -*-
"""Rebuttal generation grader for academic papers."""

import json
import re
from typing import List, Optional

from cookbooks.paper_review.disciplines.base import DisciplineConfig
from cookbooks.paper_review.prompts.rebuttal_generation import (
    REBUTTAL_GENERATION_USER_PROMPT,
    get_rebuttal_generation_system_prompt,
)
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_rebuttal_generation_response(text: str) -> dict:
    """Parse JSON-formatted rebuttal generation response."""
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            concerns = []
            for c in data.get("concerns", []):
                concerns.append(
                    {
                        "concern": c.get("concern", ""),
                        "severity": c.get("severity", "minor"),
                        "response_type": c.get("response_type", "clarification"),
                        "draft_response": c.get("draft_response", ""),
                    }
                )
            return {
                "concerns": concerns,
                "rebuttal_text": data.get("rebuttal_text", ""),
                "general_suggestions": data.get("general_suggestions", []),
            }
        except json.JSONDecodeError:
            pass

    return {
        "concerns": [],
        "rebuttal_text": text,
        "general_suggestions": [],
    }


def build_rebuttal_generation_messages(
    pdf_data: str,
    review_text: str,
    discipline: Optional[DisciplineConfig] = None,
    venue: Optional[str] = None,
    language: Optional[str] = None,
) -> List[dict]:
    """Build messages for rebuttal generation."""
    user_prompt = REBUTTAL_GENERATION_USER_PROMPT.format(review_text=review_text)
    return [
        {
            "role": "system",
            "content": get_rebuttal_generation_system_prompt(
                discipline=discipline,
                venue=venue,
                language=language,
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "file", "file": {"file_data": pdf_data}},
            ],
        },
    ]


class RebuttalGenerationGrader(LLMGrader):
    """Grader that generates a structured rebuttal draft for the author.

    Produces a point-by-point rebuttal with [TODO] placeholders for items
    that require new experiments, proofs, or other work the author must do.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        discipline: Optional[DisciplineConfig] = None,
        venue: Optional[str] = None,
        language: Optional[str] = None,
    ):
        super().__init__(
            name="rebuttal_generation",
            mode=GraderMode.POINTWISE,
            description="Generate structured rebuttal draft with TODO placeholders",
            model=model,
            template="",
        )
        self.discipline = discipline
        self.venue = venue
        self.language = language

    async def aevaluate(self, pdf_data: str, review_text: str) -> GraderScore:
        """Generate a rebuttal draft.

        Args:
            pdf_data: Base64 encoded PDF data URL
            review_text: The reviewer comments to respond to

        Returns:
            GraderScore with rebuttal text in reason, structured data in metadata
        """
        try:
            messages = build_rebuttal_generation_messages(
                pdf_data,
                review_text,
                discipline=self.discipline,
                venue=self.venue,
                language=self.language,
            )
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_rebuttal_generation_response(content)

            return GraderScore(
                name=self.name,
                score=0,
                reason=parsed["rebuttal_text"],
                metadata={
                    "concerns": parsed["concerns"],
                    "general_suggestions": parsed["general_suggestions"],
                },
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
