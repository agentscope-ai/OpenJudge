# -*- coding: utf-8 -*-
"""Rebuttal assessment grader for academic papers."""

import json
import re
from typing import List, Optional

from cookbooks.paper_review.disciplines.base import DisciplineConfig
from cookbooks.paper_review.prompts.rebuttal_assessment import (
    REBUTTAL_ASSESSMENT_USER_PROMPT,
    get_rebuttal_assessment_system_prompt,
)
from cookbooks.paper_review.utils import extract_response_content
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel


def parse_rebuttal_assessment_response(text: str) -> dict:
    """Parse JSON-formatted rebuttal assessment response."""
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            point_assessments = []
            for p in data.get("point_assessments", []):
                point_assessments.append(
                    {
                        "concern": p.get("concern", ""),
                        "author_response_summary": p.get("author_response_summary", ""),
                        "adequacy": p.get("adequacy", "not_addressed"),
                        "reasoning": p.get("reasoning", ""),
                    }
                )
            return {
                "updated_score": int(data.get("updated_score", 3)),
                "score_change_reasoning": data.get("score_change_reasoning", ""),
                "overall_assessment": data.get("overall_assessment", ""),
                "point_assessments": point_assessments,
                "unresolved_concerns": data.get("unresolved_concerns", []),
                "rebuttal_strengths": data.get("rebuttal_strengths", []),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {
        "updated_score": 3,
        "score_change_reasoning": "",
        "overall_assessment": text,
        "point_assessments": [],
        "unresolved_concerns": [],
        "rebuttal_strengths": [],
    }


def build_rebuttal_assessment_messages(
    pdf_data: str,
    review_text: str,
    rebuttal_text: str,
    original_score: int,
    discipline: Optional[DisciplineConfig] = None,
    language: Optional[str] = None,
) -> List[dict]:
    """Build messages for rebuttal assessment."""
    user_prompt = REBUTTAL_ASSESSMENT_USER_PROMPT.format(
        original_score=original_score,
        review_text=review_text,
        rebuttal_text=rebuttal_text,
    )
    return [
        {
            "role": "system",
            "content": get_rebuttal_assessment_system_prompt(
                discipline=discipline,
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


class RebuttalAssessmentGrader(LLMGrader):
    """Grader that assesses whether a rebuttal adequately addresses reviewer concerns.

    Score range: 1-6 (updated recommendation after reading the rebuttal)
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        discipline: Optional[DisciplineConfig] = None,
        language: Optional[str] = None,
    ):
        super().__init__(
            name="rebuttal_assessment",
            mode=GraderMode.POINTWISE,
            description="Assess rebuttal adequacy and update recommendation",
            model=model,
            template="",
        )
        self.discipline = discipline
        self.language = language

    async def aevaluate(
        self,
        pdf_data: str,
        review_text: str,
        rebuttal_text: str,
        original_score: int,
    ) -> GraderScore:
        """Assess a rebuttal.

        Args:
            pdf_data: Base64 encoded PDF data URL
            review_text: The original reviewer comments
            rebuttal_text: The author's rebuttal text
            original_score: The original recommendation score (1-6)

        Returns:
            GraderScore with updated score and assessment details in metadata
        """
        try:
            messages = build_rebuttal_assessment_messages(
                pdf_data,
                review_text,
                rebuttal_text,
                original_score,
                discipline=self.discipline,
                language=self.language,
            )
            response = await self.model.achat(messages=messages)
            content = await extract_response_content(response)
            parsed = parse_rebuttal_assessment_response(content)

            return GraderScore(
                name=self.name,
                score=parsed["updated_score"],
                reason=parsed["overall_assessment"],
                metadata={
                    "original_score": original_score,
                    "score_change_reasoning": parsed["score_change_reasoning"],
                    "point_assessments": parsed["point_assessments"],
                    "unresolved_concerns": parsed["unresolved_concerns"],
                    "rebuttal_strengths": parsed["rebuttal_strengths"],
                },
            )
        except Exception as e:
            return GraderError(name=self.name, error=str(e))
