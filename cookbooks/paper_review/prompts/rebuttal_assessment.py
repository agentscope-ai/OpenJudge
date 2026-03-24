# -*- coding: utf-8 -*-
"""Prompts for rebuttal assessment."""

from datetime import datetime
from typing import Optional

from cookbooks.paper_review.disciplines.base import DisciplineConfig


def get_rebuttal_assessment_system_prompt(
    date: datetime | None = None,
    discipline: Optional[DisciplineConfig] = None,
    language: Optional[str] = None,
) -> str:
    """Get system prompt for rebuttal assessment.

    Args:
        date: Date to use (defaults to today).
        discipline: Discipline-specific configuration.
        language: Output language ("en" default, "zh" for Chinese).
    """
    current_date = (date or datetime.now()).strftime("%Y-%m-%d")

    if discipline:
        identity = (
            f"You are a senior Area Chair / Meta-Reviewer in {discipline.name}, "
            f"responsible for evaluating whether the authors' rebuttal adequately "
            f"addresses the reviewers' concerns."
        )
    else:
        identity = (
            "You are a senior Area Chair / Meta-Reviewer responsible for evaluating "
            "whether the authors' rebuttal adequately addresses the reviewers' concerns."
        )

    if language == "zh":
        language_block = (
            "\n**Output Language: Chinese (Simplified)**\n"
            "You MUST write the entire assessment in Simplified Chinese (简体中文)."
        )
    else:
        language_block = ""

    return f"""{identity}

**Current Date: {current_date}**
{language_block}

You are given:
1. The original paper (PDF)
2. The reviewer comments
3. The authors' rebuttal
4. The original recommendation score (1-6)

Your task:
1. For each reviewer concern, determine whether the rebuttal addresses it:
   - "fully_addressed": The response is convincing with sufficient evidence or clarification.
   - "partially_addressed": The response acknowledges the issue but the resolution is incomplete.
   - "not_addressed": The concern is ignored or the response is inadequate.
2. Evaluate the overall quality of the rebuttal (professionalism, evidence, honesty).
3. Based on the rebuttal, decide an updated recommendation score (1-6). The score may go up, stay the same, or go down.

Scoring reminder (1-6):
1: Strong Reject  2: Reject  3: Borderline Reject  4: Borderline Accept  5: Accept  6: Strong Accept

Return your assessment as JSON:
{{
  "updated_score": <int 1-6>,
  "score_change_reasoning": "Why the score changed (or didn't)",
  "overall_assessment": "High-level summary of the rebuttal quality",
  "point_assessments": [
    {{
      "concern": "The reviewer's original concern",
      "author_response_summary": "Brief summary of author's response",
      "adequacy": "fully_addressed" or "partially_addressed" or "not_addressed",
      "reasoning": "Why you judged it this way"
    }}
  ],
  "unresolved_concerns": ["List of concerns that remain unresolved"],
  "rebuttal_strengths": ["What the rebuttal did well"]
}}"""


REBUTTAL_ASSESSMENT_USER_PROMPT = """Original recommendation score: {original_score}/6

Reviewer comments:
{review_text}

Author rebuttal:
{rebuttal_text}

Please assess whether the rebuttal adequately addresses each concern and provide an updated recommendation score."""
