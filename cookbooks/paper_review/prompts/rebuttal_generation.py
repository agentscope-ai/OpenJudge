# -*- coding: utf-8 -*-
"""Prompts for rebuttal generation."""

from datetime import datetime
from typing import Optional

from cookbooks.paper_review.disciplines.base import DisciplineConfig


def get_rebuttal_generation_system_prompt(
    date: datetime | None = None,
    discipline: Optional[DisciplineConfig] = None,
    venue: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Get system prompt for rebuttal generation.

    Args:
        date: Date to use (defaults to today).
        discipline: Discipline-specific configuration.
        venue: Target conference/journal name.
        language: Output language ("en" default, "zh" for Chinese).
    """
    current_date = (date or datetime.now()).strftime("%Y-%m-%d")

    if discipline:
        identity = (
            f"You are an experienced researcher in {discipline.name} who excels at "
            f"writing persuasive and professional academic rebuttals."
        )
    else:
        identity = (
            "You are an experienced researcher who excels at writing persuasive " "and professional academic rebuttals."
        )

    venue_block = ""
    if venue:
        venue_block = (
            f"\n**Target Venue: {venue}**\n"
            f"Tailor the rebuttal tone and conventions to {venue}'s rebuttal guidelines."
        )

    if language == "zh":
        language_block = (
            "\n**Output Language: Chinese (Simplified)**\n"
            "You MUST write the entire rebuttal and all analysis in Simplified Chinese (简体中文)."
        )
    else:
        language_block = ""

    return f"""{identity}

**Current Date: {current_date}**
{venue_block}{language_block}

Your task is to help the author draft a structured rebuttal that addresses each reviewer concern.

CRITICAL RULES:
1. For concerns that CAN be addressed by clarification, explanation, or pointing to existing content in the paper, provide a concrete draft response.
2. For concerns that REQUIRE new work the author must do themselves (additional experiments, new baselines, extra ablation studies, new theoretical proofs, data collection, etc.), you MUST insert a placeholder tag so the author knows to fill it in. Use this exact format:
   [TODO: <brief description of what the author needs to provide>]
   Examples:
   - [TODO: Insert results of the requested ablation study on component X]
   - [TODO: Add comparison table with baseline Y on dataset Z]
   - [TODO: Provide runtime/memory measurements]
   - [TODO: Include the new proof or formal argument for Theorem 2]
3. Be honest: do NOT fabricate experimental results, numbers, or proofs. If you cannot determine the answer from the paper, use a [TODO] placeholder.
4. Maintain a respectful, professional, and constructive tone throughout.
5. Structure the rebuttal as a point-by-point response to each concern.

For each reviewer concern, output a JSON object with:
- "concern": the reviewer's original point (verbatim or faithfully summarized)
- "severity": "major" or "minor"
- "response_type": "clarification" (answerable from paper) or "action_required" (needs new work)
- "draft_response": your drafted response text (with [TODO] placeholders where needed)

Return your full output as JSON:
{{
  "concerns": [
    {{
      "concern": "...",
      "severity": "major",
      "response_type": "clarification" or "action_required",
      "draft_response": "..."
    }}
  ],
  "rebuttal_text": "The complete rebuttal letter text (with [TODO] placeholders)",
  "general_suggestions": ["Optional high-level suggestions for strengthening the revision"]
}}"""


REBUTTAL_GENERATION_USER_PROMPT = """Below are the reviewer comments for this paper:

{review_text}

Please read the paper carefully and draft a point-by-point rebuttal addressing every concern. Use [TODO: ...] placeholders for anything that requires new experiments, proofs, or data that you cannot determine from the paper alone."""
