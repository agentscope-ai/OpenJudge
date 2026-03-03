# -*- coding: utf-8 -*-
"""Prompts for paper review."""

from datetime import datetime
from typing import Optional

from cookbooks.paper_review.disciplines.base import DisciplineConfig


def get_review_system_prompt(
    date: datetime | None = None,
    discipline: Optional[DisciplineConfig] = None,
    venue: Optional[str] = None,
) -> str:
    """Get the review system prompt with current date, optional discipline and venue.

    Args:
        date: Date to use (defaults to today).
        discipline: Discipline-specific configuration. If None, uses the original
                    general CS/ML-oriented prompt.
        venue: Specific conference or journal name (e.g. "NeurIPS 2025", "The Lancet").
               When provided, the reviewer is instructed to apply that venue's standards
               on top of the discipline criteria. Users can pass any custom venue name.
    """
    current_date = (date or datetime.now()).strftime("%Y-%m-%d")

    # ── Reviewer identity ──────────────────────────────────────────────────────
    if discipline:
        discipline_label = discipline.name
        reviewer_context = discipline.reviewer_context or (
            f"You specialize in {discipline_label}."
        )
        identity_block = (
            f"You are an expert academic paper reviewer specializing in {discipline_label}. "
            f"You are the best reviewer in the world.\n"
            f"{reviewer_context}"
        )
    else:
        identity_block = "You are an academic paper reviewer. You are the best reviewer in the world."

    # ── Venue context ──────────────────────────────────────────────────────────
    if venue:
        venue_block = (
            f"\n**Target Venue: {venue}**\n"
            f"You are reviewing this paper specifically for submission to **{venue}**. "
            f"Apply the standards, scope, and expectations of {venue} when evaluating this work. "
            f"Consider whether the paper fits the venue's audience and contribution bar."
        )
    elif discipline and discipline.venues:
        venue_list = discipline.format_venues()
        venue_block = (
            f"\nYou typically review for top venues in {discipline.name}, such as: {venue_list}."
        )
    else:
        venue_block = (
            "\nYou typically review for top venues such as "
            "NeurIPS, ICLR, ICML, Nature, Science."
        )

    # ── Evaluation dimensions ──────────────────────────────────────────────────
    if discipline and discipline.evaluation_dimensions:
        dimensions_block = discipline.format_evaluation_dimensions()
    else:
        dimensions_block = """Quality: Is the submission technically sound? Are claims well supported by theoretical analysis or experimental results? Are the methods appropriate? Is this a complete piece of work? Are the authors honest about strengths and weaknesses?

Clarity: Is the submission clearly written and well organized? Does it adequately inform the reader with enough information for reproduction?

Significance: Are the results impactful for the community? Will others likely use or build on these ideas? Does it address a difficult task better than previous work?

Originality: Does the work provide new insights or deepen understanding? Is it clear how this differs from previous contributions?

Reproducibility: Does the paper provide sufficient detail for an expert to reproduce the results? Are implementation details, datasets, and experimental setup clearly described?

Ethics and Limitations: Have the authors adequately addressed limitations and potential negative societal impact?

Citations and Related Work: Are relevant prior works properly cited and compared?"""

    # ── Scoring notes ─────────────────────────────────────────────────────────
    base_scoring = """Scoring (1-6):
1: Strong Reject - Well-known results, technical flaws, or unaddressed ethical considerations
2: Reject - Technical flaws, weak evaluation, inadequate reproducibility
3: Borderline reject - Technically solid but reasons to reject outweigh reasons to accept
4: Borderline accept - Technically solid where reasons to accept outweigh reasons to reject
5: Accept - Technically solid with high impact, good evaluation
6: Strong Accept - Technically flawless with groundbreaking impact"""

    if discipline and discipline.scoring_notes:
        scoring_block = base_scoring + f"\n\n{discipline.scoring_notes}"
    else:
        scoring_block = base_scoring

    return f"""{identity_block}

**Current Date: {current_date}**
Note: References to papers from 2024, 2025, or 2026 are valid and should NOT be flagged as "future" papers.
{venue_block}

You keep incredibly high standards and only the best papers get accepted.

When evaluating the paper, consider these key dimensions:

{dimensions_block}

In general:
* If a paper is bad and you are unsure, reject it.
* If a paper is good and you are unsure, accept it.

{scoring_block}

Your answer MUST follow this format:

<review>
Your detailed review here.
</review>

<answer>X</answer>

Where X is your numerical score from 1 to 6."""


REVIEW_USER_PROMPT = "Please review this paper and provide an overall recommendation score from 1 to 6."
