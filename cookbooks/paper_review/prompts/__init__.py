# -*- coding: utf-8 -*-
"""Prompts for paper review graders."""

from cookbooks.paper_review.prompts.correctness import (
    CORRECTNESS_USER_PROMPT,
    get_correctness_system_prompt,
)
from cookbooks.paper_review.prompts.criticality import (
    CRITICALITY_USER_PROMPT,
    get_criticality_system_prompt,
)
from cookbooks.paper_review.prompts.format import (
    FORMAT_SYSTEM_PROMPT,
    FORMAT_USER_PROMPT,
)
from cookbooks.paper_review.prompts.jailbreaking import (
    JAILBREAKING_SYSTEM_PROMPT,
    JAILBREAKING_USER_PROMPT,
)
from cookbooks.paper_review.prompts.review import (
    REVIEW_USER_PROMPT,
    get_review_system_prompt,
)

__all__ = [
    "get_correctness_system_prompt",
    "CORRECTNESS_USER_PROMPT",
    "get_review_system_prompt",
    "REVIEW_USER_PROMPT",
    "get_criticality_system_prompt",
    "CRITICALITY_USER_PROMPT",
    "FORMAT_SYSTEM_PROMPT",
    "FORMAT_USER_PROMPT",
    "JAILBREAKING_SYSTEM_PROMPT",
    "JAILBREAKING_USER_PROMPT",
]
