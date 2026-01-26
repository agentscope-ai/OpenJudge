# -*- coding: utf-8 -*-
"""Prompts for paper review graders."""

from cookbooks.paper_review.prompts.correctness import (
    CORRECTNESS_SYSTEM_PROMPT,
    CORRECTNESS_USER_PROMPT,
)
from cookbooks.paper_review.prompts.criticality import (
    CRITICALITY_SYSTEM_PROMPT,
    CRITICALITY_USER_PROMPT,
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
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT,
)

__all__ = [
    "CORRECTNESS_SYSTEM_PROMPT",
    "CORRECTNESS_USER_PROMPT",
    "REVIEW_SYSTEM_PROMPT",
    "REVIEW_USER_PROMPT",
    "CRITICALITY_SYSTEM_PROMPT",
    "CRITICALITY_USER_PROMPT",
    "FORMAT_SYSTEM_PROMPT",
    "FORMAT_USER_PROMPT",
    "JAILBREAKING_SYSTEM_PROMPT",
    "JAILBREAKING_USER_PROMPT",
]
