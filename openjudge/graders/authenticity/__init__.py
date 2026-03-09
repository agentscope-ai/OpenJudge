# -*- coding: utf-8 -*-
"""
Authenticity Graders

Graders that detect whether an API endpoint is backed by a genuine
Anthropic Claude model, replicating the logic of the claude-verify project.
"""

from openjudge.graders.authenticity.checks import (
    CheckInput,
    CheckResult,
    evaluate_checks,
    extract_signature_from_response,
    get_verdict,
)
from openjudge.graders.authenticity.claude_authenticity_grader import (
    ClaudeAuthenticityGrader,
)

__all__ = [
    "ClaudeAuthenticityGrader",
    "CheckInput",
    "CheckResult",
    "evaluate_checks",
    "extract_signature_from_response",
    "get_verdict",
]
