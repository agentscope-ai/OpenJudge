# -*- coding: utf-8 -*-
"""Agent names enumeration for OpenJudge agent evaluation framework."""

from enum import Enum


class AgentName(str, Enum):
    """Enumeration of supported agent names for evaluation."""

    # Built-in agents
    OPENCLAW = "openclaw"
    COPAW = "copaw"

    # Placeholder for future agents
    CLAUDE_CODE = "claude-code"
    AIDER = "aider"
    GPT_ENGINEER = "gpt-engineer"

    @classmethod
    def values(cls) -> set[str]:
        """Return set of all agent name values."""
        return {member.value for member in cls}