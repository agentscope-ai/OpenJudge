# -*- coding: utf-8 -*-
"""
Skill Graders

This module contains graders for evaluating AI Agent Skill packages:
- Threat analysis: LLM-based semantic threat scanner with AITech taxonomy (prompt injection,
  data exfiltration, command injection, obfuscation, tool exploitation, etc.)
- Alignment evaluation: detects mismatches between SKILL.md declared intent and actual script behavior
- Relevance evaluation: measures how well a skill's capabilities address a task description
- Completeness evaluation: measures whether a skill provides sufficient detail to accomplish a task
- Structure evaluation: assesses structural design quality across anti-pattern quality,
  specification compliance, progressive disclosure, and freedom calibration

For multi-dimensional skill evaluation using all graders combined, see
``cookbooks/skills_evaluation/runner.py`` (SkillsGradingRunner).
"""

from openjudge.graders.skills.completeness import SkillCompletenessGrader
from openjudge.graders.skills.declaration_alignment import (
    SkillDeclarationAlignmentGrader,
)
from openjudge.graders.skills.design import SkillDesignGrader
from openjudge.graders.skills.relevance import SkillRelevanceGrader
from openjudge.graders.skills.threat_analysis import SkillThreatAnalysisGrader

__all__ = [
    "SkillThreatAnalysisGrader",
    "SkillDeclarationAlignmentGrader",
    "SkillRelevanceGrader",
    "SkillCompletenessGrader",
    "SkillDesignGrader",
]
