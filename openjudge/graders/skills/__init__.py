# -*- coding: utf-8 -*-
"""
Skill Graders

This module contains graders for evaluating AI Agent Skill packages:
- Safety evaluation: detects dangerous operations, overly broad permissions, and missing safeguards
- Relevance evaluation: measures how well a skill's capabilities address a task description
- Completeness evaluation: measures whether a skill provides sufficient detail to accomplish a task
- Structure evaluation: assesses structural design quality across anti-pattern quality,
  specification compliance, progressive disclosure, and freedom calibration
- Comprehensive evaluation: holistic multi-dimensional assessment combining all four dimensions
- Comprehensive pairwise evaluation: head-to-head comparison of two skill candidates
"""

from openjudge.graders.skills.completeness import SkillCompletenessGrader
from openjudge.graders.skills.comprehensive import SkillComprehensiveGrader
from openjudge.graders.skills.comprehensive_pairwise import SkillComprehensivePairwiseGrader
from openjudge.graders.skills.relevance import SkillRelevanceGrader
from openjudge.graders.skills.safety import SkillSafetyGrader
from openjudge.graders.skills.structure import SkillStructureGrader

__all__ = [
    "SkillSafetyGrader",
    "SkillRelevanceGrader",
    "SkillCompletenessGrader",
    "SkillStructureGrader",
    "SkillComprehensiveGrader",
    "SkillComprehensivePairwiseGrader",
]
