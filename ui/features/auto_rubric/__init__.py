# -*- coding: utf-8 -*-
"""Auto Rubric feature module for OpenJudge Studio.

This feature provides automatic rubric generation from task descriptions
or labeled data, eliminating manual rubric design.

Phase 1 implements:
- Simple Rubric mode (zero-shot from task description)
- Basic result display
- Export functionality (Python/YAML/JSON)
"""

from features.auto_rubric.feature import AutoRubricFeature

__all__ = ["AutoRubricFeature"]
