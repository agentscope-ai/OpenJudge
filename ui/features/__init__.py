# -*- coding: utf-8 -*-
"""Feature modules for OpenJudge Studio.

Each feature module provides a self-contained UI for a specific functionality.
Features are registered with the FeatureRegistry and can be accessed via the
navigation system.
"""

from features.auto_arena import AutoArenaFeature
from features.auto_rubric import AutoRubricFeature
from features.grader import GraderFeature
from features.paper_review import PaperReviewFeature

__all__ = ["GraderFeature", "AutoArenaFeature", "PaperReviewFeature", "AutoRubricFeature"]
