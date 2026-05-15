# -*- coding: utf-8 -*-
"""
Evaluation Harness for OpenJudge Graders.

This module provides a reusable framework for running grader evaluations
against benchmark datasets, computing pairwise/pointwise accuracy metrics,
and formatting results.
"""

from openjudge.evaluation_harness.base_harness import EvalResult, GraderHarness

__all__ = [
    "GraderHarness",
    "EvalResult",
]
