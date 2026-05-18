# -*- coding: utf-8 -*-
"""
Grader Benchmark for OpenJudge Graders.

This module provides a reusable framework for running grader evaluations
against benchmark datasets, computing pairwise/pointwise accuracy metrics,
and formatting results.
"""

from openjudge.grader_benchmark.agent_grader_registry import (
    AGENT_GRADER_REGISTRY,
    build_benchmark,
    get_all_categories,
    get_graders_by_category,
)
from openjudge.grader_benchmark.benchmark import BenchmarkResult, GraderBenchmark

__all__ = [
    "GraderBenchmark",
    "BenchmarkResult",
    "AGENT_GRADER_REGISTRY",
    "build_benchmark",
    "get_graders_by_category",
    "get_all_categories",
]
