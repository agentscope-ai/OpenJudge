# -*- coding: utf-8 -*-
"""Regression tests for the image coherence rubrics cookbook.

These tests intentionally avoid importing the cookbook module (matching the
skills_evaluation precedent) so dependency and structure regressions can be
detected in a bare checkout.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKBOOK_PATH = REPO_ROOT / "cookbooks" / "image_coherence_rubrics" / "image_coherence_rubrics.py"


def _cookbook_tree() -> ast.AST:
    return ast.parse(COOKBOOK_PATH.read_text(encoding="utf-8"))


def test_cookbook_generates_rubrics_once_and_reuses_them() -> None:
    """The generate-once-reuse pattern: one generator call outside the loop,
    one ImageCoherenceGrader built with the generated rubrics."""
    source = COOKBOOK_PATH.read_text(encoding="utf-8")
    tree = _cookbook_tree()

    assert "TaskBasedRubricGenerator(" in source
    assert "await generator.generate(" in source
    assert "ImageCoherenceGrader(model=model, rubrics=rubrics)" in source

    # The rubric generation must happen once, before the per-sample loop.
    generate_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Await) and "generator.generate" in ast.unparse(node)
    )
    loop_line = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.For) and ast.unparse(node).startswith("for sample in SAMPLES:")
    )
    assert generate_line < loop_line


def test_cookbook_compares_against_built_in_rubric_with_human_labels() -> None:
    """The comparison arm: same candidates through the default rubric, and
    agreement with human labels via the repo's CorrelationAnalyzer."""
    source = COOKBOOK_PATH.read_text(encoding="utf-8")

    assert "ImageCoherenceGrader(model=model)  # built-in rubric" in source
    assert "CorrelationAnalyzer()" in source
    assert '"human_score"' in source
