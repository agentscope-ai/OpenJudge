# -*- coding: utf-8 -*-
"""Regression tests for the skills evaluation cookbook.

These tests intentionally avoid importing the cookbook modules so dependency
declaration regressions can be detected in a bare checkout.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "cookbooks" / "skills_evaluation" / "runner.py"


def _module_assignments(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    return assignments


def test_skill_design_scores_use_five_point_scale() -> None:
    assignments = _module_assignments(RUNNER_PATH)
    score_ranges = ast.literal_eval(assignments["_SCORE_RANGES"])
    thresholds = ast.literal_eval(assignments["DEFAULT_THRESHOLDS"])

    assert score_ranges["skill_design"] == (1.0, 5.0)
    assert "skill_structure" not in score_ranges
    assert thresholds["structure"] == 3.0


def test_dimension_errors_fail_overall_result() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "passed = not errors and all(d.passed for d in dimension_scores.values())" in source


def test_cookbook_runtime_dependencies_are_declared() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        dep.split("[", 1)[0].split(">=", 1)[0].split("<", 1)[0].lower()
        for dep in pyproject["project"]["dependencies"]
    }

    assert "pyyaml" in dependencies
    assert "python-dotenv" in dependencies
