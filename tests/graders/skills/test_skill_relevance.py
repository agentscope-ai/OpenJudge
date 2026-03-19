#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SkillRelevanceGrader.

Covers two test types:

1. Unit tests (offline, with mocks) — validate grader logic and contract in isolation
2. Quality tests (live, requires API keys) — validate scoring quality against
   labeled cases in ``skill_relevance_cases.json``, all based on the
   ``code-review`` skill from ``.agents/skills/code-review/SKILL.md``.

Test cases cover all three score levels on the 1-3 scale:
    - 3 (direct match)  : task maps exactly to the code-review skill's purpose
    - 2 (partial match) : task shares overlap but requires domain adaptation
    - 1 (poor match)    : task is in a completely different domain

Example:
    Run all tests::

        pytest tests/graders/skills/test_skill_relevance.py -v

    Run only unit tests::

        pytest tests/graders/skills/test_skill_relevance.py -m unit

    Run quality tests (requires OPENAI_API_KEY + OPENAI_BASE_URL)::

        pytest tests/graders/skills/test_skill_relevance.py -m quality
"""

import json
import os
from pathlib import Path
from typing import List, cast
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.base_grader import GraderScore
from openjudge.graders.skills.relevance import SkillRelevanceGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = Path(__file__).parent / "skill_relevance_cases.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSkillRelevanceGraderUnit:
    """Unit tests for SkillRelevanceGrader — all external calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Grader initialises with sensible defaults."""
        mock_model = AsyncMock()
        grader = SkillRelevanceGrader(model=mock_model)

        assert grader.name == "skill_relevance"
        assert grader.threshold == 2
        assert grader.model is mock_model

    def test_initialization_custom_threshold(self):
        """Custom threshold is stored correctly."""
        mock_model = AsyncMock()
        grader = SkillRelevanceGrader(model=mock_model, threshold=3)
        assert grader.threshold == 3

    def test_invalid_threshold_raises(self):
        """Threshold outside [1, 3] must raise ValueError."""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillRelevanceGrader(model=mock_model, threshold=0)
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillRelevanceGrader(model=mock_model, threshold=4)

    # ------------------------------------------------------------------
    # Score 3 — direct match
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_direct_match_score_3(self):
        """Model returns score 3 for a task that directly matches the skill."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": (
                "The skill is explicitly designed for reviewing GitHub Pull Requests "
                "and local git diffs, which exactly matches the task."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillRelevanceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Review a GitHub Pull Request for code quality issues, bugs, "
                    "and security vulnerabilities."
                ),
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code. It supports both local changes "
                    "and remote Pull Requests. Focuses on correctness, maintainability, "
                    "and project standards."
                ),
                skill_md="# Code Review Skill\nReviews PRs and local git diffs.",
            )

        assert result.score == 3
        assert "pull request" in result.reason.lower() or "pr" in result.reason.lower()
        assert result.metadata["threshold"] == 2

    # ------------------------------------------------------------------
    # Score 2 — partial match
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_partial_match_score_2(self):
        """Model returns score 2 for a task with overlapping but not full coverage."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": (
                "The skill covers security checks as part of code review, but the task "
                "requires a dedicated OWASP security audit with CVE scoring, which is "
                "not explicitly supported."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillRelevanceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Run a comprehensive OWASP security audit and generate a report "
                    "with CVE numbers and CVSS scores for each vulnerability found."
                ),
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code for correctness, security, "
                    "maintainability, and project standards."
                ),
                skill_md=(
                    "# Code Review Skill\n"
                    "## Security\n- Check for SQL injection, XSS, hardcoded secrets."
                ),
            )

        assert result.score == 2
        assert "security" in result.reason.lower() or "audit" in result.reason.lower()

    # ------------------------------------------------------------------
    # Score 1 — poor match
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_poor_match_score_1(self):
        """Model returns score 1 for a task from a completely different domain."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": (
                "The skill is designed for code review of git diffs and PRs. "
                "Generating financial reports from CSV data is a completely "
                "different domain with no overlap."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillRelevanceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Generate a quarterly financial report from CSV sales data with "
                    "revenue summaries, growth charts, and PDF export."
                ),
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code. Handles local git changes and "
                    "GitHub Pull Requests. Focuses on correctness and maintainability."
                ),
                skill_md="# Code Review Skill\nReviews code diffs for quality issues.",
            )

        assert result.score == 1
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # Optional skill_md parameter
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evaluation_without_skill_md(self):
        """skill_md defaults to empty string — evaluation still completes."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": "Partial overlap based on name and description only.",
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillRelevanceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review my latest git commit for issues.",
                skill_name="code-review",
                skill_description="Reviews code diffs and PRs.",
                # skill_md intentionally omitted
            )

        assert result.score == 2

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_handling_returns_grader_error(self):
        """API errors are surfaced as GraderError (not raised)."""
        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.side_effect = Exception("Simulated API timeout")
            mock_model = AsyncMock()
            grader = SkillRelevanceGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_name="code-review",
                skill_description="Reviews code.",
            )

        assert hasattr(result, "error")
        assert "Simulated API timeout" in result.error

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_threshold_propagated_to_metadata(self):
        """threshold value appears in GraderScore.metadata."""
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 3, "reason": "Direct match."}

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillRelevanceGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Check my PR for bugs.",
                skill_name="code-review",
                skill_description="Reviews PRs and local diffs.",
            )

        assert result.metadata.get("threshold") == 3


# ---------------------------------------------------------------------------
# Helpers shared by quality test classes
# ---------------------------------------------------------------------------

_GRADER_MAPPER = {
    "task_description": "task_description",
    "skill_name": "skill_name",
    "skill_description": "skill_description",
    "skill_md": "skill_md",
}


def _load_dataset(skill_group: str | None = None):
    """Load cases from JSON, optionally filtering by ``skill_group``."""
    if not DATA_FILE.exists():
        pytest.skip(f"Test data file not found: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if skill_group is not None:
        cases = [
            c for c in cases
            if c.get("skill_group", "code-review") == skill_group
        ]
    return cases


async def _run_grader(grader: SkillRelevanceGrader, cases: list) -> List[GraderScore]:
    """Flatten cases and evaluate them in one runner pass."""
    flat = [{**c["parameters"], "_index": c["index"]} for c in cases]
    runner = GradingRunner(
        grader_configs={
            "skill_relevance": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
        }
    )
    results = await runner.arun(flat)
    return cast(List[GraderScore], results["skill_relevance"])


def _make_model():
    config = {"model": "qwen-max", "api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        config["base_url"] = OPENAI_BASE_URL
    return OpenAIChatModel(**config)


# ---------------------------------------------------------------------------
# QUALITY TESTS — full dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillRelevanceGraderQuality:
    """Quality tests using all labeled cases in skill_relevance_cases.json.

    The dataset contains cases for two skills:
    - ``code-review`` (indices 0–8)
    - ``financial-consulting-research`` (indices 9–16)

    Each skill group spans scores 1 (poor), 2 (partial), and 3 (direct).
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset()

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, dataset, model):
        """All 17 evaluations return a score in [1, 3] with a non-empty reason."""
        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1 <= result.score <= 3, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

    @pytest.mark.asyncio
    async def test_score_bounds_respected(self, dataset, model):
        """Every case must satisfy its min_expect_score / max_expect_score constraints."""
        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        violations = []
        for case, result in zip(dataset, results):
            score = result.score
            idx = case["index"]
            desc = case["description"]

            if "min_expect_score" in case and score < case["min_expect_score"]:
                violations.append(
                    f"Case {idx} ({desc}): score {score} < min {case['min_expect_score']}"
                )
            if "max_expect_score" in case and score > case["max_expect_score"]:
                violations.append(
                    f"Case {idx} ({desc}): score {score} > max {case['max_expect_score']}"
                )

        assert not violations, "Score bound violations:\n" + "\n".join(violations)

    @pytest.mark.asyncio
    async def test_direct_match_cases_score_higher_than_poor_match(self, dataset, model):
        """Score-3 cases should on average score higher than score-1 cases (all skills combined)."""
        grader = SkillRelevanceGrader(model=model, threshold=2)

        direct_cases = [c for c in dataset if c.get("expect_score") == 3]
        poor_cases = [c for c in dataset if c.get("expect_score") == 1]

        direct_results = await _run_grader(grader, direct_cases)
        poor_results = await _run_grader(grader, poor_cases)

        avg_direct = sum(r.score for r in direct_results) / len(direct_results)
        avg_poor = sum(r.score for r in poor_results) / len(poor_results)

        print(f"\nAll skills — avg direct: {avg_direct:.2f}, avg poor: {avg_poor:.2f}")

        assert avg_direct > avg_poor, (
            f"Direct-match avg ({avg_direct:.2f}) should exceed poor-match avg ({avg_poor:.2f})"
        )

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, dataset, model):
        """Same cases run twice should agree ≥ 90% of the time."""
        grader = SkillRelevanceGrader(model=model, threshold=2)

        flat_dataset = [{**c["parameters"], "_index": c["index"]} for c in dataset]
        runner = GradingRunner(
            grader_configs={
                "run1": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
                "run2": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
            }
        )
        results = await runner.arun(flat_dataset)

        run1 = cast(List[GraderScore], results["run1"])
        run2 = cast(List[GraderScore], results["run2"])
        agreements = sum(
            1 for r1, r2 in zip(run1, run2) if r1 and r2 and r1.score == r2.score
        )
        total = len([r for r in run1 if r and r.score is not None])
        consistency = agreements / total if total > 0 else 1.0

        print(f"\nConsistency: {consistency:.2%} ({agreements}/{total})")
        assert consistency >= 0.9, f"Score consistency too low: {consistency:.2%}"


# ---------------------------------------------------------------------------
# QUALITY TESTS — code-review skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillRelevanceCodeReviewGroup:
    """Quality tests restricted to code-review skill cases (indices 0–8)."""

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="code-review")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_score_bounds_code_review(self, dataset, model):
        """All code-review cases satisfy their score bounds."""
        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        violations = []
        for case, result in zip(dataset, results):
            score = result.score
            if "min_expect_score" in case and score < case["min_expect_score"]:
                violations.append(
                    f"Case {case['index']}: score {score} < min {case['min_expect_score']}"
                )
            if "max_expect_score" in case and score > case["max_expect_score"]:
                violations.append(
                    f"Case {case['index']}: score {score} > max {case['max_expect_score']}"
                )

        assert not violations, "code-review score bound violations:\n" + "\n".join(violations)

    @pytest.mark.asyncio
    async def test_direct_beats_poor_code_review(self, dataset, model):
        """Within code-review cases, score-3 avg must exceed score-1 avg."""
        grader = SkillRelevanceGrader(model=model, threshold=2)

        direct = [c for c in dataset if c.get("expect_score") == 3]
        poor = [c for c in dataset if c.get("expect_score") == 1]

        direct_results = await _run_grader(grader, direct)
        poor_results = await _run_grader(grader, poor)

        avg_direct = sum(r.score for r in direct_results) / len(direct_results)
        avg_poor = sum(r.score for r in poor_results) / len(poor_results)

        print(f"\ncode-review — avg direct: {avg_direct:.2f}, avg poor: {avg_poor:.2f}")
        assert avg_direct > avg_poor


# ---------------------------------------------------------------------------
# QUALITY TESTS — financial-consulting-research skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillRelevanceFinancialConsultingGroup:
    """Quality tests restricted to financial-consulting-research skill cases (indices 9–16).

    Covers three score levels:
    - Score 3: tasks that directly match gathering/aggregating consulting firm
      reports (McKinsey/BCG/Deloitte), ESG research, Chinese-language queries.
    - Score 2: tasks with partial overlap — original report authoring or
      automated daily news monitoring.
    - Score 1: completely unrelated tasks — backend code review, AWS
      infrastructure deployment, React dashboard development.
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="financial-consulting-research")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_scores_in_valid_range_financial(self, dataset, model):
        """All financial-consulting-research cases return scores in [1, 3]."""
        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1 <= result.score <= 3, f"Score out of range: {result.score}"
            assert len(result.reason) > 0

    @pytest.mark.asyncio
    async def test_score_bounds_financial_consulting(self, dataset, model):
        """All financial-consulting-research cases satisfy their score bounds."""
        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        violations = []
        for case, result in zip(dataset, results):
            score = result.score
            if "min_expect_score" in case and score < case["min_expect_score"]:
                violations.append(
                    f"Case {case['index']}: score {score} < min {case['min_expect_score']}"
                )
            if "max_expect_score" in case and score > case["max_expect_score"]:
                violations.append(
                    f"Case {case['index']}: score {score} > max {case['max_expect_score']}"
                )

        assert not violations, (
            "financial-consulting-research score bound violations:\n" + "\n".join(violations)
        )

    @pytest.mark.asyncio
    async def test_direct_beats_poor_financial_consulting(self, dataset, model):
        """Score-3 financial consulting cases must average higher than score-1 cases."""
        grader = SkillRelevanceGrader(model=model, threshold=2)

        direct = [c for c in dataset if c.get("expect_score") == 3]
        poor = [c for c in dataset if c.get("expect_score") == 1]

        direct_results = await _run_grader(grader, direct)
        poor_results = await _run_grader(grader, poor)

        avg_direct = sum(r.score for r in direct_results) / len(direct_results)
        avg_poor = sum(r.score for r in poor_results) / len(poor_results)

        print(
            f"\nfinancial-consulting-research — avg direct: {avg_direct:.2f}, "
            f"avg poor: {avg_poor:.2f}"
        )
        assert avg_direct > avg_poor, (
            f"Direct-match avg ({avg_direct:.2f}) should exceed poor-match avg ({avg_poor:.2f})"
        )

    @pytest.mark.asyncio
    async def test_chinese_language_case_scores_direct_match(self, dataset, model):
        """The Chinese-language case (index 10) must receive a score of 3."""
        chinese_case = next((c for c in dataset if c["index"] == 10), None)
        if chinese_case is None:
            pytest.skip("Chinese-language case (index 10) not found in dataset")

        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, [chinese_case])

        assert results[0].score == 3, (
            f"Chinese-language task should be a direct match (score 3), "
            f"got {results[0].score}: {results[0].reason}"
        )


# ---------------------------------------------------------------------------
# QUALITY TESTS — cross-skill routing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillRelevanceCrossSkillRouting:
    """Validate that the grader correctly differentiates between the two skills.

    Key insight: a task that is a direct match (score 3) for one skill should
    be a poor match (score 1) for the other skill, and vice versa.  This tests
    the grader's fitness for skill-routing use cases.
    """

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_code_review_task_scores_poorly_on_financial_skill(self, model):
        """A clear code-review task should score 1 against the financial skill."""
        task = {
            "index": 900,
            "description": "cross-skill: code review task vs financial skill",
            "expect_score": 1,
            "parameters": {
                "task_description": (
                    "Review the open GitHub Pull Request #42. Check for logic errors, "
                    "missing error handling, and security vulnerabilities in the diff."
                ),
                "skill_name": "financial-consulting-research",
                "skill_description": (
                    "Collect and aggregate financial consulting information from the web. "
                    "Searches for market analysis, consulting firm reports, industry insights, "
                    "investment research, economic trends, and financial advisory content."
                ),
                "skill_md": (
                    "---\nname: financial-consulting-research\n"
                    "description: Collect and aggregate financial consulting information.\n---\n\n"
                    "# Financial Consulting Research Skill\n\n"
                    "## When to Use\n\n"
                    "- Find market analysis, industry reports, or consulting firm publications\n"
                    "- Research investment outlook, economic trends, or sector forecasts\n"
                    "- Compare views across consulting firms (McKinsey, BCG, Bain, Deloitte, etc.)"
                ),
            },
        }

        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, [task])

        assert results[0].score <= 2, (
            f"Code-review task should NOT match financial-consulting-research skill well, "
            f"but got score {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_financial_task_scores_poorly_on_code_review_skill(self, model):
        """A clear financial research task should score 1 against the code-review skill."""
        task = {
            "index": 901,
            "description": "cross-skill: financial task vs code-review skill",
            "expect_score": 1,
            "parameters": {
                "task_description": (
                    "Find the latest McKinsey and Deloitte reports on the global AI investment "
                    "outlook for 2025. Compare their key recommendations for institutional investors."
                ),
                "skill_name": "code-review",
                "skill_description": (
                    "Use this skill to review code. It supports both local changes "
                    "and remote Pull Requests. Focuses on correctness, maintainability, "
                    "and adherence to project standards."
                ),
                "skill_md": (
                    "---\nname: code-review\n"
                    "description: Reviews code diffs and GitHub Pull Requests.\n---\n\n"
                    "# Code Review Skill\n\n"
                    "## Step 3: Analyze the changes\n\n"
                    "### Correctness\n- Are there bugs, logic errors?\n\n"
                    "### Security\n- Any SQL injection, XSS risks?\n\n"
                    "### Maintainability\n- Is the code readable?"
                ),
            },
        }

        grader = SkillRelevanceGrader(model=model, threshold=2)
        results = await _run_grader(grader, [task])

        assert results[0].score <= 2, (
            f"Financial research task should NOT match code-review skill well, "
            f"but got score {results[0].score}: {results[0].reason}"
        )
