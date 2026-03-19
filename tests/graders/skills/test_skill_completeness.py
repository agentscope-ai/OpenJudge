#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SkillCompletenessGrader.

Covers two test types:

1. Unit tests (offline, with mocks) — validate grader logic and contract in isolation
2. Quality tests (live, requires API keys) — validate scoring quality against
   labeled cases in ``skill_completeness_cases.json``, all based on the
   ``code-review`` skill from ``.agents/skills/code-review/SKILL.md``.

Test cases cover all three score levels on the 1-3 scale:
    - 3 (complete)           : skill_md has explicit steps, inputs/outputs, prerequisites, and edge cases
    - 2 (partially complete) : goal is clear but steps/prerequisites are underspecified
    - 1 (incomplete)         : too vague to act on, missing core steps, or placeholder implementation

Example:
    Run all tests::

        pytest tests/graders/skills/test_skill_completeness.py -v

    Run only unit tests::

        pytest tests/graders/skills/test_skill_completeness.py -m unit

    Run quality tests (requires OPENAI_API_KEY + OPENAI_BASE_URL)::

        pytest tests/graders/skills/test_skill_completeness.py -m quality
"""

import json
import os
from pathlib import Path
from typing import List, cast
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.base_grader import GraderScore
from openjudge.graders.skills.completeness import SkillCompletenessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = Path(__file__).parent / "skill_completeness_cases.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSkillCompletenessGraderUnit:
    """Unit tests for SkillCompletenessGrader — all external calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Grader initialises with sensible defaults."""
        mock_model = AsyncMock()
        grader = SkillCompletenessGrader(model=mock_model)

        assert grader.name == "skill_completeness"
        assert grader.threshold == 2
        assert grader.model is mock_model

    def test_initialization_custom_threshold(self):
        """Custom threshold is stored correctly."""
        mock_model = AsyncMock()
        grader = SkillCompletenessGrader(model=mock_model, threshold=3)
        assert grader.threshold == 3

    def test_invalid_threshold_raises(self):
        """Threshold outside [1, 3] must raise ValueError."""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillCompletenessGrader(model=mock_model, threshold=0)
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillCompletenessGrader(model=mock_model, threshold=4)

    # ------------------------------------------------------------------
    # Score 3 — complete skill
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_complete_skill_score_3(self):
        """Model returns score 3 for a skill with explicit steps, prerequisites, and output format."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": (
                "The skill provides explicit steps with tool commands (gh pr diff, git diff), "
                "lists prerequisites (gh CLI, git), defines an output template with severity labels, "
                "and addresses failure modes such as missing authentication."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Review a GitHub Pull Request for code quality issues, bugs, "
                    "security vulnerabilities, and adherence to project standards. "
                    "Provide prioritized feedback with severity labels."
                ),
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code. Supports local changes and remote "
                    "Pull Requests. Focuses on correctness, maintainability, and standards."
                ),
                skill_md=(
                    "# Code Review Skill\n\n"
                    "## Prerequisites\n- git installed\n- gh CLI authenticated\n\n"
                    "## Steps\n"
                    "1. `gh pr diff <number>` — fetch the diff\n"
                    "2. `gh pr view <number>` — read title and description\n"
                    "3. Analyze for correctness, security, maintainability\n"
                    "4. Write review with Critical/Major/Minor issues\n\n"
                    "## Output\n```\n### Summary\n### Issues\n**[Critical]** ...\n```\n\n"
                    "## Failure Modes\n- If gh not installed, prompt user to install."
                ),
            )

        assert result.score == 3
        assert "step" in result.reason.lower() or "prerequisite" in result.reason.lower() or "output" in result.reason.lower()
        assert result.metadata["threshold"] == 2

    # ------------------------------------------------------------------
    # Score 2 — partially complete skill
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_partial_skill_score_2(self):
        """Model returns score 2 for a skill that has a clear goal but missing prerequisites and output format."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": (
                "The skill describes what to check (correctness, security, maintainability) "
                "but does not specify tool commands, prerequisites, or an output format template. "
                "The user cannot act on it without significant guesswork."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Review a GitHub Pull Request for code quality issues, bugs, "
                    "and security vulnerabilities. Provide structured feedback."
                ),
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code changes for quality issues."
                ),
                skill_md=(
                    "# Code Review Skill\n\n"
                    "## What to Check\n"
                    "- Correctness: look for bugs and edge cases\n"
                    "- Security: watch for injection risks and hardcoded secrets\n"
                    "- Maintainability: is the code readable?\n\n"
                    "## Output\n"
                    "Provide a structured review with a summary and list of issues by severity."
                ),
            )

        assert result.score == 2
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # Score 1 — incomplete skill
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_incomplete_skill_score_1(self):
        """Model returns score 1 for a skill that is too vague to act on."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": (
                "The skill provides no actionable steps, no tool commands, no output format, "
                "and no prerequisites. 'Review the code and provide feedback' is not sufficient "
                "to accomplish the task."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Review a GitHub Pull Request for code quality, bugs, and security issues."
                ),
                skill_name="code-review",
                skill_description="Use this skill to review code for correctness and maintainability.",
                skill_md="# Code Review Skill\n\nReview the code and provide feedback on quality, bugs, and security.",
            )

        assert result.score == 1
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # Score 1 — empty skill_md
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_skill_md_score_1(self):
        """Empty skill_md must produce score 1 per grader constraints."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "SKILL.md content is empty — no steps, prerequisites, or output format provided.",
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review my latest git commit and flag any bugs.",
                skill_name="code-review",
                skill_description="Use this skill to review code changes in git.",
                skill_md="",
            )

        assert result.score == 1

    # ------------------------------------------------------------------
    # Score 1 — placeholder implementation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_placeholder_implementation_score_1(self):
        """Skill that promises significant capabilities but delivers trivial placeholder must score 1."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": (
                "The skill description promises OWASP CVE scanning with CVSS scores and Snyk integration, "
                "but the SKILL.md content contains only three trivial placeholder steps with no real logic."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description=(
                    "Run a comprehensive OWASP-compliant security audit, identify CVEs, "
                    "assign CVSS scores, and generate a remediation report."
                ),
                skill_name="code-review",
                skill_description=(
                    "Comprehensive security code review with OWASP compliance, CVE identification, "
                    "CVSS scoring, and automated Snyk/Semgrep scanning."
                ),
                skill_md=(
                    "# Security Code Review Skill\n\n"
                    "This skill performs a full OWASP-compliant security audit with CVE identification, "
                    "CVSS scoring, and Snyk/Semgrep integration.\n\n"
                    "## Steps\n\n"
                    "1. Get the code.\n"
                    "2. Check for security issues.\n"
                    "3. Report findings.\n\n"
                    "## Output\n\nA security audit report."
                ),
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
            "score": 1,
            "reason": "No SKILL.md content provided; cannot assess completeness.",
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review my latest git commit for issues.",
                skill_name="code-review",
                skill_description="Reviews code diffs and PRs.",
                # skill_md intentionally omitted
            )

        assert result.score == 1

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
            grader = SkillCompletenessGrader(model=mock_model)
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
        mock_response.parsed = {"score": 3, "reason": "Complete skill with all required elements."}

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR for code quality.",
                skill_name="code-review",
                skill_description="Reviews PRs and local diffs.",
                skill_md="# Code Review\n## Steps\n1. Fetch diff\n2. Analyze\n3. Write review",
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


async def _run_grader(grader: SkillCompletenessGrader, cases: list) -> List[GraderScore]:
    """Flatten cases and evaluate them in one runner pass."""
    flat = [{**c["parameters"], "_index": c["index"]} for c in cases]
    runner = GradingRunner(
        grader_configs={
            "skill_completeness": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
        }
    )
    results = await runner.arun(flat)
    return cast(List[GraderScore], results["skill_completeness"])


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
class TestSkillCompletenessGraderQuality:
    """Quality tests using all labeled cases in skill_completeness_cases.json.

    The dataset contains 9 cases for the ``code-review`` skill group:
    - Indices 0–2: score 3 (complete)
    - Indices 3–5: score 2 (partially complete)
    - Indices 6–8: score 1 (incomplete)
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset()

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, dataset, model):
        """All 9 evaluations return a score in [1, 3] with a non-empty reason."""
        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1 <= result.score <= 3, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

    @pytest.mark.asyncio
    async def test_score_bounds_respected(self, dataset, model):
        """Every case must satisfy its min_expect_score / max_expect_score constraints."""
        grader = SkillCompletenessGrader(model=model, threshold=2)
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
    async def test_complete_cases_score_higher_than_incomplete(self, dataset, model):
        """Score-3 cases should on average score higher than score-1 cases."""
        grader = SkillCompletenessGrader(model=model, threshold=2)

        complete_cases = [c for c in dataset if c.get("expect_score") == 3]
        incomplete_cases = [c for c in dataset if c.get("expect_score") == 1]

        complete_results = await _run_grader(grader, complete_cases)
        incomplete_results = await _run_grader(grader, incomplete_cases)

        avg_complete = sum(r.score for r in complete_results) / len(complete_results)
        avg_incomplete = sum(r.score for r in incomplete_results) / len(incomplete_results)

        print(f"\nAll cases — avg complete: {avg_complete:.2f}, avg incomplete: {avg_incomplete:.2f}")

        assert avg_complete > avg_incomplete, (
            f"Complete avg ({avg_complete:.2f}) should exceed incomplete avg ({avg_incomplete:.2f})"
        )

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, dataset, model):
        """Same cases run twice should agree ≥ 90% of the time."""
        grader = SkillCompletenessGrader(model=model, threshold=2)

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

        def _has_score(r) -> bool:
            return r is not None and hasattr(r, "score") and r.score is not None

        agreements = sum(
            1 for r1, r2 in zip(run1, run2)
            if _has_score(r1) and _has_score(r2) and r1.score == r2.score
        )
        total = len([r for r in run1 if _has_score(r)])
        consistency = agreements / total if total > 0 else 1.0

        print(f"\nConsistency: {consistency:.2%} ({agreements}/{total})")
        assert consistency >= 0.9, f"Score consistency too low: {consistency:.2%}"


# ---------------------------------------------------------------------------
# QUALITY TESTS — code-review skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillCompletenessCodeReviewGroup:
    """Quality tests restricted to the code-review skill cases (all 9 cases).

    Covers three completeness levels:
    - Score 3: SKILL.md with explicit steps, tool commands, prerequisites, output template,
      and failure mode guidance.
    - Score 2: Goal is clear but steps, prerequisites, or output format are underspecified.
    - Score 1: Too vague to act on; empty SKILL.md; or promises significant capabilities
      that the implementation does not actually deliver.
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="code-review")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_score_bounds_code_review(self, dataset, model):
        """All code-review cases satisfy their score bounds."""
        grader = SkillCompletenessGrader(model=model, threshold=2)
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
    async def test_complete_beats_incomplete_code_review(self, dataset, model):
        """Within code-review cases, score-3 avg must exceed score-1 avg."""
        grader = SkillCompletenessGrader(model=model, threshold=2)

        complete = [c for c in dataset if c.get("expect_score") == 3]
        incomplete = [c for c in dataset if c.get("expect_score") == 1]

        complete_results = await _run_grader(grader, complete)
        incomplete_results = await _run_grader(grader, incomplete)

        avg_complete = sum(r.score for r in complete_results) / len(complete_results)
        avg_incomplete = sum(r.score for r in incomplete_results) / len(incomplete_results)

        print(f"\ncode-review — avg complete: {avg_complete:.2f}, avg incomplete: {avg_incomplete:.2f}")
        assert avg_complete > avg_incomplete

    @pytest.mark.asyncio
    async def test_empty_skill_md_cases_score_1(self, dataset, model):
        """The empty SKILL.md case (index 8) must receive a score of 1."""
        empty_case = next((c for c in dataset if c["index"] == 8), None)
        if empty_case is None:
            pytest.skip("Empty SKILL.md case (index 8) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [empty_case])

        assert results[0].score == 1, (
            f"Empty SKILL.md should score 1 (incomplete), "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_placeholder_implementation_scores_1(self, dataset, model):
        """The placeholder SKILL.md case (index 7) — promises OWASP audit but has trivial steps — must score 1."""
        placeholder_case = next((c for c in dataset if c["index"] == 7), None)
        if placeholder_case is None:
            pytest.skip("Placeholder implementation case (index 7) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [placeholder_case])

        assert results[0].score == 1, (
            f"Placeholder skill (promises OWASP CVE but delivers trivial steps) should score 1, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_full_skill_md_scores_3(self, dataset, model):
        """The most complete case (index 0 — full SKILL.md) must receive a score of 3."""
        full_case = next((c for c in dataset if c["index"] == 0), None)
        if full_case is None:
            pytest.skip("Full SKILL.md case (index 0) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [full_case])

        assert results[0].score >= 2, (
            f"Full SKILL.md with steps, prerequisites, output template and failure modes "
            f"should score at least 2, got {results[0].score}: {results[0].reason}"
        )


# ---------------------------------------------------------------------------
# QUALITY TESTS — financial-consulting-research skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillCompletenessFinancialConsultingGroup:
    """Quality tests restricted to financial-consulting-research skill cases (indices 9–17).

    Covers three completeness levels:
    - Score 3: SKILL.md with a 4-step workflow, concrete search query patterns (topic + firm,
      site: operators), named tools (web_search / mcp_web_fetch), a structured output template,
      a common-sources table, language-handling rules, and caveats about paywalls and date
      freshness.
    - Score 2: Goal is clear but steps are vague, search query examples are absent, output
      template is missing, or caveats are not addressed.
    - Score 1: Too vague to act on; empty SKILL.md; or promises significant capabilities
      (Bloomberg Terminal API, real-time sentiment scoring) that the implementation does not
      actually deliver.
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="financial-consulting-research")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_scores_in_valid_range_financial(self, dataset, model):
        """All financial-consulting-research cases return scores in [1, 3] with non-empty reasons."""
        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1 <= result.score <= 3, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

    @pytest.mark.asyncio
    async def test_score_bounds_financial_consulting(self, dataset, model):
        """All financial-consulting-research cases satisfy their score bounds."""
        grader = SkillCompletenessGrader(model=model, threshold=2)
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
    async def test_complete_beats_incomplete_financial_consulting(self, dataset, model):
        """Score-3 financial cases must average higher than score-1 cases."""
        grader = SkillCompletenessGrader(model=model, threshold=2)

        complete = [c for c in dataset if c.get("expect_score") == 3]
        incomplete = [c for c in dataset if c.get("expect_score") == 1]

        complete_results = await _run_grader(grader, complete)
        incomplete_results = await _run_grader(grader, incomplete)

        avg_complete = sum(r.score for r in complete_results) / len(complete_results)
        avg_incomplete = sum(r.score for r in incomplete_results) / len(incomplete_results)

        print(
            f"\nfinancial-consulting-research — avg complete: {avg_complete:.2f}, "
            f"avg incomplete: {avg_incomplete:.2f}"
        )
        assert avg_complete > avg_incomplete, (
            f"Complete avg ({avg_complete:.2f}) should exceed incomplete avg ({avg_incomplete:.2f})"
        )

    @pytest.mark.asyncio
    async def test_empty_skill_md_scores_1_financial(self, dataset, model):
        """The empty SKILL.md case (index 17) must receive a score of 1."""
        empty_case = next((c for c in dataset if c["index"] == 17), None)
        if empty_case is None:
            pytest.skip("Empty SKILL.md case (index 17) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [empty_case])

        assert results[0].score == 1, (
            f"Empty SKILL.md should score 1 (incomplete), "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_placeholder_implementation_scores_1_financial(self, dataset, model):
        """The placeholder case (index 16) — promises Bloomberg API but delivers trivial steps — must score 1."""
        placeholder_case = next((c for c in dataset if c["index"] == 16), None)
        if placeholder_case is None:
            pytest.skip("Placeholder implementation case (index 16) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [placeholder_case])

        assert results[0].score == 1, (
            f"Placeholder skill (promises Bloomberg Terminal API but delivers 4 trivial steps) "
            f"should score 1, got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_full_skill_md_scores_3_financial(self, dataset, model):
        """The most complete case (index 9 — full SKILL.md) must receive a score of 3."""
        full_case = next((c for c in dataset if c["index"] == 9), None)
        if full_case is None:
            pytest.skip("Full SKILL.md case (index 9) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [full_case])

        assert results[0].score >= 2, (
            f"Full SKILL.md with 4-step workflow, search patterns, tools, output template, "
            f"source table, language handling, and caveats should score at least 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_chinese_language_case_scores_3(self, dataset, model):
        """The Chinese-language case (index 10) must receive a score of 3."""
        chinese_case = next((c for c in dataset if c["index"] == 10), None)
        if chinese_case is None:
            pytest.skip("Chinese-language case (index 10) not found in dataset")

        grader = SkillCompletenessGrader(model=model, threshold=2)
        results = await _run_grader(grader, [chinese_case])

        assert results[0].score >= 2, (
            f"Complete bilingual SKILL.md for Chinese-language task should score at least 2, "
            f"got {results[0].score}: {results[0].reason}"
        )
