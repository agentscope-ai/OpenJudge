#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SkillStructureGrader.

Covers two test types:

1. Unit tests (offline, with mocks) — validate grader logic and contract in isolation
2. Quality tests (live, requires API keys) — validate scoring quality against
   labeled cases in ``skill_structure_cases.json``, all based on the
   ``code-review`` skill from ``.agents/skills/code-review/SKILL.md``.

Test cases cover all three score levels on the 1-3 scale, exercising the four
structural dimensions evaluated by the grader:
    - D1 Anti-Pattern Quality: expert-grade NEVER list with non-obvious domain reasons
    - D2 Specification Compliance: description answers WHAT + WHEN + searchable KEYWORDS
    - D3 Progressive Disclosure: SKILL.md properly sized; MANDATORY loading triggers
      embedded in workflow (not just listed); no orphan references
    - D4 Freedom Calibration: constraint level per section matches task fragility

Score levels:
    - 3 (structurally sound)   : all four dimensions satisfied
    - 2 (partially sound)      : passes some dimensions but notable gaps in others
    - 1 (structurally poor)    : fails most criteria; no meaningful NEVER list;
                                  description too vague; dump-style or empty SKILL.md;
                                  or severe freedom mismatch

Example:
    Run all tests::

        pytest tests/graders/skills/test_skill_structure.py -v

    Run only unit tests::

        pytest tests/graders/skills/test_skill_structure.py -m unit

    Run quality tests (requires OPENAI_API_KEY + OPENAI_BASE_URL)::

        pytest tests/graders/skills/test_skill_structure.py -m quality
"""

import json
import os
from pathlib import Path
from typing import List, cast
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.base_grader import GraderScore
from openjudge.graders.skills.structure import SkillStructureGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = Path(__file__).parent / "skill_structure_cases.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSkillStructureGraderUnit:
    """Unit tests for SkillStructureGrader — all external calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Grader initialises with sensible defaults."""
        mock_model = AsyncMock()
        grader = SkillStructureGrader(model=mock_model)

        assert grader.name == "skill_structure"
        assert grader.threshold == 2
        assert grader.model is mock_model

    def test_initialization_custom_threshold(self):
        """Custom threshold is stored correctly."""
        mock_model = AsyncMock()
        grader = SkillStructureGrader(model=mock_model, threshold=3)
        assert grader.threshold == 3

    def test_invalid_threshold_raises(self):
        """Threshold outside [1, 3] must raise ValueError."""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillStructureGrader(model=mock_model, threshold=0)
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillStructureGrader(model=mock_model, threshold=4)

    # ------------------------------------------------------------------
    # Score 3 — structurally sound skill
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_structurally_sound_score_3(self):
        """Model returns score 3 for a code review skill with expert NEVER list,
        complete description (WHAT+WHEN+KEYWORDS), appropriate size, and calibrated freedom."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": (
                "D1: Expert NEVER list — 'NEVER label style nitpicks as [Critical] because it "
                "conditions authors to dismiss all Critical flags as noise' is non-obvious domain "
                "knowledge. D2: Description answers WHAT (reviews git diffs and PRs), WHEN ('Use "
                "when asked to review a PR, check a diff'), and contains searchable keywords. "
                "D3: SKILL.md is ~60 lines, well within the 300-line preference, self-contained. "
                "D4: Medium freedom for analysis (criteria + judgment), exact template for output "
                "section — constraint level matches code review's medium fragility."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code changes. Supports local git changes "
                    "(staged or working tree) and GitHub Pull Requests by PR number or URL. "
                    "Use when the user asks to review a PR, check a diff, audit changes, "
                    "or says 'what do you think of my changes?'."
                ),
                skill_md=(
                    "---\nname: code-review\ndescription: ...\n---\n\n"
                    "# Code Review Skill\n\n"
                    "## NEVER\n\n"
                    "- NEVER label style nitpicks as [Critical] — it conditions PR authors to "
                    "dismiss all Critical flags as noise, causing them to miss actual blocking bugs\n"
                    "- NEVER skip reading `git log --oneline -10` before analyzing the diff — "
                    "code that looks like a bug is often an intentional workaround; history context "
                    "prevents false positives\n\n"
                    "## Steps\n\n"
                    "1. `gh pr diff <number>` or `git diff --staged`\n"
                    "2. Analyze correctness, security, maintainability, tests\n"
                    "3. Write review:\n\n"
                    "```\n## Code Review: [title]\n### Summary\n### Issues\n"
                    "**[Critical]** ...\n**[Major]** ...\n**[Minor]** ...\n```\n\n"
                    "## Severity Guide\n\n"
                    "- Critical: bugs, security failures, CI breakage\n"
                    "- Major: quality issues likely in production\n"
                    "- Minor: style and readability"
                ),
            )

        assert result.score == 3
        assert len(result.reason) > 0
        assert result.metadata["threshold"] == 2

    # ------------------------------------------------------------------
    # Score 2 — partially sound skill
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_partially_sound_score_2(self):
        """Model returns score 2 for a skill with a NEVER list that is generic
        and a description missing WHEN triggers."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 2,
            "reason": (
                "D1: NEVER list present but generic — 'NEVER miss edge cases' and 'NEVER submit "
                "an incomplete review' apply to any task with no domain-specific reasoning; an "
                "expert would not recognise these as hard-won knowledge. D2: Description explains "
                "WHAT (reviews code) but lacks WHEN triggers ('Use when...') and searchable action "
                "keywords. D3: SKILL.md is appropriately brief. D4: Medium freedom appropriate for "
                "code review tasks. Failure patterns detected: Vague Warning [D1], Invisible Skill [D2]."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description=(
                    "Reviews code diffs and Pull Requests for bugs, security issues, "
                    "and maintainability problems."
                ),
                skill_md=(
                    "---\nname: code-review\ndescription: Reviews code.\n---\n\n"
                    "# Code Review Skill\n\n"
                    "## NEVER\n\n"
                    "- NEVER submit an incomplete review\n"
                    "- NEVER miss edge cases or error handling\n"
                    "- NEVER forget to check security aspects\n\n"
                    "## Steps\n\n"
                    "1. Get the diff using `git diff` or `gh pr diff <number>`\n"
                    "2. Analyze for correctness, security, and maintainability\n"
                    "3. Write a review with severity labels\n\n"
                    "## Output\n\n"
                    "Provide a structured review with a Summary section and an Issues section."
                ),
            )

        assert result.score == 2
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # Score 1 — structurally poor skill
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_structurally_poor_score_1(self):
        """Model returns score 1 for a skill with no NEVER list, vague description,
        and a trivial SKILL.md placeholder."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": (
                "D1: No NEVER list — absent entirely. D2: Description is 'A skill for reviewing "
                "code' — fails WHAT (no specifics), WHEN (no triggers), and KEYWORDS (none "
                "searchable). D3: SKILL.md is 2 lines of vague prose with no steps, tools, or "
                "output format — purely a dump of intent. D4: No constraint guidance at all. "
                "Failure patterns: Vague Warning [D1], Invisible Skill [D2], The Dump [D3]."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description="A skill for reviewing code.",
                skill_md=(
                    "# Code Review Skill\n\n"
                    "Review the code and provide feedback. Check for bugs and style problems."
                ),
            )

        assert result.score == 1
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # Score 1 — empty SKILL.md
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_skill_md_scores_1(self):
        """Model returns score 1 when skill_md is empty — per rubric this is automatic."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "SKILL.md content is empty; automatic score 1 per rubric constraint.",
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description="Use this skill to review code.",
                skill_md="",
            )

        assert result.score == 1
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # Score 1 — freedom mismatch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_freedom_mismatch_scores_low(self):
        """Model returns score 1 for a skill that imposes rigid mechanical scripts
        on a creative judgment task (code review requires expert judgment, not
        deterministic pattern matching)."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": (
                "D4: Severe freedom mismatch — code review is a judgment-based task (medium "
                "freedom: criteria + expert judgment), but the skill imposes rigid mechanical "
                "scripts ('for each changed line: if line contains if: check condition', "
                "'output must follow this exact format') that stifle valid variation and prevent "
                "the agent from applying domain expertise. D1: No NEVER list. D2: Description "
                "missing WHEN triggers and keywords."
            ),
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description="Use this skill to review code changes and provide feedback.",
                skill_md=(
                    "---\nname: code-review\ndescription: Review code.\n---\n\n"
                    "# Code Review Skill\n\n"
                    "## Step 1\n\nRun: `git diff --staged`\n\n"
                    "## Step 2\n\n"
                    "For each file in the diff:\n"
                    "  For each changed line:\n"
                    "    If line contains `if`: check if condition is logically correct\n"
                    "    If line contains `for`: verify loop bound is exactly `len(collection) - 1`\n\n"
                    "## Step 3\n\n"
                    "Use [Critical] if and only if the line matches one of these exact patterns:\n"
                    "  - Buffer overflow\n"
                    "  - SQL injection string\n"
                    "  - Hardcoded password string\n\n"
                    "## Step 4\n\n"
                    "Output must follow this exact format:\n"
                    "  Line 1: '## Code Review'\n"
                    "  Line 2: '### Summary'\n"
                    "  Line 3: exactly 2 sentences, no more, no less\n"
                    "  Do not deviate from this format under any circumstances."
                ),
            )

        assert result.score == 1
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # skill_md defaults to empty string when omitted
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evaluation_without_skill_md(self):
        """skill_md defaults to empty string — evaluation still completes."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "SKILL.md content is empty; automatic score 1.",
        }

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description="Reviews code diffs and PRs.",
                # skill_md intentionally omitted — should default to ""
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
            grader = SkillStructureGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description="Reviews code diffs.",
                skill_md="# Code Review\n\nReads diffs.",
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
        mock_response.parsed = {"score": 3, "reason": "Structurally sound skill."}

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillStructureGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="code-review",
                skill_description="Reviews PRs and local diffs.",
                skill_md=(
                    "# Code Review\n\n"
                    "## NEVER\n- NEVER label style issues as Critical.\n\n"
                    "## Steps\n1. Get diff\n2. Analyze\n3. Write review"
                ),
            )

        assert result.metadata.get("threshold") == 3


# ---------------------------------------------------------------------------
# Helpers shared by quality test classes
# ---------------------------------------------------------------------------

_GRADER_MAPPER = {
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


async def _run_grader(grader: SkillStructureGrader, cases: list) -> List[GraderScore]:
    """Flatten cases and evaluate them in one runner pass."""
    flat = [{**c["parameters"], "_index": c["index"]} for c in cases]
    runner = GradingRunner(
        grader_configs={
            "skill_structure": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
        }
    )
    results = await runner.arun(flat)
    return cast(List[GraderScore], results["skill_structure"])


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
class TestSkillStructureGraderQuality:
    """Quality tests using all labeled cases in skill_structure_cases.json.

    The dataset contains 9 cases for the ``code-review`` skill group:
    - Indices 0–2: score 3 (structurally sound across all four dimensions)
    - Indices 3–5: score 2 (partially sound — gaps in D1, D2, or D3)
    - Indices 6–8: score 1 (structurally poor — no NEVER list; vague description;
                             empty SKILL.md; or rigid scripts on creative task)
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
        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1 <= result.score <= 3, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

    @pytest.mark.asyncio
    async def test_score_bounds_respected(self, dataset, model):
        """Every case must satisfy its min_expect_score / max_expect_score constraints."""
        grader = SkillStructureGrader(model=model, threshold=2)
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
    async def test_sound_cases_score_higher_than_poor(self, dataset, model):
        """Score-3 cases should on average score higher than score-1 cases."""
        grader = SkillStructureGrader(model=model, threshold=2)

        sound_cases = [c for c in dataset if c.get("expect_score") == 3]
        poor_cases = [c for c in dataset if c.get("expect_score") == 1]

        sound_results = await _run_grader(grader, sound_cases)
        poor_results = await _run_grader(grader, poor_cases)

        avg_sound = sum(r.score for r in sound_results) / len(sound_results)
        avg_poor = sum(r.score for r in poor_results) / len(poor_results)

        print(f"\nAll cases — avg sound: {avg_sound:.2f}, avg poor: {avg_poor:.2f}")

        assert avg_sound > avg_poor, (
            f"Structurally sound avg ({avg_sound:.2f}) should exceed poor avg ({avg_poor:.2f})"
        )

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, dataset, model):
        """Same cases run twice should agree ≥ 90% of the time."""
        grader = SkillStructureGrader(model=model, threshold=2)

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
class TestSkillStructureCodeReviewGroup:
    """Quality tests restricted to the code-review skill cases (all 9 cases).

    Covers three structural quality levels:

    - Score 3: Expert NEVER list with non-obvious domain reasoning; description
      answers WHAT + WHEN + contains searchable keywords; SKILL.md properly sized
      with constraint level calibrated per section.
    - Score 2: Passes some structural dimensions but has notable gaps — NEVER list
      exists but is generic (no domain-specific reasoning); or description missing
      WHEN triggers; or references listed but never loaded via embedded MANDATORY triggers.
    - Score 1: Fails most criteria — no NEVER list; description too vague to trigger;
      trivial placeholder SKILL.md; or rigid mechanical scripts imposed on a creative
      judgment task (severe D4 freedom mismatch); or empty SKILL.md.
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
        grader = SkillStructureGrader(model=model, threshold=2)
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
    async def test_sound_beats_poor_code_review(self, dataset, model):
        """Within code-review cases, score-3 avg must exceed score-1 avg."""
        grader = SkillStructureGrader(model=model, threshold=2)

        sound = [c for c in dataset if c.get("expect_score") == 3]
        poor = [c for c in dataset if c.get("expect_score") == 1]

        sound_results = await _run_grader(grader, sound)
        poor_results = await _run_grader(grader, poor)

        avg_sound = sum(r.score for r in sound_results) / len(sound_results)
        avg_poor = sum(r.score for r in poor_results) / len(poor_results)

        print(f"\ncode-review — avg sound: {avg_sound:.2f}, avg poor: {avg_poor:.2f}")
        assert avg_sound > avg_poor

    @pytest.mark.asyncio
    async def test_empty_skill_md_scores_1(self, dataset, model):
        """The empty SKILL.md case (index 8) must receive score 1."""
        empty_case = next((c for c in dataset if c["index"] == 8), None)
        if empty_case is None:
            pytest.skip("Empty SKILL.md case (index 8) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [empty_case])

        assert results[0].score == 1, (
            f"Skill with empty SKILL.md should score 1 (structurally poor), "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_vague_placeholder_scores_1(self, dataset, model):
        """The vague two-line placeholder SKILL.md (index 6) must receive score 1."""
        placeholder_case = next((c for c in dataset if c["index"] == 6), None)
        if placeholder_case is None:
            pytest.skip("Vague placeholder case (index 6) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [placeholder_case])

        assert results[0].score == 1, (
            f"Skill with no NEVER list, vague description, and trivial SKILL.md "
            f"should score 1, got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_expert_never_list_with_full_structure_scores_high(self, dataset, model):
        """The fully structured code review skill (index 0) must score at least 2."""
        full_case = next((c for c in dataset if c["index"] == 0), None)
        if full_case is None:
            pytest.skip("Full structure case (index 0) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [full_case])

        assert results[0].score >= 2, (
            f"Skill with expert NEVER list, complete description (WHAT+WHEN+KEYWORDS), "
            f"and calibrated freedom should score at least 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_generic_never_list_penalized(self, dataset, model):
        """The generic NEVER list case (index 3) must score at most 2 — generic
        anti-patterns with no domain-specific reasoning are penalized under D1."""
        generic_case = next((c for c in dataset if c["index"] == 3), None)
        if generic_case is None:
            pytest.skip("Generic NEVER list case (index 3) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [generic_case])

        assert results[0].score <= 2, (
            f"Skill with generic NEVER list ('never miss edge cases', 'never be incomplete') "
            f"and missing WHEN triggers should score at most 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_orphan_references_penalized(self, dataset, model):
        """The orphan references case (index 5) must score at most 2 — references
        listed at the end without embedded MANDATORY loading triggers are penalized under D3."""
        orphan_case = next((c for c in dataset if c["index"] == 5), None)
        if orphan_case is None:
            pytest.skip("Orphan references case (index 5) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [orphan_case])

        assert results[0].score <= 2, (
            f"Skill with references listed at end but no MANDATORY workflow triggers "
            f"should score at most 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_mandatory_triggered_references_scores_high(self, dataset, model):
        """The skill with MANDATORY loading triggers embedded in workflow (index 2)
        must score at least 2."""
        triggered_case = next((c for c in dataset if c["index"] == 2), None)
        if triggered_case is None:
            pytest.skip("MANDATORY triggers case (index 2) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [triggered_case])

        assert results[0].score >= 2, (
            f"Skill with MANDATORY loading triggers embedded at workflow decision points "
            f"should score at least 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_missing_when_in_description_penalized(self, dataset, model):
        """The description-missing-WHEN case (index 4) must score at most 2 — descriptions
        without WHEN triggers make the skill discoverable only by chance under D2."""
        missing_when_case = next((c for c in dataset if c["index"] == 4), None)
        if missing_when_case is None:
            pytest.skip("Missing WHEN case (index 4) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [missing_when_case])

        assert results[0].score <= 2, (
            f"Skill with description missing WHEN triggers should score at most 2, "
            f"got {results[0].score}: {results[0].reason}"
        )


# ---------------------------------------------------------------------------
# QUALITY TESTS — financial-consulting-research skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillStructureFinancialConsultingGroup:
    """Quality tests restricted to the financial-consulting-research skill cases (indices 9–17).

    Covers three structural quality levels:

    - Score 3: Expert NEVER list with non-obvious domain reasons specific to financial
      research (consulting bias in forecasts, attribution preservation, publication date
      staleness, paywalled access); description answers WHAT + WHEN + KEYWORDS including
      firm names; SKILL.md properly sized with MANDATORY loading triggers embedded at
      workflow decision points.
    - Score 2: Passes some structural dimensions but has notable gaps — NEVER list exists
      but is generic ("never fabricate data"); description has WHAT+keywords but missing
      WHEN triggers; or the actual SKILL.md (no NEVER list + orphan reference to sources.md).
    - Score 1: Fails most criteria — no NEVER list; vague description; trivial two-line
      placeholder; rigid mechanical extraction scripts imposed on a creative research+synthesis
      task (severe D4 freedom mismatch); or empty SKILL.md.
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
        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1 <= result.score <= 3, f"Score out of range: {result.score}"
            assert len(result.reason) > 0, "Reason should not be empty"

    @pytest.mark.asyncio
    async def test_score_bounds_financial_consulting(self, dataset, model):
        """All financial-consulting-research cases satisfy their score bounds."""
        grader = SkillStructureGrader(model=model, threshold=2)
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
    async def test_sound_beats_poor_financial_consulting(self, dataset, model):
        """Score-3 financial cases must average higher than score-1 cases."""
        grader = SkillStructureGrader(model=model, threshold=2)

        sound = [c for c in dataset if c.get("expect_score") == 3]
        poor = [c for c in dataset if c.get("expect_score") == 1]

        sound_results = await _run_grader(grader, sound)
        poor_results = await _run_grader(grader, poor)

        avg_sound = sum(r.score for r in sound_results) / len(sound_results)
        avg_poor = sum(r.score for r in poor_results) / len(poor_results)

        print(
            f"\nfinancial-consulting-research — avg sound: {avg_sound:.2f}, "
            f"avg poor: {avg_poor:.2f}"
        )
        assert avg_sound > avg_poor, (
            f"Sound avg ({avg_sound:.2f}) should exceed poor avg ({avg_poor:.2f})"
        )

    @pytest.mark.asyncio
    async def test_expert_never_list_with_mandatory_trigger_scores_high(self, dataset, model):
        """The fully structured skill with MANDATORY references trigger (index 9) must score
        at least 2 — expert NEVER list + complete description + embedded loading trigger."""
        case = next((c for c in dataset if c["index"] == 9), None)
        if case is None:
            pytest.skip("Case index 9 not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [case])

        assert results[0].score >= 2, (
            f"Skill with expert NEVER list, complete description, and MANDATORY references "
            f"trigger should score at least 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_actual_skill_md_scores_partial(self, dataset, model):
        """The actual financial-consulting-research SKILL.md (index 12) should score at most 2
        because it has no NEVER list and the reference to sources.md is an orphan (no
        MANDATORY loading trigger embedded in the workflow)."""
        actual_case = next((c for c in dataset if c["index"] == 12), None)
        if actual_case is None:
            pytest.skip("Actual SKILL.md case (index 12) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [actual_case])

        assert results[0].score <= 2, (
            f"Skill with no NEVER list and an orphan reference (sources.md mentioned but "
            f"never triggered) should score at most 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_generic_never_list_penalized_financial(self, dataset, model):
        """The generic NEVER list case (index 13) must score at most 2 — 'NEVER fabricate
        data' and 'NEVER skip citing sources' apply to any task with no domain reasoning."""
        generic_case = next((c for c in dataset if c["index"] == 13), None)
        if generic_case is None:
            pytest.skip("Generic NEVER list case (index 13) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [generic_case])

        assert results[0].score <= 2, (
            f"Skill with generic NEVER list and missing WHEN triggers should score at most 2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_missing_when_in_description_penalized_financial(self, dataset, model):
        """The description-missing-WHEN case (index 14) must score at most 2 — expert NEVER
        list and good workflow cannot compensate for a description that lacks WHEN triggers."""
        missing_when_case = next((c for c in dataset if c["index"] == 14), None)
        if missing_when_case is None:
            pytest.skip("Missing WHEN case (index 14) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [missing_when_case])

        assert results[0].score <= 2, (
            f"Skill with description missing WHEN triggers should score at most 2 "
            f"(D2 failure — Invisible Skill pattern), "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_rigid_scripts_score_1_financial(self, dataset, model):
        """The rigid mechanical extraction scripts case (index 16) must score 1 — imposing
        'exactly 3 keywords', 'exactly 5 findings' on a creative research task is a severe
        D4 freedom mismatch."""
        rigid_case = next((c for c in dataset if c["index"] == 16), None)
        if rigid_case is None:
            pytest.skip("Rigid scripts case (index 16) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [rigid_case])

        assert results[0].score <= 2, (
            f"Skill imposing rigid mechanical scripts on a creative research+synthesis task "
            f"should score 1 or 2 (D4 freedom mismatch), "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_empty_skill_md_scores_1_financial(self, dataset, model):
        """The empty SKILL.md case (index 17) must receive score 1."""
        empty_case = next((c for c in dataset if c["index"] == 17), None)
        if empty_case is None:
            pytest.skip("Empty SKILL.md case (index 17) not found in dataset")

        grader = SkillStructureGrader(model=model, threshold=2)
        results = await _run_grader(grader, [empty_case])

        assert results[0].score == 1, (
            f"Skill with empty SKILL.md should score 1 (automatic per rubric), "
            f"got {results[0].score}: {results[0].reason}"
        )
