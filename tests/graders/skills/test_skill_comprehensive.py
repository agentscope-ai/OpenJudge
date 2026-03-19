#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SkillComprehensiveGrader.

Covers two test types:

1. Unit tests (offline, with mocks) — validate grader logic and contract in isolation,
   including the ``_compute_score`` helper and per-dimension metadata.
2. Quality tests (live, requires API keys) — validate scoring quality against
   labeled cases in ``skill_comprehensive_cases.json``, covering two skill groups:
   - ``code-review`` (indices 0–4)
   - ``financial-consulting-research`` (indices 5–8)

The comprehensive grader evaluates four dimensions in a single LLM call:
    - Relevance    (weight 0.4): how well the skill matches the task
    - Completeness (weight 0.3): whether the skill provides sufficient detail
    - Safety       (weight 0.2): whether the skill avoids dangerous operations
    - Structure    (weight 0.1): NEVER list, description quality, content layering

The final score is a weighted aggregate in [1.0, 3.0] (float).

Quality tiers in the dataset:
    - ``high``   : all four dimensions excellent → aggregate >= 2.5
    - ``medium`` : mixed dimensions with notable gaps → aggregate 1.5–2.9
    - ``low``    : wrong domain or minimal content → aggregate <= 2.2

Example:
    Run all tests::

        pytest tests/graders/skills/test_skill_comprehensive.py -v

    Run only unit tests::

        pytest tests/graders/skills/test_skill_comprehensive.py -m unit

    Run quality tests (requires OPENAI_API_KEY + OPENAI_BASE_URL)::

        pytest tests/graders/skills/test_skill_comprehensive.py -m quality
"""

import json
import os
from pathlib import Path
from typing import List, cast
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.base_grader import GraderScore
from openjudge.graders.skills.comprehensive import (
    DEFAULT_DIMENSION_WEIGHTS,
    SkillComprehensiveCallback,
    SkillComprehensiveGrader,
    _compute_score,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = Path(__file__).parent / "skill_comprehensive_cases.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parsed(
    relevance: int = 3,
    completeness: int = 3,
    safety: int = 3,
    structure: int = 3,
    reason: str = "Overall assessment.",
) -> SkillComprehensiveCallback:
    """Create a SkillComprehensiveCallback instance for use in mocked LLM responses."""
    return SkillComprehensiveCallback(
        relevance_score=relevance,
        relevance_reason=f"Relevance score {relevance}.",
        completeness_score=completeness,
        completeness_reason=f"Completeness score {completeness}.",
        safety_score=safety,
        safety_reason=f"Safety score {safety}.",
        structure_score=structure,
        structure_reason=f"Structure score {structure}.",
        reason=reason,
    )


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSkillComprehensiveGraderUnit:
    """Unit tests for SkillComprehensiveGrader — all external calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Grader initialises with sensible defaults."""
        mock_model = AsyncMock()
        grader = SkillComprehensiveGrader(model=mock_model)

        assert grader.name == "skill_comprehensive"
        assert grader.threshold == 2
        assert grader.model is mock_model
        assert grader.dimension_weights == DEFAULT_DIMENSION_WEIGHTS

    def test_initialization_custom_threshold(self):
        """Custom threshold is stored correctly."""
        mock_model = AsyncMock()
        grader = SkillComprehensiveGrader(model=mock_model, threshold=3)
        assert grader.threshold == 3

    def test_initialization_custom_dimension_weights(self):
        """Custom dimension weights are merged with defaults."""
        mock_model = AsyncMock()
        grader = SkillComprehensiveGrader(
            model=mock_model,
            dimension_weights={"relevance": 0.6, "completeness": 0.2},
        )
        assert grader.dimension_weights["relevance"] == 0.6
        assert grader.dimension_weights["completeness"] == 0.2
        # Keys not overridden should retain default values
        assert grader.dimension_weights["safety"] == DEFAULT_DIMENSION_WEIGHTS["safety"]
        assert grader.dimension_weights["structure"] == DEFAULT_DIMENSION_WEIGHTS["structure"]

    def test_invalid_threshold_raises(self):
        """Threshold outside [1, 3] must raise ValueError."""
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillComprehensiveGrader(model=mock_model, threshold=0)
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillComprehensiveGrader(model=mock_model, threshold=4)

    # ------------------------------------------------------------------
    # _compute_score helper
    # ------------------------------------------------------------------

    def test_compute_score_all_3s(self):
        """All dimension scores of 3 produce a final score of 3.0."""
        parsed = _make_parsed(3, 3, 3, 3)
        score = _compute_score(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert score == 3.0

    def test_compute_score_all_1s(self):
        """All dimension scores of 1 produce a final score of 1.0."""
        parsed = _make_parsed(1, 1, 1, 1)
        score = _compute_score(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert score == 1.0

    def test_compute_score_mixed_weighted(self):
        """Weighted aggregate is correctly computed from mixed dimension scores."""
        parsed = _make_parsed(relevance=3, completeness=2, safety=3, structure=1)
        # Default weights: relevance=0.4, completeness=0.3, safety=0.2, structure=0.1
        expected = round((3 * 0.4 + 2 * 0.3 + 3 * 0.2 + 1 * 0.1) / 1.0, 1)
        score = _compute_score(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert score == expected

    def test_compute_score_custom_weights(self):
        """Custom weights correctly shift the aggregate score."""
        parsed = _make_parsed(relevance=1, completeness=3, safety=3, structure=3)
        weights = {"relevance": 0.9, "completeness": 0.033, "safety": 0.033, "structure": 0.034}
        score = _compute_score(parsed, weights)
        # Relevance dominates: score should be closer to 1 than 3
        assert score < 2.0

    def test_compute_score_zero_total_weight_returns_1(self):
        """When total weight is zero, _compute_score returns 1.0 without raising."""
        parsed = _make_parsed(3, 3, 3, 3)
        score = _compute_score(parsed, {"relevance": 0.0, "completeness": 0.0, "safety": 0.0, "structure": 0.0})
        assert score == 1.0

    def test_compute_score_result_is_rounded_to_1_decimal(self):
        """Final score is always rounded to 1 decimal place."""
        parsed = _make_parsed(relevance=2, completeness=3, safety=1, structure=2)
        # (2*0.4 + 3*0.3 + 1*0.2 + 2*0.1) / 1.0 = 0.8+0.9+0.2+0.2 = 2.1
        score = _compute_score(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert score == round(score, 1)

    # ------------------------------------------------------------------
    # All-3s response → score 3.0
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_all_dimensions_score_3(self):
        """LLM returning all dimension scores of 3 yields final score 3.0."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(3, 3, 3, 3, reason="Excellent skill across all four dimensions.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a GitHub Pull Request for code quality issues.",
                skill_name="code-review",
                skill_description=(
                    "Use this skill to review code. Supports PRs and local diffs. "
                    "Use when: reviewing code changes, auditing PRs."
                ),
                skill_md=(
                    "---\nname: code-review\ndescription: Review code. Use when: reviewing PRs.\n---\n"
                    "# Code Review\n## NEVER\n- NEVER suggest out-of-scope refactors.\n"
                    "## Steps\n1. `gh pr diff` — fetch diff\n2. Analyze for bugs and security"
                ),
            )

        assert result.score == 3.0
        assert len(result.reason) > 0
        assert result.metadata["relevance_score"] == 3
        assert result.metadata["completeness_score"] == 3
        assert result.metadata["safety_score"] == 3
        assert result.metadata["structure_score"] == 3

    # ------------------------------------------------------------------
    # All-1s response → score 1.0
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_all_dimensions_score_1(self):
        """LLM returning all dimension scores of 1 yields final score 1.0."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(1, 1, 1, 1, reason="Poor skill across all dimensions.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Deploy the app to Kubernetes.",
                skill_name="paper-review",
                skill_description="Review academic papers.",
                skill_md="# Paper Review\n\nReview papers.",
            )

        assert result.score == 1.0

    # ------------------------------------------------------------------
    # Mixed dimension scores — verify weighted calculation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_mixed_dimension_scores_weighted_aggregate(self):
        """Mixed dimension scores produce the correct weighted aggregate."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(
            relevance=3,
            completeness=2,
            safety=1,
            structure=2,
            reason="Good relevance, partial completeness, unsafe operations, partial structure.",
        )

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review my code changes.",
                skill_name="code-review",
                skill_description="Reviews code for quality issues.",
                skill_md="# Code Review\n## Steps\n1. Get diff\n2. Analyze",
            )

        # (3*0.4 + 2*0.3 + 1*0.2 + 2*0.1) / 1.0 = 1.2+0.6+0.2+0.2 = 2.2
        expected = round(3 * 0.4 + 2 * 0.3 + 1 * 0.2 + 2 * 0.1, 1)
        assert result.score == expected

    # ------------------------------------------------------------------
    # Custom dimension_weights — verify they override defaults
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_custom_dimension_weights_shift_score(self):
        """A relevance-heavy weight scheme boosts the relevance dimension's impact."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(
            relevance=3, completeness=1, safety=1, structure=1,
            reason="Great relevance, poor everything else.",
        )

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()

            # With default weights, score ≈ 1.8; with relevance=0.9 it should be > 2.5
            grader = SkillComprehensiveGrader(
                model=mock_model,
                dimension_weights={"relevance": 0.9, "completeness": 0.033, "safety": 0.033, "structure": 0.034},
            )
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_name="code-review",
                skill_description="Reviews PRs.",
                skill_md="# Code Review\n",
            )

        assert result.score > 2.5, (
            f"Relevance-heavy weights should push score above 2.5 when relevance=3, "
            f"got {result.score}"
        )

    # ------------------------------------------------------------------
    # Metadata structure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_metadata_contains_all_required_fields(self):
        """GraderScore.metadata contains all expected per-dimension and configuration keys."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(2, 2, 3, 1, reason="Mixed quality.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review code changes.",
                skill_name="code-review",
                skill_description="Code review skill.",
                skill_md="# Review\n",
            )

        required_keys = {
            "relevance_score", "relevance_reason",
            "completeness_score", "completeness_reason",
            "safety_score", "safety_reason",
            "structure_score", "structure_reason",
            "dimension_weights",
            "threshold",
        }
        assert required_keys.issubset(set(result.metadata.keys())), (
            f"Missing metadata keys: {required_keys - set(result.metadata.keys())}"
        )

    @pytest.mark.asyncio
    async def test_threshold_propagated_to_metadata(self):
        """threshold value appears in GraderScore.metadata."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(3, 3, 3, 3, reason="Perfect.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Check my PR for bugs.",
                skill_name="code-review",
                skill_description="Reviews PRs and local diffs.",
                skill_md="# Code Review\n",
            )

        assert result.metadata.get("threshold") == 3

    @pytest.mark.asyncio
    async def test_dimension_weights_reported_in_metadata(self):
        """dimension_weights in metadata reflect the grader's configured weights."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(2, 2, 2, 2)

        custom_weights = {"relevance": 0.5, "completeness": 0.3, "safety": 0.15, "structure": 0.05}

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model, dimension_weights=custom_weights)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_name="code-review",
                skill_description="Reviews code.",
                skill_md="# Review\n",
            )

        assert result.metadata["dimension_weights"]["relevance"] == 0.5
        assert result.metadata["dimension_weights"]["completeness"] == 0.3
        assert result.metadata["dimension_weights"]["safety"] == 0.15
        assert result.metadata["dimension_weights"]["structure"] == 0.05

    @pytest.mark.asyncio
    async def test_per_dimension_scores_stored_in_metadata(self):
        """Per-dimension integer scores from the LLM are correctly stored in metadata."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(
            relevance=3, completeness=1, safety=2, structure=3,
            reason="Mixed.",
        )

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review code.",
                skill_name="code-review",
                skill_description="Reviews code.",
                skill_md="# Review\n",
            )

        assert result.metadata["relevance_score"] == 3
        assert result.metadata["completeness_score"] == 1
        assert result.metadata["safety_score"] == 2
        assert result.metadata["structure_score"] == 3

    # ------------------------------------------------------------------
    # Optional parameters default correctly
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evaluation_without_optional_params(self):
        """scripts and allowed_tools default to empty string — evaluation still completes."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(2, 2, 3, 2, reason="Partial match.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review my latest git commit for issues.",
                skill_name="code-review",
                skill_description="Reviews code diffs and PRs.",
                # skill_md, scripts, allowed_tools intentionally omitted
            )

        assert 1.0 <= result.score <= 3.0

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
            grader = SkillComprehensiveGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_name="code-review",
                skill_description="Reviews code.",
            )

        assert hasattr(result, "error")
        assert "Simulated API timeout" in result.error


# ---------------------------------------------------------------------------
# Helpers shared by quality test classes
# ---------------------------------------------------------------------------

_GRADER_MAPPER = {
    "task_description": "task_description",
    "skill_name": "skill_name",
    "skill_description": "skill_description",
    "skill_md": "skill_md",
    "scripts": "scripts",
    "allowed_tools": "allowed_tools",
}


def _has_score(r) -> bool:
    """Return True if r is a valid GraderScore (not a GraderError)."""
    return r is not None and hasattr(r, "score") and r.score is not None


def _load_dataset(skill_group: str | None = None, quality_tier: str | None = None):
    """Load cases from JSON, optionally filtering by ``skill_group`` or ``quality_tier``."""
    if not DATA_FILE.exists():
        pytest.skip(f"Test data file not found: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if skill_group is not None:
        cases = [
            c for c in cases
            if c.get("skill_group", "code-review") == skill_group
        ]
    if quality_tier is not None:
        cases = [c for c in cases if c.get("quality_tier") == quality_tier]
    return cases


async def _run_grader(grader: SkillComprehensiveGrader, cases: list) -> List[GraderScore]:
    """Flatten cases and evaluate them in one runner pass."""
    flat = [{**c["parameters"], "_index": c["index"]} for c in cases]
    runner = GradingRunner(
        grader_configs={
            "skill_comprehensive": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
        }
    )
    results = await runner.arun(flat)
    return cast(List[GraderScore], results["skill_comprehensive"])


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.5-plus")


def _make_model():
    config = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        config["base_url"] = OPENAI_BASE_URL
    return OpenAIChatModel(**config)


# ---------------------------------------------------------------------------
# QUALITY TESTS — full dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillComprehensiveGraderQuality:
    """Quality tests using all 9 labeled cases in skill_comprehensive_cases.json.

    The dataset covers two skill groups:
    - ``code-review`` (indices 0–4)
    - ``financial-consulting-research`` (indices 5–8)

    Quality tiers:
    - ``high``   : indices 0 and 5 — all four dimensions excellent
    - ``medium`` : indices 1, 3, 6 — direct relevance but execution gaps
    - ``low``    : indices 2, 4, 7, 8 — wrong domain or minimal SKILL.md
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset()

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_scores_in_valid_range(self, dataset, model):
        """All 9 evaluations return a score in [1.0, 3.0] with a non-empty reason."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        errors = [r for r in results if not _has_score(r)]
        assert not errors, f"{len(errors)} evaluation(s) returned GraderError: {[getattr(r, 'error', '') for r in errors]}"
        for result in results:
            assert 1.0 <= result.score <= 3.0, f"Score out of range: {result.score}"
            assert len(result.reason) >= 0, "Reason should be a string"

    @pytest.mark.asyncio
    async def test_score_bounds_respected(self, dataset, model):
        """Every case must satisfy its min_expect_score / max_expect_score constraints."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        violations = []
        for case, result in zip(dataset, results):
            if not _has_score(result):
                violations.append(f"Case {case['index']}: evaluation error — {getattr(result, 'error', 'unknown')}")
                continue
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
    async def test_high_quality_scores_higher_than_low_quality(self, dataset, model):
        """High-quality cases should on average score higher than low-quality cases."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)

        high_cases = [c for c in dataset if c.get("quality_tier") == "high"]
        low_cases = [c for c in dataset if c.get("quality_tier") == "low"]

        high_results = await _run_grader(grader, high_cases)
        low_results = await _run_grader(grader, low_cases)

        valid_high = [r for r in high_results if _has_score(r)]
        valid_low = [r for r in low_results if _has_score(r)]
        assert valid_high and valid_low, "Not enough valid results to compare"

        avg_high = sum(r.score for r in valid_high) / len(valid_high)
        avg_low = sum(r.score for r in valid_low) / len(valid_low)

        print(f"\nAll skills — avg high: {avg_high:.2f}, avg low: {avg_low:.2f}")
        assert avg_high > avg_low, (
            f"High-quality avg ({avg_high:.2f}) should exceed low-quality avg ({avg_low:.2f})"
        )

    @pytest.mark.asyncio
    async def test_per_dimension_scores_present_in_metadata(self, dataset, model):
        """All results include per-dimension scores in metadata."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        for idx, result in enumerate(results):
            if not _has_score(result):
                pytest.fail(f"Result {idx} is GraderError: {getattr(result, 'error', 'unknown')}")
            for dim in ("relevance", "completeness", "safety", "structure"):
                assert f"{dim}_score" in result.metadata, (
                    f"Result {idx}: missing '{dim}_score' in metadata"
                )
                assert 1 <= result.metadata[f"{dim}_score"] <= 3, (
                    f"Result {idx}: {dim}_score {result.metadata[f'{dim}_score']} out of range"
                )

    @pytest.mark.asyncio
    async def test_dimension_weights_reported_correctly(self, dataset, model):
        """Metadata dimension_weights matches the grader's configured DEFAULT_DIMENSION_WEIGHTS."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset[:1])  # test with just one case

        weights = results[0].metadata.get("dimension_weights", {})
        assert weights == DEFAULT_DIMENSION_WEIGHTS, (
            f"Reported weights {weights} differ from configured {DEFAULT_DIMENSION_WEIGHTS}"
        )

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, dataset, model):
        """Same cases run twice should agree within ±0.5 for ≥ 80% of cases."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)

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
            if _has_score(r1) and _has_score(r2) and abs(r1.score - r2.score) <= 0.5
        )
        total = len([r for r in run1 if _has_score(r)])
        consistency = agreements / total if total > 0 else 1.0

        print(f"\nConsistency (±0.5): {consistency:.2%} ({agreements}/{total})")
        assert consistency >= 0.8, f"Score consistency too low: {consistency:.2%}"


# ---------------------------------------------------------------------------
# QUALITY TESTS — code-review skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillComprehensiveCodeReviewGroup:
    """Quality tests restricted to code-review skill cases (indices 0–4).

    Cases:
    - Index 0: All four dimensions excellent → min_expect_score: 2.5
    - Index 1: Direct match but partial quality → 1.5–2.9
    - Index 2: Wrong domain (AWS deployment vs code-review) → max 2.2
    - Index 3: Unsafe skill (unrestricted bash, rm -rf without confirmation) → max 2.5
    - Index 4: Minimal/vague SKILL.md → max 2.0
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
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        violations = []
        for case, result in zip(dataset, results):
            if not _has_score(result):
                violations.append(f"Case {case['index']}: GraderError — {getattr(result, 'error', 'unknown')}")
                continue
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
    async def test_excellent_case_scores_above_2_5(self, dataset, model):
        """The full-quality code-review case (index 0) must score >= 2.5."""
        excellent_case = next((c for c in dataset if c["index"] == 0), None)
        if excellent_case is None:
            pytest.skip("Excellent code-review case (index 0) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [excellent_case])

        assert results[0].score >= 2.5, (
            f"Full-quality code-review skill should score >= 2.5, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_case_scores_low(self, dataset, model):
        """The wrong-domain case (index 2: AWS deployment vs code-review) must score <= 2.2."""
        wrong_domain_case = next((c for c in dataset if c["index"] == 2), None)
        if wrong_domain_case is None:
            pytest.skip("Wrong domain case (index 2) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [wrong_domain_case])

        assert results[0].score <= 2.2, (
            f"Wrong-domain case (AWS deployment vs code-review) should score <= 2.2, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_case_has_low_relevance_score(self, dataset, model):
        """The wrong-domain case (index 2) must have relevance_score == 1 in metadata."""
        wrong_domain_case = next((c for c in dataset if c["index"] == 2), None)
        if wrong_domain_case is None:
            pytest.skip("Wrong domain case (index 2) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [wrong_domain_case])

        relevance = results[0].metadata.get("relevance_score")
        assert relevance == 1, (
            f"AWS deployment task against code-review skill should have relevance_score=1, "
            f"got {relevance}"
        )

    @pytest.mark.asyncio
    async def test_unsafe_case_has_low_safety_score(self, dataset, model):
        """The unsafe skill case (index 3: unrestricted bash, auto-fix, rm -rf) must have safety_score == 1."""
        unsafe_case = next((c for c in dataset if c["index"] == 3), None)
        if unsafe_case is None:
            pytest.skip("Unsafe skill case (index 3) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [unsafe_case])

        safety = results[0].metadata.get("safety_score")
        assert safety == 1, (
            f"Skill with unrestricted bash execution and rm -rf without confirmation "
            f"should have safety_score=1, got {safety}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_minimal_case_scores_low(self, dataset, model):
        """The minimal/vague SKILL.md case (index 4) must score <= 2.0."""
        minimal_case = next((c for c in dataset if c["index"] == 4), None)
        if minimal_case is None:
            pytest.skip("Minimal skill case (index 4) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [minimal_case])

        assert results[0].score <= 2.0, (
            f"Nearly-empty SKILL.md should score <= 2.0, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_high_quality_beats_low_quality_code_review(self, dataset, model):
        """Within code-review cases, high-quality avg must exceed low-quality avg."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)

        high = [c for c in dataset if c.get("quality_tier") == "high"]
        low = [c for c in dataset if c.get("quality_tier") == "low"]

        high_results = await _run_grader(grader, high)
        low_results = await _run_grader(grader, low)

        avg_high = sum(r.score for r in high_results) / len(high_results)
        avg_low = sum(r.score for r in low_results) / len(low_results)

        print(f"\ncode-review — avg high: {avg_high:.2f}, avg low: {avg_low:.2f}")
        assert avg_high > avg_low


# ---------------------------------------------------------------------------
# QUALITY TESTS — financial-consulting-research skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillComprehensiveFinancialConsultingGroup:
    """Quality tests restricted to financial-consulting-research skill cases (indices 5–8).

    Cases:
    - Index 5: All four dimensions excellent, with investment disclaimer → min 2.5
    - Index 6: Same skill without investment disclaimer → safety should be penalised
    - Index 7: Wrong domain (code review task vs financial skill) → max 2.2
    - Index 8: Minimal/vague SKILL.md → max 2.2

    Key contract:
    - Safety dimension should penalise the no-disclaimer case (index 6)
    - Relevance should be 1 for the wrong-domain case (index 7)
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="financial-consulting-research")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_scores_in_valid_range_financial(self, dataset, model):
        """All financial-consulting-research cases return scores in [1.0, 3.0]."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for result in results:
            assert 1.0 <= result.score <= 3.0, f"Score out of range: {result.score}"
            assert len(result.reason) > 0

    @pytest.mark.asyncio
    async def test_score_bounds_financial_consulting(self, dataset, model):
        """All financial-consulting-research cases satisfy their score bounds."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, dataset)

        violations = []
        for case, result in zip(dataset, results):
            if not _has_score(result):
                violations.append(f"Case {case['index']}: GraderError — {getattr(result, 'error', 'unknown')}")
                continue
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
    async def test_excellent_financial_case_scores_above_2_5(self, dataset, model):
        """The full-quality financial case (index 5) with disclaimer must score >= 2.5."""
        excellent_case = next((c for c in dataset if c["index"] == 5), None)
        if excellent_case is None:
            pytest.skip("Excellent financial case (index 5) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [excellent_case])

        assert results[0].score >= 2.5, (
            f"Full-quality financial skill with disclaimer should score >= 2.5, "
            f"got {results[0].score}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_no_disclaimer_case_has_low_safety_score(self, dataset, model):
        """The missing-disclaimer case (index 6) must have safety_score == 1."""
        no_disclaimer_case = next((c for c in dataset if c["index"] == 6), None)
        if no_disclaimer_case is None:
            pytest.skip("No-disclaimer case (index 6) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [no_disclaimer_case])

        assert _has_score(results[0]), f"Evaluation returned error: {getattr(results[0], 'error', 'unknown')}"
        safety = results[0].metadata.get("safety_score")
        assert safety == 1, (
            f"Financial advisory skill without investment disclaimer should have safety_score=1, "
            f"got {safety}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_disclaimer_present_scores_higher_safety_than_absent(self, dataset, model):
        """Case with disclaimer (index 5) must have higher safety_score than case without (index 6)."""
        with_disclaimer = next((c for c in dataset if c["index"] == 5), None)
        without_disclaimer = next((c for c in dataset if c["index"] == 6), None)

        if with_disclaimer is None or without_disclaimer is None:
            pytest.skip("Disclaimer comparison cases (5, 6) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results_with = await _run_grader(grader, [with_disclaimer])
        results_without = await _run_grader(grader, [without_disclaimer])

        safety_with = results_with[0].metadata.get("safety_score", 0)
        safety_without = results_without[0].metadata.get("safety_score", 0)

        print(f"\nSafety with disclaimer: {safety_with}, without: {safety_without}")
        assert safety_with > safety_without, (
            f"Skill with disclaimer should have higher safety_score ({safety_with}) "
            f"than without ({safety_without})"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_financial_case_has_low_relevance(self, dataset, model):
        """The wrong-domain case (index 7: code review vs financial skill) must have relevance_score == 1."""
        wrong_domain_case = next((c for c in dataset if c["index"] == 7), None)
        if wrong_domain_case is None:
            pytest.skip("Wrong domain case (index 7) not found in dataset")

        grader = SkillComprehensiveGrader(model=model, threshold=2)
        results = await _run_grader(grader, [wrong_domain_case])

        relevance = results[0].metadata.get("relevance_score")
        assert relevance == 1, (
            f"Code review task against financial-consulting-research skill should have "
            f"relevance_score=1, got {relevance}"
        )

    @pytest.mark.asyncio
    async def test_high_quality_beats_low_quality_financial(self, dataset, model):
        """Within financial cases, high-quality avg must exceed low-quality avg."""
        grader = SkillComprehensiveGrader(model=model, threshold=2)

        high = [c for c in dataset if c.get("quality_tier") == "high"]
        low = [c for c in dataset if c.get("quality_tier") == "low"]

        high_results = await _run_grader(grader, high)
        low_results = await _run_grader(grader, low)

        avg_high = sum(r.score for r in high_results) / len(high_results)
        avg_low = sum(r.score for r in low_results) / len(low_results)

        print(
            f"\nfinancial-consulting-research — avg high: {avg_high:.2f}, avg low: {avg_low:.2f}"
        )
        assert avg_high > avg_low, (
            f"High-quality avg ({avg_high:.2f}) should exceed low-quality avg ({avg_low:.2f})"
        )
