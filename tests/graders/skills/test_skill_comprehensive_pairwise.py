#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for SkillComprehensivePairwiseGrader.

Covers two test types:

1. Unit tests (offline, with mocks) — validate grader logic and contract in isolation,
   including the ``_compute_ranking`` helper, per-dimension metadata, and weighted scoring.
2. Quality tests (live, requires API keys) — validate ranking quality against
   labeled cases in ``skill_comprehensive_pairwise_cases.json``, covering two skill groups:
   - ``code-review`` (indices 0–3)
   - ``financial-consulting-research`` (indices 4–6)

The pairwise grader evaluates four dimensions in a single LLM call:
    - Relevance    (weight 0.5): which skill more directly addresses the specified task
    - Completeness (weight 0.2): which skill provides more actionable, complete guidance
    - Safety       (weight 0.3): which skill better avoids dangerous operations
    - Structure    (weight 0.1): NEVER list, description quality, content layering

The final ranking is computed from per-dimension verdicts:
    - winner of a dimension earns its full weight; loser earns 0; tie → 0 each
    - rank = [1, 2] if skill_1 total >= skill_2 total, else [2, 1]

Expected winners in the dataset:
    - ``1``    : cases where Skill 1 should be ranked 1st (rank[0] == 1)
    - ``2``    : cases where Skill 2 should be ranked 1st (rank[0] == 2)
    - ``null`` : near-tie, either outcome is acceptable

Example:
    Run all tests::

        pytest tests/graders/skills/test_skill_comprehensive_pairwise.py -v

    Run only unit tests::

        pytest tests/graders/skills/test_skill_comprehensive_pairwise.py -m unit

    Run quality tests (requires OPENAI_API_KEY + OPENAI_BASE_URL)::

        pytest tests/graders/skills/test_skill_comprehensive_pairwise.py -m quality
"""

import json
import os
from pathlib import Path
from typing import List, cast
from unittest.mock import AsyncMock, patch

import pytest

from openjudge.graders.base_grader import GraderRank
from openjudge.graders.skills.comprehensive_pairwise import (
    DEFAULT_DIMENSION_WEIGHTS,
    DimensionComparison,
    SkillComprehensivePairwiseCallback,
    SkillComprehensivePairwiseGrader,
    _compute_ranking,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------

DATA_FILE = Path(__file__).parent / "skill_comprehensive_pairwise_cases.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_comparison(winner: int = 0, reason: str = "reason") -> DimensionComparison:
    """Create a DimensionComparison with the given winner verdict."""
    return DimensionComparison(winner=winner, reason=reason)


def _make_parsed(
    relevance_winner: int = 0,
    completeness_winner: int = 0,
    safety_winner: int = 0,
    structure_winner: int = 0,
    reason: str = "Overall comparison.",
) -> SkillComprehensivePairwiseCallback:
    """Create a SkillComprehensivePairwiseCallback for use in mocked LLM responses."""
    return SkillComprehensivePairwiseCallback(
        relevance_comparison=_make_comparison(relevance_winner, f"Relevance: winner={relevance_winner}"),
        completeness_comparison=_make_comparison(completeness_winner, f"Completeness: winner={completeness_winner}"),
        safety_comparison=_make_comparison(safety_winner, f"Safety: winner={safety_winner}"),
        structure_comparison=_make_comparison(structure_winner, f"Structure: winner={structure_winner}"),
        reason=reason,
    )


_SKILL_1_EXAMPLE = {
    "skill_name": "code-review",
    "skill_description": "Use when reviewing PRs, diffs, or code changes.",
    "skill_md": "---\nname: code-review\ndescription: Review code.\n---\n# NEVER\n- NEVER suggest out-of-scope refactors.\n",
    "scripts": "",
    "allowed_tools": "read_file",
}

_SKILL_2_EXAMPLE = {
    "skill_name": "pr-summarizer",
    "skill_description": "Summarizes pull requests. Use when generating PR descriptions.",
    "skill_md": "---\nname: pr-summarizer\ndescription: Summarizes PRs.\n---\n",
    "scripts": "",
    "allowed_tools": "read_file",
}


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSkillComprehensivePairwiseGraderUnit:
    """Unit tests for SkillComprehensivePairwiseGrader — all external calls are mocked."""

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Grader initialises with sensible defaults."""
        mock_model = AsyncMock()
        grader = SkillComprehensivePairwiseGrader(model=mock_model)

        assert grader.name == "skill_comprehensive_pairwise"
        assert grader.model is mock_model
        assert grader.dimension_weights == DEFAULT_DIMENSION_WEIGHTS

    def test_initialization_custom_dimension_weights(self):
        """Custom dimension weights override defaults; unspecified keys retain defaults."""
        mock_model = AsyncMock()
        grader = SkillComprehensivePairwiseGrader(
            model=mock_model,
            dimension_weights={"relevance": 0.8, "completeness": 0.1},
        )
        assert grader.dimension_weights["relevance"] == 0.8
        assert grader.dimension_weights["completeness"] == 0.1
        assert grader.dimension_weights["safety"] == DEFAULT_DIMENSION_WEIGHTS["safety"]
        assert grader.dimension_weights["structure"] == DEFAULT_DIMENSION_WEIGHTS["structure"]

    # ------------------------------------------------------------------
    # _compute_ranking helper
    # ------------------------------------------------------------------

    def test_compute_ranking_skill1_wins_all(self):
        """Skill 1 wins all dimensions → rank [1, 2] and skill_1 scores > 0."""
        parsed = _make_parsed(1, 1, 1, 1)
        rank, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert rank == [1, 2]
        assert scores["skill_1"] > 0
        assert scores["skill_2"] == 0.0

    def test_compute_ranking_skill2_wins_all(self):
        """Skill 2 wins all dimensions → rank [2, 1] and skill_2 scores > 0."""
        parsed = _make_parsed(2, 2, 2, 2)
        rank, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert rank == [2, 1]
        assert scores["skill_2"] > 0
        assert scores["skill_1"] == 0.0

    def test_compute_ranking_all_tied(self):
        """All dimensions tied → both score 0, rank = [1, 2] (tie goes to skill_1)."""
        parsed = _make_parsed(0, 0, 0, 0)
        rank, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert rank == [1, 2]
        assert scores["skill_1"] == 0.0
        assert scores["skill_2"] == 0.0

    def test_compute_ranking_skill2_wins_only_relevance(self):
        """Skill 2 wins relevance (weight 0.5), Skill 1 wins rest (0.2+0.3+0.1=0.6)
        → Skill 1 total (0.6) > Skill 2 total (0.5) → rank [1, 2]."""
        parsed = _make_parsed(
            relevance_winner=2,
            completeness_winner=1,
            safety_winner=1,
            structure_winner=1,
        )
        rank, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert rank == [1, 2]
        assert scores["skill_1"] == round(0.2 + 0.3 + 0.1, 4)
        assert scores["skill_2"] == round(0.5, 4)

    def test_compute_ranking_skill2_wins_relevance_and_safety(self):
        """Skill 2 wins relevance (0.5) and safety (0.3) = 0.8, Skill 1 wins rest (0.2+0.1=0.3)
        → Skill 2 total (0.8) > Skill 1 total (0.3) → rank [2, 1]."""
        parsed = _make_parsed(
            relevance_winner=2,
            completeness_winner=1,
            safety_winner=2,
            structure_winner=1,
        )
        rank, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert rank == [2, 1]
        assert scores["skill_2"] == round(0.5 + 0.3, 4)
        assert scores["skill_1"] == round(0.2 + 0.1, 4)

    def test_compute_ranking_full_weighted_scores(self):
        """Weighted scores are the sum of earned dimension weights only."""
        # Skill 1 wins relevance (0.5) + completeness (0.2) = 0.7
        # Skill 2 wins safety (0.3) + structure (0.1) = 0.4
        parsed = _make_parsed(
            relevance_winner=1,
            completeness_winner=1,
            safety_winner=2,
            structure_winner=2,
        )
        rank, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        assert rank == [1, 2]
        assert scores["skill_1"] == round(0.5 + 0.2, 4)
        assert scores["skill_2"] == round(0.3 + 0.1, 4)

    def test_compute_ranking_custom_weights_reverses_outcome(self):
        """Custom weights where safety dominates can reverse the outcome."""
        # With safety weight = 0.9, Skill 2 winning safety alone dominates
        parsed = _make_parsed(
            relevance_winner=1,
            completeness_winner=1,
            safety_winner=2,
            structure_winner=1,
        )
        custom_weights = {"relevance": 0.05, "completeness": 0.03, "safety": 0.9, "structure": 0.02}
        rank, scores = _compute_ranking(parsed, custom_weights)
        assert rank == [2, 1]
        assert scores["skill_2"] == 0.9

    def test_compute_ranking_scores_rounded_to_4_decimals(self):
        """Scores are rounded to 4 decimal places."""
        parsed = _make_parsed(1, 2, 1, 2)
        _, scores = _compute_ranking(parsed, DEFAULT_DIMENSION_WEIGHTS)
        for v in scores.values():
            assert round(v, 4) == v

    # ------------------------------------------------------------------
    # LLM response → rank [1, 2]
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_skill1_wins_all_dimensions(self):
        """LLM declaring Skill 1 the winner on all dimensions yields rank [1, 2]."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(1, 1, 1, 1, reason="Skill 1 is clearly better across the board.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a GitHub Pull Request for code quality.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert result.rank == [1, 2]
        assert len(result.reason) > 0

    # ------------------------------------------------------------------
    # LLM response → rank [2, 1]
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_skill2_wins_all_dimensions(self):
        """LLM declaring Skill 2 the winner on all dimensions yields rank [2, 1]."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(2, 2, 2, 2, reason="Skill 2 is superior across all dimensions.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a GitHub PR.",
                skill_1=_SKILL_2_EXAMPLE,
                skill_2=_SKILL_1_EXAMPLE,
            )

        assert result.rank == [2, 1]

    # ------------------------------------------------------------------
    # Tie scenario
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_all_tied_yields_rank_1_2(self):
        """All dimensions tied → both earn 0, tie broken in favour of Skill 1 → rank [1, 2]."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(0, 0, 0, 0, reason="Both skills are equivalent on all dimensions.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Deploy a web app.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert result.rank == [1, 2]

    # ------------------------------------------------------------------
    # Mixed dimension verdicts
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_mixed_verdicts_weighted_correctly(self):
        """Skill 2 wins relevance (0.5) and safety (0.3); Skill 1 wins rest → rank [2, 1]."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(
            relevance_winner=2,
            completeness_winner=1,
            safety_winner=2,
            structure_winner=1,
            reason="Skill 2 is more relevant and safer.",
        )

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert result.rank == [2, 1]
        assert result.metadata["weighted_scores"]["skill_2"] == round(0.5 + 0.3, 4)
        assert result.metadata["weighted_scores"]["skill_1"] == round(0.2 + 0.1, 4)

    # ------------------------------------------------------------------
    # Custom dimension weights
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_custom_weights_shift_ranking(self):
        """When safety weight dominates, Skill 2 winning only safety can flip the rank."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(
            relevance_winner=1,
            completeness_winner=1,
            safety_winner=2,
            structure_winner=1,
            reason="Skill 2 wins safety; Skill 1 wins the rest.",
        )

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            # Safety-dominant weight: skill 2 winning safety (0.9) beats skill 1 rest (0.07)
            grader = SkillComprehensivePairwiseGrader(
                model=mock_model,
                dimension_weights={"relevance": 0.05, "completeness": 0.03, "safety": 0.9, "structure": 0.02},
            )
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert result.rank == [2, 1], (
            "Safety-dominant weights should rank Skill 2 first when it wins safety"
        )

    # ------------------------------------------------------------------
    # Metadata structure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_metadata_contains_all_required_fields(self):
        """GraderRank.metadata contains all expected per-dimension and configuration keys."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(1, 0, 2, 1, reason="Mixed comparison.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review code changes.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        required_keys = {
            "relevance_comparison",
            "completeness_comparison",
            "safety_comparison",
            "structure_comparison",
            "weighted_scores",
            "dimension_weights",
        }
        assert required_keys.issubset(set(result.metadata.keys())), (
            f"Missing metadata keys: {required_keys - set(result.metadata.keys())}"
        )

    @pytest.mark.asyncio
    async def test_per_dimension_comparisons_stored_in_metadata(self):
        """Per-dimension winner verdicts from the LLM are stored correctly in metadata."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(
            relevance_winner=1,
            completeness_winner=2,
            safety_winner=0,
            structure_winner=1,
            reason="Mixed.",
        )

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review code.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert result.metadata["relevance_comparison"]["winner"] == 1
        assert result.metadata["completeness_comparison"]["winner"] == 2
        assert result.metadata["safety_comparison"]["winner"] == 0
        assert result.metadata["structure_comparison"]["winner"] == 1

    @pytest.mark.asyncio
    async def test_dimension_weights_reported_in_metadata(self):
        """dimension_weights in metadata match the grader's configured weights."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(1, 1, 1, 1)

        custom_weights = {"relevance": 0.6, "completeness": 0.2, "safety": 0.15, "structure": 0.05}

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model, dimension_weights=custom_weights)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert result.metadata["dimension_weights"]["relevance"] == 0.6
        assert result.metadata["dimension_weights"]["completeness"] == 0.2
        assert result.metadata["dimension_weights"]["safety"] == 0.15
        assert result.metadata["dimension_weights"]["structure"] == 0.05

    @pytest.mark.asyncio
    async def test_weighted_scores_present_in_metadata(self):
        """metadata['weighted_scores'] contains both skill_1 and skill_2 float values."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(1, 2, 1, 0)

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review code.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        scores = result.metadata["weighted_scores"]
        assert "skill_1" in scores
        assert "skill_2" in scores
        assert isinstance(scores["skill_1"], float)
        assert isinstance(scores["skill_2"], float)

    # ------------------------------------------------------------------
    # Optional parameters
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_missing_optional_skill_fields_default_gracefully(self):
        """Skills with only skill_name are accepted without raising — evaluation completes."""
        mock_response = AsyncMock()
        mock_response.parsed = _make_parsed(0, 0, 0, 0, reason="Both minimal.")

        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_1={"skill_name": "skill-a"},
                skill_2={"skill_name": "skill-b"},
            )

        assert result.rank in ([1, 2], [2, 1])

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_handling_returns_grader_error(self):
        """API errors surface as GraderError (not raised), with the error message captured."""
        with patch(
            "openjudge.graders.llm_grader.BaseChatModel.achat",
            new_callable=AsyncMock,
        ) as mock_achat:
            mock_achat.side_effect = Exception("Simulated API timeout")
            mock_model = AsyncMock()
            grader = SkillComprehensivePairwiseGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Review a PR.",
                skill_1=_SKILL_1_EXAMPLE,
                skill_2=_SKILL_2_EXAMPLE,
            )

        assert hasattr(result, "error")
        assert "Simulated API timeout" in result.error


# ---------------------------------------------------------------------------
# Helpers shared by quality test classes
# ---------------------------------------------------------------------------

_GRADER_MAPPER = {
    "task_description": "task_description",
    "skill_1": "skill_1",
    "skill_2": "skill_2",
}


def _has_rank(r) -> bool:
    """Return True if r is a valid GraderRank (not a GraderError)."""
    return r is not None and hasattr(r, "rank") and r.rank is not None


def _load_dataset(skill_group: str | None = None, expected_winner: int | None = None):
    """Load cases from JSON, optionally filtering by skill_group or expected_winner."""
    if not DATA_FILE.exists():
        pytest.skip(f"Test data file not found: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if skill_group is not None:
        cases = [c for c in cases if c.get("skill_group") == skill_group]
    if expected_winner is not None:
        cases = [c for c in cases if c.get("expected_winner") == expected_winner]
    return cases


async def _run_grader(
    grader: SkillComprehensivePairwiseGrader, cases: list
) -> List[GraderRank]:
    """Run grader over cases via GradingRunner and return results."""
    flat = [{**c["parameters"], "_index": c["index"]} for c in cases]
    runner = GradingRunner(
        grader_configs={
            "skill_comprehensive_pairwise": GraderConfig(
                grader=grader, mapper=_GRADER_MAPPER
            ),
        }
    )
    results = await runner.arun(flat)
    return cast(List[GraderRank], results["skill_comprehensive_pairwise"])


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
class TestSkillComprehensivePairwiseGraderQuality:
    """Quality tests using all 7 labeled cases in skill_comprehensive_pairwise_cases.json.

    The dataset covers two skill groups:
    - ``code-review`` (indices 0–3)
    - ``financial-consulting-research`` (indices 4–6)

    Expected winners:
    - Index 0: Skill 1 (excellent code-review vs minimal)
    - Index 1: Skill 2 (correct domain vs wrong domain)
    - Index 2: Skill 1 (safe vs unsafe with rm -rf)
    - Index 3: null (near-tie, medium quality both)
    - Index 4: Skill 1 (with disclaimer vs without)
    - Index 5: Skill 2 (financial domain vs wrong code-review domain)
    - Index 6: Skill 1 (full financial vs minimal)
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset()

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_all_results_return_valid_rank(self, dataset, model):
        """All 7 evaluations return a valid GraderRank with rank in {[1,2], [2,1]}."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        errors = [r for r in results if not _has_rank(r)]
        assert not errors, (
            f"{len(errors)} evaluation(s) returned GraderError: "
            f"{[getattr(r, 'error', '') for r in errors]}"
        )
        for result in results:
            assert result.rank in ([1, 2], [2, 1]), (
                f"rank must be [1, 2] or [2, 1], got {result.rank}"
            )

    @pytest.mark.asyncio
    async def test_reason_is_non_empty(self, dataset, model):
        """All results include a non-empty reason string."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, dataset)

        for idx, result in enumerate(results):
            if not _has_rank(result):
                pytest.fail(f"Result {idx} is GraderError: {getattr(result, 'error', 'unknown')}")
            assert len(result.reason) > 0, f"Result {idx}: reason is empty"

    @pytest.mark.asyncio
    async def test_per_dimension_comparisons_in_metadata(self, dataset, model):
        """All results include per-dimension comparison dicts with winner (0/1/2) and reason."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, dataset)

        for idx, result in enumerate(results):
            if not _has_rank(result):
                pytest.fail(f"Result {idx} is GraderError: {getattr(result, 'error', 'unknown')}")
            for dim in ("relevance", "completeness", "safety", "structure"):
                key = f"{dim}_comparison"
                assert key in result.metadata, (
                    f"Result {idx}: missing '{key}' in metadata"
                )
                cmp = result.metadata[key]
                assert "winner" in cmp, f"Result {idx}: '{key}' missing 'winner'"
                assert cmp["winner"] in (0, 1, 2), (
                    f"Result {idx}: {key}.winner={cmp['winner']} not in {{0,1,2}}"
                )
                assert "reason" in cmp, f"Result {idx}: '{key}' missing 'reason'"

    @pytest.mark.asyncio
    async def test_weighted_scores_and_dimension_weights_in_metadata(self, dataset, model):
        """metadata contains weighted_scores and dimension_weights for all results."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, dataset[:1])

        assert _has_rank(results[0])
        assert "weighted_scores" in results[0].metadata
        assert "dimension_weights" in results[0].metadata
        assert results[0].metadata["dimension_weights"] == DEFAULT_DIMENSION_WEIGHTS

    @pytest.mark.asyncio
    async def test_expected_winners_for_decisive_cases(self, dataset, model):
        """Cases with a non-null expected_winner must produce the correct rank[0]."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        decisive_cases = [c for c in dataset if c.get("expected_winner") is not None]
        results = await _run_grader(grader, decisive_cases)

        violations = []
        for case, result in zip(decisive_cases, results):
            if not _has_rank(result):
                violations.append(
                    f"Case {case['index']}: GraderError — {getattr(result, 'error', 'unknown')}"
                )
                continue
            expected = case["expected_winner"]
            actual = result.rank[0]
            if actual != expected:
                violations.append(
                    f"Case {case['index']} ({case['description']}): "
                    f"expected Skill {expected} to rank 1st, got rank={result.rank}"
                )

        assert not violations, "Winner prediction violations:\n" + "\n".join(violations)

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self, dataset, model):
        """Same cases run twice should agree on rank[0] for ≥ 75% of decisive cases."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        decisive = [c for c in dataset if c.get("expected_winner") is not None]
        flat = [{**c["parameters"], "_index": c["index"]} for c in decisive]

        runner = GradingRunner(
            grader_configs={
                "run1": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
                "run2": GraderConfig(grader=grader, mapper=_GRADER_MAPPER),
            }
        )
        results = await runner.arun(flat)
        run1 = cast(List[GraderRank], results["run1"])
        run2 = cast(List[GraderRank], results["run2"])

        agreements = sum(
            1
            for r1, r2 in zip(run1, run2)
            if _has_rank(r1) and _has_rank(r2) and r1.rank[0] == r2.rank[0]
        )
        total = sum(1 for r in run1 if _has_rank(r))
        consistency = agreements / total if total > 0 else 1.0

        print(f"\nPairwise rank consistency: {consistency:.2%} ({agreements}/{total})")
        assert consistency >= 0.75, f"Rank consistency too low: {consistency:.2%}"


# ---------------------------------------------------------------------------
# QUALITY TESTS — code-review skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillComprehensivePairwiseCodeReviewGroup:
    """Quality tests restricted to code-review pairwise cases (indices 0–3).

    Cases:
    - Index 0: Skill 1 (excellent) vs Skill 2 (minimal/vague) → rank[0] == 1
    - Index 1: Skill 1 (wrong domain: AWS deploy) vs Skill 2 (proper code-review) → rank[0] == 2
    - Index 2: Skill 1 (safe, read-only) vs Skill 2 (unsafe: rm -rf, auto-fix) → rank[0] == 1
    - Index 3: Near tie (both medium quality) → either rank acceptable
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="code-review")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_excellent_vs_minimal_skill1_wins(self, dataset, model):
        """Excellent code-review skill (index 0) must rank above minimal/vague skill."""
        case = next((c for c in dataset if c["index"] == 0), None)
        if case is None:
            pytest.skip("Case index 0 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank[0] == 1, (
            f"Excellent code-review should rank 1st, got rank={results[0].rank}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_vs_correct_domain_skill2_wins(self, dataset, model):
        """Wrong-domain Skill 1 (AWS deploy) vs proper code-review Skill 2 — Skill 2 must rank 1st."""
        case = next((c for c in dataset if c["index"] == 1), None)
        if case is None:
            pytest.skip("Case index 1 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank[0] == 2, (
            f"Wrong-domain Skill 1 should lose to proper code-review Skill 2, "
            f"got rank={results[0].rank}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_case_relevance_winner_is_skill2(self, dataset, model):
        """In wrong-domain case (index 1), relevance comparison winner must be Skill 2."""
        case = next((c for c in dataset if c["index"] == 1), None)
        if case is None:
            pytest.skip("Case index 1 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0])
        relevance_winner = results[0].metadata["relevance_comparison"]["winner"]
        assert relevance_winner == 2, (
            f"AWS deploy skill vs code-review task: relevance winner should be 2 (proper skill), "
            f"got {relevance_winner}"
        )

    @pytest.mark.asyncio
    async def test_safe_vs_unsafe_skill1_wins(self, dataset, model):
        """Safe code-review (index 2) must rank above unsafe skill with auto-fix and rm -rf."""
        case = next((c for c in dataset if c["index"] == 2), None)
        if case is None:
            pytest.skip("Case index 2 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank[0] == 1, (
            f"Safe read-only code-review should rank above unsafe (rm -rf, auto-fix) skill, "
            f"got rank={results[0].rank}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_unsafe_skill_loses_on_safety_dimension(self, dataset, model):
        """In the safe vs unsafe case (index 2), the safety comparison winner must be Skill 1."""
        case = next((c for c in dataset if c["index"] == 2), None)
        if case is None:
            pytest.skip("Case index 2 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0])
        safety_winner = results[0].metadata["safety_comparison"]["winner"]
        assert safety_winner == 1, (
            f"Safe read-only skill should win the safety dimension vs unsafe rm-rf skill, "
            f"got safety_winner={safety_winner}"
        )

    @pytest.mark.asyncio
    async def test_near_tie_case_produces_valid_rank(self, dataset, model):
        """Near-tie case (index 3) must produce a valid rank without erroring."""
        case = next((c for c in dataset if c["index"] == 3), None)
        if case is None:
            pytest.skip("Case index 3 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank in ([1, 2], [2, 1])


# ---------------------------------------------------------------------------
# QUALITY TESTS — financial-consulting-research skill group
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillComprehensivePairwiseFinancialGroup:
    """Quality tests restricted to financial-consulting-research pairwise cases (indices 4–6).

    Cases:
    - Index 4: Skill 1 (with disclaimer) vs Skill 2 (no disclaimer, gives buy/sell advice) → rank[0] == 1
    - Index 5: Skill 1 (code-review, wrong domain) vs Skill 2 (financial research) → rank[0] == 2
    - Index 6: Skill 1 (full financial skill) vs Skill 2 (minimal/vague) → rank[0] == 1
    """

    @pytest.fixture
    def dataset(self):
        return _load_dataset(skill_group="financial-consulting-research")

    @pytest.fixture
    def model(self):
        return _make_model()

    @pytest.mark.asyncio
    async def test_disclaimer_vs_no_disclaimer_skill1_wins(self, dataset, model):
        """Skill 1 with investment disclaimer (index 4) must rank above Skill 2 without disclaimer."""
        case = next((c for c in dataset if c["index"] == 4), None)
        if case is None:
            pytest.skip("Case index 4 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank[0] == 1, (
            f"Financial skill with disclaimer should rank above no-disclaimer skill, "
            f"got rank={results[0].rank}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_no_disclaimer_skill_loses_safety_dimension(self, dataset, model):
        """In the disclaimer comparison (index 4), Skill 2 (no disclaimer) must lose the safety dimension."""
        case = next((c for c in dataset if c["index"] == 4), None)
        if case is None:
            pytest.skip("Case index 4 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0])
        safety_winner = results[0].metadata["safety_comparison"]["winner"]
        assert safety_winner == 1, (
            f"Skill with explicit investment disclaimer should win the safety dimension, "
            f"got safety_winner={safety_winner}"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_vs_financial_skill2_wins(self, dataset, model):
        """Wrong-domain Skill 1 (code-review) vs financial research Skill 2 (index 5) — Skill 2 wins."""
        case = next((c for c in dataset if c["index"] == 5), None)
        if case is None:
            pytest.skip("Case index 5 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank[0] == 2, (
            f"Code-review skill should lose to financial research skill on a financial task, "
            f"got rank={results[0].rank}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_wrong_domain_financial_relevance_winner_is_skill2(self, dataset, model):
        """In the cross-domain case (index 5), relevance comparison winner must be Skill 2."""
        case = next((c for c in dataset if c["index"] == 5), None)
        if case is None:
            pytest.skip("Case index 5 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0])
        relevance_winner = results[0].metadata["relevance_comparison"]["winner"]
        assert relevance_winner == 2, (
            f"Code-review skill vs financial research task: relevance winner should be Skill 2, "
            f"got {relevance_winner}"
        )

    @pytest.mark.asyncio
    async def test_full_vs_minimal_financial_skill1_wins(self, dataset, model):
        """Full-featured financial skill (index 6) must rank above minimal/vague financial skill."""
        case = next((c for c in dataset if c["index"] == 6), None)
        if case is None:
            pytest.skip("Case index 6 not found in dataset")

        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, [case])

        assert _has_rank(results[0]), f"GraderError: {getattr(results[0], 'error', 'unknown')}"
        assert results[0].rank[0] == 1, (
            f"Full financial skill should rank above minimal/vague skill, "
            f"got rank={results[0].rank}: {results[0].reason}"
        )

    @pytest.mark.asyncio
    async def test_all_financial_cases_return_valid_rank(self, dataset, model):
        """All financial-consulting-research cases return valid GraderRank without error."""
        grader = SkillComprehensivePairwiseGrader(model=model)
        results = await _run_grader(grader, dataset)

        assert len(results) == len(dataset)
        for idx, result in enumerate(results):
            assert _has_rank(result), (
                f"Case {dataset[idx]['index']}: GraderError — {getattr(result, 'error', 'unknown')}"
            )
            assert result.rank in ([1, 2], [2, 1])
