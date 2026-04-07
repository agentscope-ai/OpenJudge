# -*- coding: utf-8 -*-
"""
Skills Grading Runner

Orchestrates multi-dimensional evaluation of Agent Skill packages loaded from a
directory. Combines five grader dimensions (threat analysis, alignment, completeness,
relevance, structure) into a single weighted aggregate score per skill.

Typical usage::

    import asyncio
    from openjudge.models.openai_chat_model import OpenAIChatModel
    from cookbooks.skills_evaluation.runner import SkillsGradingRunner

    model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
    runner = SkillsGradingRunner(
        model=model,
        weights={"threat_analysis": 2.0, "structure": 0.5},
    )
    results = asyncio.run(runner.arun("/path/to/skills", task_description="..."))
    for r in results:
        print(r.skill_name, r.weighted_score, "PASS" if r.passed else "FAIL")
"""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from cookbooks.skills_evaluation.skill_models import SkillLoader, SkillPackage
from openjudge.graders.base_grader import GraderError, GraderScore
from openjudge.graders.skills.completeness import SkillCompletenessGrader
from openjudge.graders.skills.declaration_alignment import (
    SkillDeclarationAlignmentGrader,
)
from openjudge.graders.skills.design import SkillDesignGrader
from openjudge.graders.skills.relevance import SkillRelevanceGrader
from openjudge.graders.skills.threat_analysis import SkillThreatAnalysisGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.prompt_template import LanguageEnum
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# pylint: disable=line-too-long

# ── Grading result models ──────────────────────────────────────────────────────


@dataclass
class DimensionScore:
    """Score produced by a single grader dimension.

    Attributes:
        name: Short dimension name (``threat_analysis``, ``alignment``, etc.).
        score: Raw score on the grader's native scale.
        normalized_score: Score normalised to ``[0, 1]``.
        weight: Weight assigned to this dimension in the final aggregate.
        reason: Human-readable explanation from the LLM.
        passed: Whether ``score >= threshold``.
        metadata: Extra grader metadata (findings, threshold, etc.).
        error: Error message if evaluation failed, otherwise ``None``.
    """

    name: str
    score: float
    normalized_score: float
    weight: float
    reason: str
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def weighted_contribution(self) -> float:
        """Weight × normalised score, used when computing the aggregate."""
        return self.weight * self.normalized_score


@dataclass
class SkillGradingResult:
    """Complete grading result for a single :class:`SkillPackage`.

    Attributes:
        skill_name: Name from the skill manifest.
        skill_path: Absolute path to the skill directory.
        dimension_scores: Mapping from dimension name to :class:`DimensionScore`.
        weighted_score: Final weighted aggregate score in ``[0, 1]``.
        passed: ``True`` if every successful dimension score is at or above its threshold.
        errors: List of error messages from failed dimensions.
        grading_duration_seconds: Wall-clock time for the entire grading run.
    """

    skill_name: str
    skill_path: str
    dimension_scores: Dict[str, DimensionScore] = field(default_factory=dict)
    weighted_score: float = 0.0
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    grading_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary suitable for JSON output."""
        return {
            "skill_name": self.skill_name,
            "skill_path": self.skill_path,
            "weighted_score": round(self.weighted_score, 4),
            "passed": self.passed,
            "grading_duration_seconds": round(self.grading_duration_seconds, 3),
            "dimensions": {
                name: {
                    "score": d.score,
                    "normalized_score": round(d.normalized_score, 4),
                    "weight": d.weight,
                    "reason": d.reason,
                    "passed": d.passed,
                    "error": d.error,
                    "metadata": d.metadata,
                }
                for name, d in self.dimension_scores.items()
            },
            "errors": self.errors,
        }

    def to_markdown(self) -> str:
        """Render the grading result as a Markdown report.

        Returns a self-contained Markdown string suitable for writing to a
        ``.md`` file or embedding in a notebook cell.
        """
        verdict = "✅ PASS" if self.passed else "❌ FAIL"
        pct = self.weighted_score * 100
        lines: List[str] = []

        lines += [
            f"# Skill Evaluation Report: `{self.skill_name}`",
            "",
            f"> **Overall score: {pct:.1f} / 100 — {verdict}**  "
            f"_(evaluated in {self.grading_duration_seconds:.1f}s)_",
            "",
            f"**Path:** `{self.skill_path}`",
            "",
        ]

        # ── Dimension summary table ────────────────────────────────────────
        lines += [
            "## Dimension Summary",
            "",
            "| Dimension | Score | Normalised | Weight | Result |",
            "|-----------|------:|-----------:|-------:|--------|",
        ]
        _dim_labels = {
            "threat_analysis": "Threat Analysis",
            "alignment": "Alignment",
            "completeness": "Completeness",
            "relevance": "Relevance",
            "structure": "Structure",
        }
        for dim_name, d in self.dimension_scores.items():
            label = _dim_labels.get(dim_name, dim_name.replace("_", " ").title())
            if d.error:
                lines.append(f"| {label} | — | — | {d.weight:.1f} | ⚠️ Error |")
            else:
                status = "✅ Pass" if d.passed else "❌ Fail"
                lines.append(f"| {label} | {d.score:.0f} | {d.normalized_score:.2f}" f" | {d.weight:.1f} | {status} |")
        lines.append("")

        # ── Per-dimension detail ───────────────────────────────────────────
        lines.append("## Dimension Details")
        lines.append("")
        for dim_name, d in self.dimension_scores.items():
            label = _dim_labels.get(dim_name, dim_name.replace("_", " ").title())
            lines.append(f"### {label}")
            lines.append("")
            if d.error:
                lines += [f"> ⚠️ **Evaluation error:** {d.error}", ""]
                continue
            status = "✅ Pass" if d.passed else "❌ Fail"
            lines += [
                f"- **Score:** {d.score:.0f}  |  "
                f"**Normalised:** {d.normalized_score:.2f}  |  "
                f"**Weight:** {d.weight:.1f}  |  "
                f"**Result:** {status}",
                "",
            ]
            if d.reason:
                lines += [f"{d.reason}", ""]

        # ── Errors section (if any) ────────────────────────────────────────
        if self.errors:
            lines += ["## Errors", ""]
            for err in self.errors:
                lines.append(f"- {err}")
            lines.append("")

        return "\n".join(lines)


# ── Score normalisation ────────────────────────────────────────────────────────

# Native score ranges for each grader (lo, hi).
_SCORE_RANGES: Dict[str, tuple[float, float]] = {
    "skill_completeness": (1.0, 3.0),
    "skill_relevance": (1.0, 3.0),
    "skill_structure": (1.0, 3.0),
    "skill_alignment": (1.0, 3.0),
    "skill_threat_analysis": (1.0, 4.0),
}


def _normalize_score(grader_name: str, score: float) -> float:
    lo, hi = _SCORE_RANGES.get(grader_name, (1.0, 3.0))
    if hi == lo:
        return 1.0
    return max(0.0, min(1.0, (score - lo) / (hi - lo)))


# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "threat_analysis": 1.0,
    "alignment": 1.0,
    "completeness": 1.0,
    "relevance": 1.0,
    "structure": 1.0,
}

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "threat_analysis": 3.0,  # [1, 4]: LOW severity or better → pass
    "alignment": 2.0,  # [1, 3]: Uncertain or better → pass
    "completeness": 2.0,  # [1, 3]: Partially complete or better → pass
    "relevance": 2.0,  # [1, 3]: Partial match or better → pass
    "structure": 2.0,  # [1, 3]: Partially sound or better → pass
}


# ── SkillsGradingRunner ────────────────────────────────────────────────────────


class SkillsGradingRunner(GradingRunner):
    """Orchestrates multi-dimensional evaluation of Agent Skill packages.

    Loads skills from a directory, runs each enabled grader dimension in parallel,
    computes a weighted aggregate score, and returns a :class:`SkillGradingResult`
    for every loaded skill.

    Inherits from :class:`~openjudge.runner.grading_runner.GradingRunner` and
    overrides :meth:`arun` to accept a skills directory instead of a flat dataset.

    Dimensions
    ----------
    - **threat_analysis** (scale 1–4): LLM threat scanner using the AITech taxonomy.
    - **alignment** (scale 1–3): Detects mismatches between SKILL.md and script behaviour.
      For multi-script skills the worst per-script score is used.
    - **completeness** (scale 1–3): Whether the skill provides enough detail to act on.
    - **relevance** (scale 1–3): How well the skill matches a task description.
    - **structure** (scale 1–3): Structural design quality (NEVER list, description, etc.).

    All raw scores are normalised to ``[0, 1]`` before weighting.

    Args:
        model: :class:`~openjudge.models.base_chat_model.BaseChatModel` instance or a
            dict config that will be forwarded to ``OpenAIChatModel``.
        weights: Per-dimension weights.  Keys: ``"threat_analysis"``, ``"alignment"``,
            ``"completeness"``, ``"relevance"``, ``"structure"``.  Defaults to ``1.0``
            for every dimension.  Set a weight to ``0.0`` to disable that dimension.
        thresholds: Per-dimension pass/fail thresholds.  Uses the default threshold for
            each grader's scale if not overridden.
        language: Prompt language for all graders.  Defaults to
            :attr:`~openjudge.models.schema.prompt_template.LanguageEnum.EN`.
        concurrency: Maximum number of grader coroutines running concurrently per skill.
            Defaults to ``5`` (all dimensions in parallel).

    Example::

        import asyncio
        from openjudge.models.openai_chat_model import OpenAIChatModel
        from cookbooks.skills_evaluation.runner import SkillsGradingRunner

        model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        runner = SkillsGradingRunner(
            model=model,
            weights={"threat_analysis": 2.0, "alignment": 1.5, "structure": 0.5},
        )
        results = asyncio.run(runner.arun(
            "/path/to/skills",
            task_description="Automate code review for pull requests.",
        ))
        for r in results:
            print(f"{r.skill_name}: {r.weighted_score:.3f} ({'PASS' if r.passed else 'FAIL'})")

    See also the accompanying ``evaluate_skills.ipynb`` notebook in this directory
    for an interactive walkthrough.
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        language: LanguageEnum = LanguageEnum.EN,
        concurrency: int = 5,
    ) -> None:
        self.model = model
        self.weights: Dict[str, float] = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.thresholds: Dict[str, float] = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.language = language
        super().__init__(
            grader_configs=self._build_grader_configs(),
            max_concurrency=concurrency,
            show_progress=False,
        )

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _build_grader_configs(self) -> dict:
        """Instantiate graders for all dimensions with weight > 0 and wrap in GraderConfig."""
        configs: Dict[str, GraderConfig] = {}
        if self.weights.get("threat_analysis", 0) > 0:
            configs["threat_analysis"] = GraderConfig(
                grader=SkillThreatAnalysisGrader(
                    model=self.model,
                    threshold=self.thresholds["threat_analysis"],
                    language=self.language,
                )
            )
        if self.weights.get("alignment", 0) > 0:
            configs["alignment"] = GraderConfig(
                grader=SkillDeclarationAlignmentGrader(
                    model=self.model,
                    threshold=self.thresholds["alignment"],
                    language=self.language,
                )
            )
        if self.weights.get("completeness", 0) > 0:
            configs["completeness"] = GraderConfig(
                grader=SkillCompletenessGrader(
                    model=self.model,
                    threshold=self.thresholds["completeness"],
                    language=self.language,
                )
            )
        if self.weights.get("relevance", 0) > 0:
            configs["relevance"] = GraderConfig(
                grader=SkillRelevanceGrader(
                    model=self.model,
                    threshold=self.thresholds["relevance"],
                    language=self.language,
                )
            )
        if self.weights.get("structure", 0) > 0:
            configs["structure"] = GraderConfig(
                grader=SkillDesignGrader(
                    model=self.model,
                    threshold=self.thresholds["structure"],
                    language=self.language,
                )
            )
        return configs

    # ── Per-dimension grading methods ──────────────────────────────────────────

    async def _grade_threat_analysis(self, skill: SkillPackage) -> DimensionScore:
        grader = self.grader_configs["threat_analysis"].grader
        result = await grader.aevaluate(
            skill_name=skill.name,
            skill_manifest=skill.manifest.raw_yaml,
            instruction_body=skill.instruction_body,
            script_contents=skill.script_contents,
            reference_contents=skill.reference_contents,
        )
        return self._to_dimension_score("threat_analysis", result, grader.name)

    async def _grade_alignment(self, skill: SkillPackage) -> DimensionScore:
        grader = self.grader_configs["alignment"].grader

        if not skill.get_scripts():
            return DimensionScore(
                name="alignment",
                score=3.0,
                normalized_score=1.0,
                weight=self.weights.get("alignment", 1.0),
                reason="No scripts found; alignment check not applicable.",
                passed=True,
                metadata={"skipped": "no_scripts"},
            )

        result = await grader.aevaluate(
            skill_name=skill.name,
            skill_manifest=skill.manifest.raw_yaml,
            instruction_body=skill.instruction_body,
            script_contents=skill.script_contents,
            reference_contents=skill.reference_contents,
        )
        return self._to_dimension_score("alignment", result, grader.name)

    async def _grade_completeness(self, skill: SkillPackage, task_description: Optional[str] = None) -> DimensionScore:
        grader = self.grader_configs["completeness"].grader
        task_desc = task_description or skill.description
        result = await grader.aevaluate(
            task_description=task_desc,
            skill_name=skill.name,
            skill_manifest=skill.manifest.raw_yaml,
            instruction_body=skill.instruction_body,
            script_contents=skill.script_contents,
            reference_contents=skill.reference_contents,
        )
        return self._to_dimension_score("completeness", result, grader.name)

    async def _grade_relevance(self, skill: SkillPackage, task_description: Optional[str] = None) -> DimensionScore:
        grader = self.grader_configs["relevance"].grader
        task_desc = task_description or skill.description
        result = await grader.aevaluate(
            task_description=task_desc,
            skill_name=skill.name,
            skill_manifest=skill.manifest.raw_yaml,
            instruction_body=skill.instruction_body,
            script_contents=skill.script_contents,
            reference_contents=skill.reference_contents,
        )
        return self._to_dimension_score("relevance", result, grader.name)

    async def _grade_structure(self, skill: SkillPackage) -> DimensionScore:
        grader = self.grader_configs["structure"].grader
        result = await grader.aevaluate(
            skill_name=skill.name,
            skill_manifest=skill.manifest.raw_yaml,
            instruction_body=skill.instruction_body,
            script_contents=skill.script_contents,
            reference_contents=skill.reference_contents,
        )
        return self._to_dimension_score("structure", result, grader.name)

    # ── Score helpers ──────────────────────────────────────────────────────────

    def _to_dimension_score(
        self,
        dimension: str,
        result: Union[GraderScore, GraderError],
        grader_name: str,
    ) -> DimensionScore:
        weight = self.weights.get(dimension, 1.0)
        threshold = self.thresholds.get(dimension, 2.0)

        if isinstance(result, GraderError):
            return DimensionScore(
                name=dimension,
                score=0.0,
                normalized_score=0.0,
                weight=weight,
                reason="",
                passed=False,
                metadata={},
                error=result.error,
            )

        normalized = _normalize_score(grader_name, result.score)
        return DimensionScore(
            name=dimension,
            score=result.score,
            normalized_score=normalized,
            weight=weight,
            reason=result.reason,
            passed=result.score >= threshold,
            metadata=result.metadata,
        )

    def _compute_weighted_score(self, dimension_scores: Dict[str, DimensionScore]) -> float:
        """Compute the weighted average of successful dimension scores."""
        total_weight = sum(d.weight for d in dimension_scores.values() if d.error is None)
        if total_weight == 0.0:
            return 0.0
        weighted_sum = sum(d.weighted_contribution for d in dimension_scores.values() if d.error is None)
        return weighted_sum / total_weight

    # ── Public API ─────────────────────────────────────────────────────────────

    async def agrade_skill(self, skill: SkillPackage, task_description: Optional[str] = None) -> SkillGradingResult:
        """Grade a single :class:`SkillPackage` across all enabled dimensions.

        Dimensions are evaluated concurrently (bounded by *concurrency*).

        Args:
            skill: The skill package to grade.
            task_description: Optional task description supplied to the completeness and
                relevance graders.  When omitted, each skill's own ``description`` field
                is used as a proxy (self-consistency check).

        Returns:
            :class:`SkillGradingResult` with per-dimension scores and the weighted
            aggregate.
        """
        t0 = time.monotonic()
        dimension_scores: Dict[str, DimensionScore] = {}
        errors: List[str] = []

        _dispatch: Dict[str, Any] = {
            "threat_analysis": lambda s: self._grade_threat_analysis(s),
            "alignment": lambda s: self._grade_alignment(s),
            "completeness": lambda s: self._grade_completeness(s, task_description),
            "relevance": lambda s: self._grade_relevance(s, task_description),
            "structure": lambda s: self._grade_structure(s),
        }

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _run(dim: str) -> tuple[str, DimensionScore]:
            async with sem:
                try:
                    return dim, await _dispatch[dim](skill)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(f"Unexpected error grading {skill.name}[{dim}]: {exc}")
                    return dim, DimensionScore(
                        name=dim,
                        score=0.0,
                        normalized_score=0.0,
                        weight=self.weights.get(dim, 1.0),
                        reason="",
                        passed=False,
                        error=str(exc),
                    )

        active_dims = [d for d in _dispatch if d in self.grader_configs]
        gathered = await asyncio.gather(*[_run(d) for d in active_dims])

        for dim_name, dim_score in gathered:
            dimension_scores[dim_name] = dim_score
            if dim_score.error:
                errors.append(f"{dim_name}: {dim_score.error}")

        weighted_score = self._compute_weighted_score(dimension_scores)
        passed = all(d.passed for d in dimension_scores.values() if d.error is None)

        return SkillGradingResult(
            skill_name=skill.name,
            skill_path=str(skill.directory),
            dimension_scores=dimension_scores,
            weighted_score=weighted_score,
            passed=passed,
            errors=errors,
            grading_duration_seconds=time.monotonic() - t0,
        )

    async def arun(  # type: ignore[override]
        self,
        skills_dir: Union[str, Path],
        task_description: Optional[str] = None,
    ) -> List[SkillGradingResult]:
        """Load all skills from *skills_dir* and grade each one.

        Args:
            skills_dir: Path to a directory containing one or more skill packages.
                Both single-skill and multi-skill registry layouts are supported
                (see :class:`SkillLoader`).
            task_description: Optional task description supplied to the completeness and
                relevance graders.  When omitted, each skill's own ``description`` field
                is used as a proxy (self-consistency check).

        Returns:
            List of :class:`SkillGradingResult`, one per successfully loaded skill,
            in the order they were loaded.

        Raises:
            ValueError: If *skills_dir* does not exist or is not a directory.
        """
        skills = SkillLoader.load_from_directory(skills_dir)
        if not skills:
            logger.warning(f"No skills found in {skills_dir}")
            return []

        logger.info(f"Loaded {len(skills)} skill(s) from {skills_dir}")
        results: List[SkillGradingResult] = []
        for skill in skills:
            logger.info(f"Grading skill: {skill.name}")
            result = await self.agrade_skill(skill, task_description=task_description)
            results.append(result)

        return results


def build_markdown_report(results: List[SkillGradingResult]) -> str:
    """Build a combined Markdown report for multiple skills.

    The report contains:

    1. A top-level summary table (one row per skill).
    2. Individual per-skill sections generated by :meth:`SkillGradingResult.to_markdown`.

    Args:
        results: List of :class:`SkillGradingResult` objects from
            :meth:`SkillsGradingRunner.arun`.

    Returns:
        A single Markdown string covering all skills.
    """
    if not results:
        return "# Skills Evaluation Report\n\n_No skills evaluated._\n"

    lines: List[str] = [
        "# Skills Evaluation Report",
        "",
        f"_Total skills evaluated: **{len(results)}** — "
        f"Passed: **{sum(1 for r in results if r.passed)}** / {len(results)}_",
        "",
        "## Summary",
        "",
        "| Skill | Score | Result |",
        "|-------|------:|--------|",
    ]
    for r in results:
        verdict = "✅ Pass" if r.passed else "❌ Fail"
        lines.append(f"| `{r.skill_name}` | {r.weighted_score * 100:.1f} | {verdict} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for r in results:
        lines.append(r.to_markdown())
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "DimensionScore",
    "SkillGradingResult",
    "SkillsGradingRunner",
    "build_markdown_report",
    "DEFAULT_WEIGHTS",
    "DEFAULT_THRESHOLDS",
]
