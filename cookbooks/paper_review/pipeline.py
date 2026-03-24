# -*- coding: utf-8 -*-
"""End-to-end paper review pipeline."""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from cookbooks.paper_review.disciplines import DisciplineConfig, get_discipline
from cookbooks.paper_review.graders import (
    CorrectnessGrader,
    CriticalityGrader,
    FormatGrader,
    JailbreakingGrader,
    RebuttalAssessmentGrader,
    RebuttalGenerationGrader,
    ReviewGrader,
)
from cookbooks.paper_review.processors import BibChecker, TexPackageProcessor
from cookbooks.paper_review.report import generate_report
from cookbooks.paper_review.schema import (
    BibVerificationSummary,
    CorrectnessResult,
    CriticalityResult,
    PaperReviewResult,
    ProgressCallback,
    RebuttalAssessmentResult,
    RebuttalConcern,
    RebuttalPointAssessment,
    RebuttalResult,
    ReviewProgress,
    ReviewResult,
    ReviewStage,
    TexPackageInfo,
)
from cookbooks.paper_review.utils import encode_pdf_base64, load_pdf_bytes
from openjudge.graders.schema import GraderError


@dataclass
class PipelineConfig:
    """Configuration for the paper review pipeline."""

    model_name: str = "gpt-4o"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    timeout: int = 7500  # 扩大5倍：支持更长论文
    enable_safety_checks: bool = True
    enable_correctness: bool = True
    enable_review: bool = True
    enable_criticality: bool = True
    enable_bib_verification: bool = True
    crossref_mailto: Optional[str] = None
    progress_callback: Optional[ProgressCallback] = field(default=None)
    # ── Discipline and venue ──────────────────────────────────────────────────
    # discipline: a discipline ID string (e.g. "cs", "medicine", "physics") or
    #             a DisciplineConfig instance for full customization, or None
    #             to use the original general-purpose CS/ML-oriented prompts.
    discipline: Optional[Union[str, DisciplineConfig]] = None
    # venue: a specific conference/journal name the user wants to target
    #        (e.g. "NeurIPS 2025", "The Lancet", "CVPR 2026").
    #        The AI will apply that venue's standards on top of discipline criteria.
    #        Users may pass any custom string — it appears verbatim in the prompt.
    venue: Optional[str] = None
    # instructions: optional free-form reviewer instructions provided by the user
    #        (e.g. "Focus on experimental design", "This is a short paper").
    #        Rendered as a dedicated section in the review system prompt, separate
    #        from the venue block.
    instructions: Optional[str] = None
    # language: output language for the review text. Supported values:
    #        "en" (default) — English
    #        "zh"           — Simplified Chinese (简体中文)
    language: Optional[str] = None
    # use_vision_for_pdf: when True, PDF pages are rendered to images and sent
    #   as image_url blocks instead of extracting text.  Suitable for
    #   multi-modal DashScope models such as qwen3.5-plus that support
    #   image_url but not the OpenAI-style type:'file' content block.
    #   Requires pypdfium2 (pip install pypdfium2).
    use_vision_for_pdf: bool = False
    # vision_max_pages: maximum number of pages to send when use_vision_for_pdf
    #   is True.  Only the first N pages are rendered; the rest are dropped.
    #   Set to None to send all pages (may exceed API request size limits for
    #   long papers).  Default 30 covers most paper bodies while staying within
    #   typical API payload limits (~8 MB).
    vision_max_pages: Optional[int] = 30
    # format_vision_max_pages: page limit used exclusively by the Format grader.
    #   Format checking only needs the first few pages (title, abstract,
    #   section headers) so a smaller limit keeps the request size small and
    #   avoids connection errors on long papers.  Defaults to 10.
    #   Set to None to fall back to vision_max_pages.
    format_vision_max_pages: Optional[int] = 10
    # ── Rebuttal ─────────────────────────────────────────────────────────────
    enable_rebuttal_generation: bool = False
    enable_rebuttal_assessment: bool = False


class PaperReviewPipeline:
    """End-to-end paper review pipeline using OpenJudge."""

    # Stage i18n keys for progress reporting - UI should translate these keys
    # Format: (stage_name_key, stage_description_key)
    STAGE_I18N_KEYS = {
        ReviewStage.LOADING_PDF: ("paper_review.progress.loading_pdf", "paper_review.progress.loading_pdf_desc"),
        ReviewStage.SAFETY_CHECK: ("paper_review.progress.safety_check", "paper_review.progress.safety_check_desc"),
        ReviewStage.CORRECTNESS: ("paper_review.progress.correctness", "paper_review.progress.correctness_desc"),
        ReviewStage.REVIEW: ("paper_review.progress.review", "paper_review.progress.review_desc"),
        ReviewStage.CRITICALITY: ("paper_review.progress.criticality", "paper_review.progress.criticality_desc"),
        ReviewStage.BIB_VERIFICATION: (
            "paper_review.progress.bib_verification",
            "paper_review.progress.bib_verification_desc",
        ),
        ReviewStage.REBUTTAL_GENERATION: (
            "paper_review.progress.rebuttal_generation",
            "paper_review.progress.rebuttal_generation_desc",
        ),
        ReviewStage.REBUTTAL_ASSESSMENT: (
            "paper_review.progress.rebuttal_assessment",
            "paper_review.progress.rebuttal_assessment_desc",
        ),
        ReviewStage.COMPLETED: ("paper_review.progress.completed", "paper_review.progress.completed_desc"),
        ReviewStage.FAILED: ("paper_review.progress.failed", "paper_review.progress.failed_desc"),
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._progress_callback = config.progress_callback
        self._progress = ReviewProgress()

        # Resolve discipline (string ID → DisciplineConfig or None)
        self._discipline = get_discipline(config.discipline)
        self._venue = config.venue
        self._instructions = config.instructions
        self._language = config.language

        # Use LiteLLM for native PDF support
        from cookbooks.paper_review.models import LiteLLMModel

        self.model = LiteLLMModel(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            timeout=config.timeout,
            use_vision_for_pdf=config.use_vision_for_pdf,
            vision_max_pages=config.vision_max_pages,
        )
        self._init_graders()

    def _init_graders(self):
        """Initialize all graders with discipline and venue context."""
        self.correctness_grader = CorrectnessGrader(self.model, discipline=self._discipline)
        self.review_grader = ReviewGrader(
            self.model,
            discipline=self._discipline,
            venue=self._venue,
            instructions=self._instructions,
            language=self._language,
        )
        self.criticality_grader = CriticalityGrader(self.model)
        self.format_grader = FormatGrader(self.model)
        self.jailbreaking_grader = JailbreakingGrader(self.model)
        self.rebuttal_generation_grader = RebuttalGenerationGrader(
            self.model,
            discipline=self._discipline,
            venue=self._venue,
            language=self._language,
        )
        self.rebuttal_assessment_grader = RebuttalAssessmentGrader(
            self.model,
            discipline=self._discipline,
            language=self._language,
        )

    def _count_enabled_stages(self, has_bib: bool = False) -> int:
        """Count total enabled stages for progress tracking."""
        count = 1  # Loading PDF is always counted
        if self.config.enable_safety_checks:
            count += 1
        if self.config.enable_correctness:
            count += 1
        if self.config.enable_review:
            count += 1
        if self.config.enable_criticality:
            count += 1  # May be skipped but we count it
        if self.config.enable_bib_verification and has_bib:
            count += 1
        return count

    def _notify_progress(
        self,
        stage: ReviewStage,
        completed_stages: int,
        total_stages: int,
    ) -> None:
        """Notify progress callback if set."""
        if self._progress_callback:
            # Get i18n keys for this stage (UI will translate)
            i18n_keys = self.STAGE_I18N_KEYS.get(stage, ("", ""))
            self._progress.update(
                stage=stage,
                stage_name=i18n_keys[0],  # i18n key for stage name
                stage_description=i18n_keys[1],  # i18n key for description
                completed_stages=completed_stages,
                total_stages=total_stages,
            )
            self._progress_callback(self._progress)

    def _notify_completed(self) -> None:
        """Notify progress callback that review is completed."""
        if self._progress_callback:
            self._progress.mark_completed()
            self._progress_callback(self._progress)

    def _notify_failed(self, error: str) -> None:
        """Notify progress callback that review has failed."""
        if self._progress_callback:
            self._progress.mark_failed(error)
            self._progress_callback(self._progress)

    async def review_paper(
        self,
        pdf_input: Union[str, Path, bytes],
        bib_path: Optional[str] = None,
        bib_content: Optional[str] = None,
    ) -> PaperReviewResult:
        """Review a single paper.

        Args:
            pdf_input: Path to PDF file or PDF bytes
            bib_path: Optional path to .bib file for reference verification
            bib_content: Optional .bib file content string for reference verification

        Returns:
            PaperReviewResult with all evaluation results
        """
        has_bib = bool(bib_path or bib_content)
        total_stages = self._count_enabled_stages(has_bib)
        completed_stages = 0

        # Reset progress state for new review
        self._progress.reset(total_stages)

        try:
            # Stage: Loading PDF
            self._notify_progress(ReviewStage.LOADING_PDF, completed_stages, total_stages)
            logger.info("Loading and encoding PDF...")

            if isinstance(pdf_input, bytes):
                pdf_bytes = pdf_input
            else:
                pdf_bytes = load_pdf_bytes(pdf_input)
            pdf_data = encode_pdf_base64(pdf_bytes)

            # Pre-render PDF pages once (single-threaded) so all parallel grader
            # threads hit the cache instead of racing to render simultaneously.
            if self.config.use_vision_for_pdf:
                self.model.warmup_vision_cache(pdf_data)
                # Pre-render the smaller Format-grader slice if it differs.
                fmt_limit = self.config.format_vision_max_pages
                if fmt_limit != self.config.vision_max_pages:
                    self.model.warmup_vision_cache(pdf_data, max_pages=fmt_limit)

            completed_stages += 1

            result = PaperReviewResult()

            # Stage: Safety checks
            if self.config.enable_safety_checks:
                self._notify_progress(ReviewStage.SAFETY_CHECK, completed_stages, total_stages)
                logger.info("Running safety checks...")
                safety_result = await self._run_safety_checks(pdf_data)
                completed_stages += 1

                if not safety_result["is_safe"]:
                    result.is_safe = False
                    result.safety_issues = safety_result["issues"]
                    # Notify failure using dedicated method
                    self._notify_failed("Safety check failed: " + "; ".join(safety_result["issues"]))
                    return result
                result.format_compliant = safety_result["format_ok"]

            # Stage: Correctness + Review（并行，互相无依赖）
            parallel_coros: Dict[str, Any] = {}
            if self.config.enable_correctness:
                self._notify_progress(ReviewStage.CORRECTNESS, completed_stages, total_stages)
                parallel_coros["correctness"] = self.correctness_grader.aevaluate(pdf_data=pdf_data)
            if self.config.enable_review:
                self._notify_progress(ReviewStage.REVIEW, completed_stages, total_stages)
                parallel_coros["review"] = self.review_grader.aevaluate(pdf_data=pdf_data)

            if parallel_coros:
                logger.info(f"Running {', '.join(parallel_coros)} in parallel...")
                parallel_results = await asyncio.gather(*parallel_coros.values())
                parallel_map = dict(zip(parallel_coros.keys(), parallel_results))

                if "correctness" in parallel_map:
                    correctness = parallel_map["correctness"]
                    if isinstance(correctness, GraderError):
                        logger.error(f"Correctness grader error: {correctness.error}")
                    else:
                        result.correctness = CorrectnessResult(
                            score=correctness.score,
                            reasoning=correctness.reason,
                            key_issues=correctness.metadata.get("key_issues", []),
                        )
                    completed_stages += 1

                if "review" in parallel_map:
                    review = parallel_map["review"]
                    if isinstance(review, GraderError):
                        logger.error(f"Review grader error: {review.error}")
                    else:
                        result.review = ReviewResult(score=review.score, review=review.reason)
                    completed_stages += 1

            # Stage: Criticality verification
            if self.config.enable_criticality:
                self._notify_progress(ReviewStage.CRITICALITY, completed_stages, total_stages)
                if result.correctness and result.correctness.score > 1:
                    logger.info("Running criticality verification...")
                    findings = self._format_findings(result.correctness)
                    criticality = await self.criticality_grader.aevaluate(pdf_data=pdf_data, findings=findings)
                    if isinstance(criticality, GraderError):
                        logger.error(f"Criticality grader error: {criticality.error}")
                        # Continue with partial results - leave criticality as None
                    else:
                        from cookbooks.paper_review.schema import CriticalityIssues

                        result.criticality = CriticalityResult(
                            score=criticality.score,
                            reasoning=criticality.reason,
                            issues=CriticalityIssues(**criticality.metadata.get("issues", {})),
                        )
                else:
                    logger.info("Skipping criticality verification (no issues found)")
                completed_stages += 1

            # Stage: BibTeX verification
            if self.config.enable_bib_verification and has_bib:
                self._notify_progress(ReviewStage.BIB_VERIFICATION, completed_stages, total_stages)
                logger.info("Running BibTeX verification...")
                if bib_content:
                    result.bib_verification = await self._verify_bib_content(bib_content)
                elif bib_path:
                    result.bib_verification = await self._verify_bib(bib_path)
                completed_stages += 1

            # Notify completion using dedicated method
            self._notify_completed()

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._notify_failed(str(e))
            raise
        finally:
            # Clean up any files uploaded to DashScope during this review run.
            if hasattr(self.model, "cleanup_files"):
                self.model.cleanup_files()

    async def generate_rebuttal(
        self,
        pdf_input: Union[str, Path, bytes],
        review_text: Optional[str] = None,
        review_result: Optional[PaperReviewResult] = None,
    ) -> RebuttalResult:
        """Generate a rebuttal draft for the author.

        Items that require new experiments or proofs will contain [TODO: ...]
        placeholders for the author to fill in.

        Args:
            pdf_input: Path to PDF file or PDF bytes
            review_text: Reviewer comments text. If None, extracted from review_result.
            review_result: Previous review_paper() result to extract review_text from.

        Returns:
            RebuttalResult with structured rebuttal draft
        """
        if not review_text and review_result and review_result.review:
            review_text = review_result.review.review
        if not review_text:
            raise ValueError("Either review_text or review_result with review must be provided")

        self._progress.reset(2)
        try:
            self._notify_progress(ReviewStage.LOADING_PDF, 0, 2)
            pdf_data = self._load_pdf(pdf_input)

            self._notify_progress(ReviewStage.REBUTTAL_GENERATION, 1, 2)
            logger.info("Generating rebuttal draft...")
            gen_result = await self.rebuttal_generation_grader.aevaluate(
                pdf_data=pdf_data,
                review_text=review_text,
            )

            if isinstance(gen_result, GraderError):
                logger.error(f"Rebuttal generation error: {gen_result.error}")
                self._notify_failed(f"Rebuttal generation failed: {gen_result.error}")
                return RebuttalResult(rebuttal_text="")

            concerns = [RebuttalConcern(**c) for c in gen_result.metadata.get("concerns", [])]

            result = RebuttalResult(
                rebuttal_text=gen_result.reason,
                concerns=concerns,
                general_suggestions=gen_result.metadata.get("general_suggestions", []),
            )

            self._notify_completed()
            return result
        except Exception as e:
            logger.error(f"Rebuttal generation failed: {e}")
            self._notify_failed(str(e))
            raise
        finally:
            if hasattr(self.model, "cleanup_files"):
                self.model.cleanup_files()

    async def assess_rebuttal(
        self,
        pdf_input: Union[str, Path, bytes],
        rebuttal_text: str,
        review_text: Optional[str] = None,
        original_score: Optional[int] = None,
        review_result: Optional[PaperReviewResult] = None,
    ) -> RebuttalAssessmentResult:
        """Assess whether a rebuttal adequately addresses reviewer concerns.

        Args:
            pdf_input: Path to PDF file or PDF bytes
            rebuttal_text: The author's rebuttal text
            review_text: Reviewer comments. If None, extracted from review_result.
            original_score: Original recommendation score (1-6). If None, extracted from review_result.
            review_result: Previous review_paper() result.

        Returns:
            RebuttalAssessmentResult with point-by-point assessment and updated score
        """
        if not review_text and review_result and review_result.review:
            review_text = review_result.review.review
        if original_score is None and review_result and review_result.review:
            original_score = review_result.review.score
        if not review_text:
            raise ValueError("Either review_text or review_result with review must be provided")
        if original_score is None:
            original_score = 3

        self._progress.reset(2)
        try:
            self._notify_progress(ReviewStage.LOADING_PDF, 0, 2)
            pdf_data = self._load_pdf(pdf_input)

            self._notify_progress(ReviewStage.REBUTTAL_ASSESSMENT, 1, 2)
            logger.info("Assessing rebuttal...")
            assess_result = await self.rebuttal_assessment_grader.aevaluate(
                pdf_data=pdf_data,
                review_text=review_text,
                rebuttal_text=rebuttal_text,
                original_score=original_score,
            )

            if isinstance(assess_result, GraderError):
                logger.error(f"Rebuttal assessment error: {assess_result.error}")
                self._notify_failed(f"Rebuttal assessment failed: {assess_result.error}")
                return RebuttalAssessmentResult(
                    updated_score=original_score,
                    original_score=original_score,
                )

            point_assessments = [
                RebuttalPointAssessment(**p) for p in assess_result.metadata.get("point_assessments", [])
            ]

            result = RebuttalAssessmentResult(
                updated_score=assess_result.score,
                original_score=original_score,
                score_change_reasoning=assess_result.metadata.get("score_change_reasoning", ""),
                overall_assessment=assess_result.reason,
                point_assessments=point_assessments,
                unresolved_concerns=assess_result.metadata.get("unresolved_concerns", []),
                rebuttal_strengths=assess_result.metadata.get("rebuttal_strengths", []),
            )

            self._notify_completed()
            return result
        except Exception as e:
            logger.error(f"Rebuttal assessment failed: {e}")
            self._notify_failed(str(e))
            raise
        finally:
            if hasattr(self.model, "cleanup_files"):
                self.model.cleanup_files()

    def _load_pdf(self, pdf_input: Union[str, Path, bytes]) -> str:
        """Load and encode PDF input to base64 data URL."""
        if isinstance(pdf_input, bytes):
            pdf_bytes = pdf_input
        else:
            pdf_bytes = load_pdf_bytes(pdf_input)
        pdf_data = encode_pdf_base64(pdf_bytes)
        if self.config.use_vision_for_pdf:
            self.model.warmup_vision_cache(pdf_data)
        return pdf_data

    async def _run_safety_checks(self, pdf_data: str) -> Dict[str, Any]:
        """Run jailbreaking and format checks in parallel."""
        issues = []
        format_ok = True

        # Jailbreaking + Format 并行执行，互相无依赖
        jailbreak_result, format_result = await asyncio.gather(
            self.jailbreaking_grader.aevaluate(pdf_data=pdf_data),
            self.format_grader.aevaluate(
                pdf_data=pdf_data,
                vision_max_pages=self.config.format_vision_max_pages,
            ),
        )

        if isinstance(jailbreak_result, GraderError):
            logger.error(f"Jailbreaking grader error: {jailbreak_result.error}")
        elif jailbreak_result.metadata.get("is_abuse"):
            issues.append(f"Jailbreaking detected: {jailbreak_result.reason}")

        if isinstance(format_result, GraderError):
            logger.error(f"Format grader error: {format_result.error}")
        elif format_result.score == 1:
            format_ok = False
            violations = format_result.metadata.get("violations", [])
            if violations:
                issues.append(f"Format violations: {', '.join(violations)}")

        return {
            "is_safe": len(issues) == 0 or not any("Jailbreaking" in i for i in issues),
            "format_ok": format_ok,
            "issues": issues,
        }

    def _format_findings(self, correctness: CorrectnessResult) -> str:
        """Format correctness findings for criticality verification."""
        lines = [
            f"Score: {correctness.score}",
            f"Reasoning: {correctness.reasoning}",
            "Key Issues:",
        ]
        for issue in correctness.key_issues:
            lines.append(f"- {issue}")
        return "\n".join(lines)

    async def _verify_bib(self, bib_path: str) -> Dict[str, BibVerificationSummary]:
        """Verify references in a .bib file."""
        try:
            checker = BibChecker(mailto=self.config.crossref_mailto)
            results = checker.check_bib_file(bib_path)

            suspect_refs = [r.reference.title for r in results["results"] if r.status.value == "suspect"]

            return {
                bib_path: BibVerificationSummary(
                    total_references=results["total_references"],
                    verified=results["verified"],
                    suspect=results["suspect"],
                    errors=results["errors"],
                    verification_rate=results["verification_rate"],
                    suspect_references=suspect_refs,
                )
            }
        except Exception as e:
            logger.error(f"BibTeX verification failed: {e}")
            return {}

    async def _verify_bib_content(self, bib_content: str) -> Dict[str, BibVerificationSummary]:
        """Verify references from .bib file content string."""
        try:
            checker = BibChecker(mailto=self.config.crossref_mailto)
            results = checker.check_bib_content(bib_content)

            suspect_refs = [r.reference.title for r in results["results"] if r.status.value == "suspect"]

            return {
                "uploaded.bib": BibVerificationSummary(
                    total_references=results["total_references"],
                    verified=results["verified"],
                    suspect=results["suspect"],
                    errors=results["errors"],
                    verification_rate=results["verification_rate"],
                    suspect_references=suspect_refs,
                )
            }
        except Exception as e:
            logger.error(f"BibTeX content verification failed: {e}")
            return {}

    async def review_tex_package(
        self,
        package_path: Union[str, Path, bytes],
        package_name: Optional[str] = None,
    ) -> PaperReviewResult:
        """Review a paper from TeX source package.

        Args:
            package_path: Path to .tar.gz or .zip file
            package_name: Name hint when passing bytes

        Returns:
            PaperReviewResult with evaluation results
        """
        processor = TexPackageProcessor()
        package = processor.process_package(package_path, package_name)

        bib_paths = [bib.path for bib in package.bib_files]
        logger.info(f"Found main.tex: {package.main_tex}")
        logger.info(f"Total .tex files: {len(package.files)}")
        logger.info(f"Found .bib files: {bib_paths}")

        result = PaperReviewResult()
        result.tex_info = TexPackageInfo(
            main_tex=package.main_tex,
            total_files=len(package.files),
            bib_files=bib_paths,
            figures=package.figure_paths,
        )

        # Verify bib if available
        if self.config.enable_bib_verification and package.bib_files:
            logger.info("Verifying references...")
            try:
                checker = BibChecker(mailto=self.config.crossref_mailto)
                result.bib_verification = {}

                for bib_file in package.bib_files:
                    bib_results = checker.check_bib_content(bib_file.content)
                    suspect_refs = [r.reference.title for r in bib_results["results"] if r.status.value == "suspect"]
                    result.bib_verification[bib_file.path] = BibVerificationSummary(
                        total_references=bib_results["total_references"],
                        verified=bib_results["verified"],
                        suspect=bib_results["suspect"],
                        errors=bib_results["errors"],
                        verification_rate=bib_results["verification_rate"],
                        suspect_references=suspect_refs,
                    )
            except ImportError:
                logger.warning("habanero not installed, skipping bib verification")

        # Store merged content for potential further processing
        result.metadata["merged_tex_content"] = package.merged_content
        result.metadata["tex_length"] = len(package.merged_content)

        return result

    async def review_and_report(
        self,
        pdf_input: Union[str, Path, bytes],
        paper_name: str = "Paper",
        bib_path: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        rebuttal_text: Optional[str] = None,
    ) -> tuple[PaperReviewResult, str]:
        """Review a paper, optionally handle rebuttal, and generate a Markdown report.

        When ``enable_rebuttal_generation`` is on and no *rebuttal_text* is supplied,
        a rebuttal draft is generated after the review and attached to the result.

        When ``enable_rebuttal_assessment`` is on and *rebuttal_text* is supplied,
        the rebuttal is assessed and the assessment is attached to the result.

        Args:
            pdf_input: Path to PDF file or PDF bytes
            paper_name: Name of the paper for the report
            bib_path: Optional path to .bib file
            output_path: Optional path to save the report (.md file)
            rebuttal_text: Optional author rebuttal text. When provided together
                with ``enable_rebuttal_assessment``, the pipeline will assess it.

        Returns:
            Tuple of (PaperReviewResult, markdown_report_string)
        """
        result = await self.review_paper(pdf_input, bib_path)

        if rebuttal_text and self.config.enable_rebuttal_assessment:
            result.rebuttal_assessment = await self.assess_rebuttal(
                pdf_input,
                rebuttal_text=rebuttal_text,
                review_result=result,
            )
        elif not rebuttal_text and self.config.enable_rebuttal_generation and result.review:
            result.rebuttal = await self.generate_rebuttal(
                pdf_input,
                review_result=result,
            )

        report = generate_report(result, paper_name, output_path)
        return result, report
