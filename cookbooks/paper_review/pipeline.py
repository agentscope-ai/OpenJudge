# -*- coding: utf-8 -*-
"""End-to-end paper review pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from cookbooks.paper_review.graders import (
    CorrectnessGrader,
    CriticalityGrader,
    FormatGrader,
    JailbreakingGrader,
    ReviewGrader,
)
from cookbooks.paper_review.processors import BibChecker, TexPackageProcessor
from cookbooks.paper_review.report import generate_report
from cookbooks.paper_review.schema import (
    BibVerificationSummary,
    CorrectnessResult,
    CriticalityResult,
    PaperReviewResult,
    ReviewResult,
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


class PaperReviewPipeline:
    """End-to-end paper review pipeline using OpenJudge."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        # Use LiteLLM for native PDF support
        from cookbooks.paper_review.models import LiteLLMModel

        self.model = LiteLLMModel(
            model=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            timeout=config.timeout,
        )
        self._init_graders()

    def _init_graders(self):
        """Initialize all graders."""
        self.correctness_grader = CorrectnessGrader(self.model)
        self.review_grader = ReviewGrader(self.model)
        self.criticality_grader = CriticalityGrader(self.model)
        self.format_grader = FormatGrader(self.model)
        self.jailbreaking_grader = JailbreakingGrader(self.model)

    async def review_paper(
        self,
        pdf_input: Union[str, Path, bytes],
        bib_path: Optional[str] = None,
    ) -> PaperReviewResult:
        """Review a single paper.

        Args:
            pdf_input: Path to PDF file or PDF bytes
            bib_path: Optional path to .bib file for reference verification

        Returns:
            PaperReviewResult with all evaluation results
        """
        # Load and encode PDF
        if isinstance(pdf_input, bytes):
            pdf_bytes = pdf_input
        else:
            pdf_bytes = load_pdf_bytes(pdf_input)
        pdf_data = encode_pdf_base64(pdf_bytes)

        result = PaperReviewResult()

        # Phase 1: Safety checks
        if self.config.enable_safety_checks:
            logger.info("Running safety checks...")
            safety_result = await self._run_safety_checks(pdf_data)
            if not safety_result["is_safe"]:
                result.is_safe = False
                result.safety_issues = safety_result["issues"]
                return result
            result.format_compliant = safety_result["format_ok"]

        # Phase 2: Core evaluation
        if self.config.enable_correctness:
            logger.info("Running correctness detection...")
            correctness = await self.correctness_grader.aevaluate(pdf_data=pdf_data)
            if isinstance(correctness, GraderError):
                logger.error(f"Correctness grader error: {correctness.error}")
            else:
                result.correctness = CorrectnessResult(
                    score=correctness.score,
                    reasoning=correctness.reason,
                    key_issues=correctness.metadata.get("key_issues", []),
                )

        if self.config.enable_review:
            logger.info("Running paper review...")
            review = await self.review_grader.aevaluate(pdf_data=pdf_data)
            if isinstance(review, GraderError):
                logger.error(f"Review grader error: {review.error}")
            else:
                result.review = ReviewResult(score=review.score, review=review.reason)

        # Phase 3: Criticality verification
        if self.config.enable_criticality and result.correctness and result.correctness.score > 1:
            logger.info("Running criticality verification...")
            findings = self._format_findings(result.correctness)
            criticality = await self.criticality_grader.aevaluate(pdf_data=pdf_data, findings=findings)
            if isinstance(criticality, GraderError):
                logger.error(f"Criticality grader error: {criticality.error}")
            else:
                from cookbooks.paper_review.schema import CriticalityIssues

                result.criticality = CriticalityResult(
                    score=criticality.score,
                    reasoning=criticality.reason,
                    issues=CriticalityIssues(**criticality.metadata.get("issues", {})),
                )

        # Phase 4: BibTeX verification
        if self.config.enable_bib_verification and bib_path:
            logger.info("Running BibTeX verification...")
            result.bib_verification = await self._verify_bib(bib_path)

        return result

    async def _run_safety_checks(self, pdf_data: str) -> Dict[str, Any]:
        """Run jailbreaking and format checks."""
        issues = []
        format_ok = True

        # Jailbreaking check
        jailbreak_result = await self.jailbreaking_grader.aevaluate(pdf_data=pdf_data)
        if isinstance(jailbreak_result, GraderError):
            logger.error(f"Jailbreaking grader error: {jailbreak_result.error}")
        elif jailbreak_result.metadata.get("is_abuse"):
            issues.append(f"Jailbreaking detected: {jailbreak_result.reason}")

        # Format check
        format_result = await self.format_grader.aevaluate(pdf_data=pdf_data)
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
    ) -> tuple[PaperReviewResult, str]:
        """Review a paper and generate a Markdown report.

        Args:
            pdf_input: Path to PDF file or PDF bytes
            paper_name: Name of the paper for the report
            bib_path: Optional path to .bib file
            output_path: Optional path to save the report (.md file)

        Returns:
            Tuple of (PaperReviewResult, markdown_report_string)
        """
        result = await self.review_paper(pdf_input, bib_path)
        report = generate_report(result, paper_name, output_path)
        return result, report
