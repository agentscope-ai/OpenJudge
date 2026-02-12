# -*- coding: utf-8 -*-
"""Objective scorer: compute verification-based metrics for each model."""

from collections import defaultdict
from typing import Dict, List

from loguru import logger

from cookbooks.ref_hallucination_arena.schema import (
    DisciplineScore,
    ModelScore,
    ModelVerificationResult,
    VerificationStatus,
)


class ObjectiveScorer:
    """Compute objective scores based purely on reference verification results.

    Metrics computed per model:
      - verification_rate / overall_accuracy: verified / total  (all fields match)
      - hallucination_rate: not_found / total
      - Per-field accuracy: title / author / year / DOI accuracy
      - avg_confidence: mean confidence of verified refs
      - completeness: ratio of refs that have DOI, year, and authors filled
      - verified_by_source: breakdown by verification source
      - discipline_scores: per-discipline breakdown
    """

    def score_model(
        self,
        model_name: str,
        verification_results: List[ModelVerificationResult],
    ) -> ModelScore:
        """Compute aggregate score for a model across all queries.

        Args:
            model_name: Name of the model.
            verification_results: Per-query verification results for this model.

        Returns:
            ModelScore with all objective metrics.
        """
        total_refs = 0
        verified = 0
        suspect = 0
        not_found = 0
        errors = 0
        confidence_sum = 0.0
        confidence_count = 0
        completeness_sum = 0.0
        source_counts: Dict[str, int] = defaultdict(int)

        # Per-field accuracy accumulators
        title_correct = 0
        author_correct = 0
        year_correct = 0
        doi_correct = 0

        # Year constraint accumulators
        year_constrained_refs = 0
        year_compliant_count = 0

        # Per-discipline accumulators
        disc_totals: Dict[str, int] = defaultdict(int)
        disc_verified: Dict[str, int] = defaultdict(int)
        disc_confidence_sum: Dict[str, float] = defaultdict(float)
        disc_confidence_count: Dict[str, int] = defaultdict(int)

        for mvr in verification_results:
            disc = mvr.discipline or "other"

            # Accumulate year constraint stats from per-query results
            if mvr.has_year_constraint:
                year_constrained_refs += mvr.total_refs
                year_compliant_count += mvr.year_compliant

            for vr in mvr.results:
                total_refs += 1
                disc_totals[disc] += 1

                # Per-field accuracy: read from match_detail flags
                md = vr.match_detail
                if md:
                    if md.title_exact:
                        title_correct += 1
                    if md.author_exact:
                        author_correct += 1
                    if md.year_exact:
                        year_correct += 1
                    if md.doi_exact:
                        doi_correct += 1

                if vr.status == VerificationStatus.VERIFIED:
                    verified += 1
                    disc_verified[disc] += 1
                    if vr.source:
                        source_counts[vr.source] += 1
                    if vr.confidence > 0:
                        confidence_sum += vr.confidence
                        confidence_count += 1
                        disc_confidence_sum[disc] += vr.confidence
                        disc_confidence_count[disc] += 1
                elif vr.status == VerificationStatus.SUSPECT:
                    suspect += 1
                elif vr.status == VerificationStatus.NOT_FOUND:
                    not_found += 1
                else:
                    errors += 1

                # Completeness: has title (always), + DOI + year + authors
                ref = vr.reference
                filled = sum(
                    [
                        bool(ref.doi),
                        bool(ref.year),
                        bool(ref.authors),
                    ]
                )
                completeness_sum += filled / 3.0

        verification_rate = verified / total_refs if total_refs > 0 else 0.0
        hallucination_rate = (suspect + not_found) / total_refs if total_refs > 0 else 0.0
        avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0.0
        completeness = completeness_sum / total_refs if total_refs > 0 else 0.0
        year_compliance_rate = year_compliant_count / year_constrained_refs if year_constrained_refs > 0 else 0.0

        # Per-field accuracy rates
        title_accuracy = title_correct / total_refs if total_refs > 0 else 0.0
        author_accuracy = author_correct / total_refs if total_refs > 0 else 0.0
        year_accuracy = year_correct / total_refs if total_refs > 0 else 0.0
        doi_accuracy = doi_correct / total_refs if total_refs > 0 else 0.0
        overall_accuracy = verification_rate  # all fields correct = verified

        # Per-discipline scores
        discipline_scores = {}
        for disc in disc_totals:
            dt = disc_totals[disc]
            dv = disc_verified[disc]
            d_conf = disc_confidence_sum[disc] / disc_confidence_count[disc] if disc_confidence_count[disc] > 0 else 0.0
            discipline_scores[disc] = DisciplineScore(
                discipline=disc,
                total_refs=dt,
                verified=dv,
                verification_rate=dv / dt if dt > 0 else 0.0,
                hallucination_rate=(dt - dv) / dt if dt > 0 else 0.0,
                avg_confidence=d_conf,
            )

        return ModelScore(
            model_name=model_name,
            total_refs=total_refs,
            verified_count=verified,
            suspect_count=suspect,
            not_found_count=not_found,
            error_count=errors,
            verification_rate=verification_rate,
            hallucination_rate=hallucination_rate,
            avg_confidence=avg_confidence,
            completeness=completeness,
            title_accuracy=title_accuracy,
            author_accuracy=author_accuracy,
            year_accuracy=year_accuracy,
            doi_accuracy=doi_accuracy,
            overall_accuracy=overall_accuracy,
            year_constrained_refs=year_constrained_refs,
            year_compliant_count=year_compliant_count,
            year_compliance_rate=year_compliance_rate,
            discipline_scores=discipline_scores,
            verified_by_source=dict(source_counts),
            overall_score=overall_accuracy,
        )

    def score_all_models(
        self,
        all_results: Dict[str, List[ModelVerificationResult]],
    ) -> Dict[str, ModelScore]:
        """Compute scores for all models.

        Args:
            all_results: {model_name: [ModelVerificationResult, ...]}

        Returns:
            {model_name: ModelScore}
        """
        scores = {}
        for model_name, results in all_results.items():
            scores[model_name] = self.score_model(model_name, results)
            ms = scores[model_name]
            logger.info(
                f"  {model_name}: overall={ms.overall_accuracy:.1%}, "
                f"title={ms.title_accuracy:.1%}, author={ms.author_accuracy:.1%}, "
                f"year={ms.year_accuracy:.1%}, doi={ms.doi_accuracy:.1%}, "
                f"refs={ms.total_refs}"
            )
        return scores
