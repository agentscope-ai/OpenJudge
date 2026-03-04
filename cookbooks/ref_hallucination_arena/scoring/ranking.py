# -*- coding: utf-8 -*-
"""Ranking calculator: produce final rankings from objective scores."""

from typing import Dict

from loguru import logger

from cookbooks.ref_hallucination_arena.schema import ArenaResult, ModelScore


class RankingCalculator:
    """Calculate final rankings based on verification metrics.

    Ranking is by verification_rate (descending).
    Ties are broken by avg_confidence, then by completeness.
    """

    def calculate(self, model_scores: Dict[str, ModelScore]) -> ArenaResult:
        """Produce final ArenaResult with rankings.

        Args:
            model_scores: {model_name: ModelScore}

        Returns:
            ArenaResult with sorted rankings and aggregate stats.
        """
        # Sort: verification_rate > year_compliance_rate > avg_confidence > completeness
        sorted_models = sorted(
            model_scores.values(),
            key=lambda s: (
                s.verification_rate,
                s.year_compliance_rate,
                s.avg_confidence,
                s.completeness,
            ),
            reverse=True,
        )

        rankings = [(s.model_name, s.overall_score) for s in sorted_models]

        total_refs = sum(s.total_refs for s in sorted_models)
        total_verified = sum(s.verified_count for s in sorted_models)
        overall_accuracy = total_verified / total_refs if total_refs > 0 else 0.0

        result = ArenaResult(
            rankings=rankings,
            model_scores=model_scores,
            total_queries=0,  # Will be set by pipeline
            total_references=total_refs,
            total_verified=total_verified,
            overall_accuracy=overall_accuracy,
        )

        # Log rankings
        logger.info("=" * 60)
        logger.info("REFERENCE HALLUCINATION ARENA - RANKINGS")
        logger.info("=" * 60)
        for rank, (name, score) in enumerate(rankings, 1):
            ms = model_scores[name]
            bar_len = int(ms.overall_accuracy * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            yc_str = f"  year_ok={ms.year_compliance_rate:.1%}" if ms.year_constrained_refs > 0 else ""
            logger.info(
                f"  {rank}. {name:<16} [{bar}] "
                f"overall={ms.overall_accuracy:.1%}  "
                f"title={ms.title_accuracy:.1%}  "
                f"author={ms.author_accuracy:.1%}  "
                f"doi={ms.doi_accuracy:.1%}{yc_str}  "
                f"refs={ms.total_refs}"
            )
        logger.info("=" * 60)

        return result
