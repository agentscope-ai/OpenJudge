# -*- coding: utf-8 -*-
"""Composite verifier: chains multiple sources with discipline-aware routing."""

from typing import Dict, List, Optional

from loguru import logger

from cookbooks.ref_hallucination_arena.schema import (
    Discipline,
    Reference,
    VerificationConfig,
    VerificationResult,
    VerificationStatus,
)
from cookbooks.ref_hallucination_arena.verifiers.arxiv_verifier import ArxivVerifier
from cookbooks.ref_hallucination_arena.verifiers.base_verifier import BaseVerifier
from cookbooks.ref_hallucination_arena.verifiers.crossref_verifier import (
    CrossrefVerifier,
)
from cookbooks.ref_hallucination_arena.verifiers.dblp_verifier import DblpVerifier
from cookbooks.ref_hallucination_arena.verifiers.pubmed_verifier import PubmedVerifier

# Default verification order per discipline
DISCIPLINE_ROUTES: Dict[str, List[str]] = {
    Discipline.COMPUTER_SCIENCE: ["crossref", "dblp", "arxiv", "pubmed"],
    Discipline.BIOMEDICAL: ["crossref", "pubmed", "arxiv", "dblp"],
    Discipline.PHYSICS: ["crossref", "arxiv", "dblp", "pubmed"],
    Discipline.CHEMISTRY: ["crossref", "pubmed", "arxiv", "dblp"],
    Discipline.SOCIAL_SCIENCE: ["crossref", "pubmed", "dblp", "arxiv"],
    Discipline.INTERDISCIPLINARY: ["crossref", "arxiv", "pubmed", "dblp"],
    Discipline.OTHER: ["crossref", "arxiv", "pubmed", "dblp"],
}

# Default order (DOI always goes first via crossref)
DEFAULT_ORDER = ["crossref", "arxiv", "pubmed", "dblp"]


class CompositeVerifier:
    """Chains multiple verifiers with discipline-aware routing.

    For each reference, tries verifiers in priority order until one returns
    VERIFIED. The priority order depends on the discipline of the query.

    Args:
        config: Verification configuration.
        discipline: Default discipline for routing (can be overridden per-call).
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        discipline: str = Discipline.OTHER,
    ):
        config = config or VerificationConfig()
        self.config = config
        self.default_discipline = discipline

        # Initialize individual verifiers
        self._verifiers: Dict[str, BaseVerifier] = {
            "crossref": CrossrefVerifier(mailto=config.crossref_mailto, timeout=config.timeout),
            "pubmed": PubmedVerifier(api_key=config.pubmed_api_key, timeout=config.timeout),
            "arxiv": ArxivVerifier(timeout=config.timeout),
            "dblp": DblpVerifier(timeout=config.timeout),
        }

    def verify(
        self,
        ref: Reference,
        discipline: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a single reference using discipline-routed chain.

        Tries verifiers in priority order. Stops on first VERIFIED result.
        If DOI is present, Crossref DOI lookup always goes first regardless of route.

        Args:
            ref: Reference to verify.
            discipline: Discipline for routing. Uses default if not provided.

        Returns:
            Best VerificationResult found.
        """
        disc = discipline or self.default_discipline
        order = DISCIPLINE_ROUTES.get(disc, DEFAULT_ORDER)

        # DOI shortcut: crossref always first when DOI is available
        if ref.doi and order[0] != "crossref":
            order = ["crossref"] + [v for v in order if v != "crossref"]

        best_partial: Optional[VerificationResult] = None

        for verifier_name in order:
            verifier = self._verifiers.get(verifier_name)
            if not verifier:
                continue

            try:
                result = verifier.verify(ref)

                if result.status == VerificationStatus.VERIFIED:
                    return result

                # Track best partial match (has match_detail) for stats
                if result.match_detail is not None:
                    if best_partial is None or result.confidence > best_partial.confidence:
                        best_partial = result

            except Exception as e:
                logger.debug(f"Verifier {verifier_name} error for '{ref.title[:50]}': {e}")
                continue

        # Return best partial (with match_detail for per-field stats), or plain NOT_FOUND
        if best_partial:
            return best_partial

        return VerificationResult(
            reference=ref,
            status=VerificationStatus.NOT_FOUND,
            message=f"Not found in any source ({', '.join(order)})",
        )

    def verify_batch(
        self,
        references: List[Reference],
        discipline: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> List[VerificationResult]:
        """Verify a batch of references sequentially.

        Note: Parallelism is now handled at the pipeline level (query-level
        parallel workers), so each batch runs sequentially to avoid thread
        explosion and excessive API pressure.

        Args:
            references: List of references to verify.
            discipline: Discipline for routing.
            max_workers: Ignored (kept for API compatibility).

        Returns:
            List of VerificationResult in same order as input.
        """
        results: List[VerificationResult] = []
        for ref in references:
            try:
                results.append(self.verify(ref, discipline))
            except Exception as e:
                results.append(
                    VerificationResult(
                        reference=ref,
                        status=VerificationStatus.ERROR,
                        message=f"Verification error: {e}",
                    )
                )
        return results

    def close(self) -> None:
        """Close all verifier HTTP clients."""
        for verifier in self._verifiers.values():
            try:
                verifier.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
