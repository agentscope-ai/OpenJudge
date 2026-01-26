# -*- coding: utf-8 -*-
"""Processors for paper review."""

from cookbooks.paper_review.processors.bib_checker import (
    BibChecker,
    MatchDetail,
    Reference,
    VerificationResult,
)
from cookbooks.paper_review.processors.tex_processor import (
    BibFile,
    TexFile,
    TexPackage,
    TexPackageProcessor,
)

__all__ = [
    "BibChecker",
    "Reference",
    "VerificationResult",
    "MatchDetail",
    "TexPackageProcessor",
    "TexPackage",
    "TexFile",
    "BibFile",
]
