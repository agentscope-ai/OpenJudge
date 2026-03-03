# -*- coding: utf-8 -*-
"""Physics discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

PHYSICS = DisciplineConfig(
    id="physics",
    name="Physics",
    venues=[
        "Physical Review Letters (PRL)", "Physical Review X (PRX)",
        "Nature Physics", "Nature", "Science",
        "Journal of High Energy Physics (JHEP)", "Nuclear Physics B",
        "The Astrophysical Journal (ApJ)", "Astronomy & Astrophysics",
        "Communications Physics", "New Journal of Physics",
    ],
    reviewer_context=(
        "You specialize in physics, covering areas such as condensed matter, "
        "high energy physics, astrophysics, quantum mechanics, optics, and "
        "statistical physics. You are proficient in evaluating theoretical derivations "
        "and experimental physics research."
    ),
    evaluation_dimensions=[
        "Physical Validity: Are the physical arguments, derivations, and models correct? "
        "Do the results obey known conservation laws and symmetries?",

        "Significance: Does this advance fundamental understanding or open new research directions? "
        "Is it a breakthrough or incremental improvement?",

        "Originality: Is this genuinely new physics or a new theoretical framework? "
        "How does it compare to existing models and prior experimental results?",

        "Experimental Rigor (if applicable): Are measurements described in sufficient detail? "
        "Are systematic and statistical uncertainties properly quantified and propagated?",

        "Theoretical Rigor (if applicable): Are assumptions clearly stated? "
        "Are approximations justified? Are limits and boundary conditions properly handled?",

        "Clarity: Is the notation standard and consistent? Are derivations easy to follow? "
        "Are key results highlighted?",

        "Reproducibility: Is the experimental setup described in enough detail to reproduce? "
        "Are simulation/computation methods and codes described?",

        "Citations and Related Work: Are seminal works cited? Is the relationship to "
        "existing theory and experiment clearly discussed?",
    ],
    correctness_categories=[
        "Mathematical Derivation Errors - Incorrect algebra, calculus, or tensor operations; "
        "invalid steps in theoretical derivations",
        "Dimensional Analysis Errors - Units or dimensions that do not match in equations",
        "Physical Law Violations - Results that violate conservation laws, symmetry principles, "
        "or established physical constraints",
        "Experimental Data Inconsistencies - Figures, tables, or text that contradict each other; "
        "error bars that are inconsistent with stated uncertainties",
        "Logical Contradictions - Claims that contradict the paper's own results or methodology",
        "Approximation Errors - Invalid approximations applied outside their domain of validity",
    ],
    correctness_context=(
        "Pay special attention to: dimensional consistency of all equations, "
        "correctness of key derivations, whether experimental uncertainties are "
        "properly propagated, and whether conclusions are consistent with the data presented."
    ),
    scoring_notes=(
        "For physics papers: a single fatal flaw in a core derivation is sufficient for rejection. "
        "Experimental papers without proper uncertainty quantification should score lower. "
        "Extraordinary claims require extraordinary evidence."
    ),
)
