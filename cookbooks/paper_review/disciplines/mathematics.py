# -*- coding: utf-8 -*-
"""Mathematics discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

MATHEMATICS = DisciplineConfig(
    id="mathematics",
    name="Mathematics",
    venues=[
        "Annals of Mathematics", "Journal of the American Mathematical Society (JAMS)",
        "Inventiones Mathematicae", "Acta Mathematica",
        "Duke Mathematical Journal", "Mathematische Annalen",
        "Communications in Mathematical Physics", "Advances in Mathematics",
        "Journal of Differential Geometry", "Compositio Mathematica",
        "SIAM Journal on Mathematical Analysis",
    ],
    reviewer_context=(
        "You specialize in mathematics, including pure mathematics (algebra, analysis, "
        "geometry, topology, number theory, combinatorics) and applied mathematics "
        "(differential equations, numerical analysis, mathematical physics, optimization, "
        "probability theory). You are capable of verifying rigorous mathematical proofs."
    ),
    evaluation_dimensions=[
        "Mathematical Correctness: Are all theorems, lemmas, and propositions correctly stated "
        "and rigorously proved? Are all cases handled? Are there gaps in the arguments?",

        "Significance: Does this resolve an open problem, establish a new connection between "
        "fields, or introduce a powerful new technique with broad applicability?",

        "Originality: Is the mathematical approach genuinely new? Does it introduce "
        "new concepts, structures, or proof techniques?",

        "Clarity and Exposition: Is the paper well-organized? Are definitions precise "
        "and notation consistent? Are key ideas explained before technical details?",

        "Completeness: Are all relevant cases and edge cases addressed? "
        "Are counterexamples or limitations of the results discussed?",

        "Context and Motivation: Is the mathematical motivation clearly explained? "
        "Is the result placed appropriately in the existing mathematical landscape?",

        "Citations and Related Work: Are prior results that are used or generalized properly cited?",
    ],
    correctness_categories=[
        "Proof Errors - Invalid logical steps, missing cases, incorrect use of theorems, "
        "circular reasoning, or unjustified claims",
        "Definitional Errors - Imprecise or internally inconsistent definitions that undermine "
        "the statements being proved",
        "Counterexamples to Stated Results - Cases that violate the stated theorem or lemma",
        "Notation and Formula Errors - Incorrect formulas, wrong indices, or notation "
        "that is inconsistently used across the paper",
        "Logical Contradictions - Propositions that contradict each other or are inconsistent "
        "with stated hypotheses",
        "Missing Hypotheses - Theorems that require unstated assumptions to hold",
    ],
    correctness_context=(
        "This is a mathematics paper. Proof correctness is the single most important criterion. "
        "Read each proof step carefully. Check that all assumptions are used, all cases are "
        "covered, and each logical step follows from the previous ones. "
        "Use LaTeX notation when citing specific errors in formulas."
    ),
    scoring_notes=(
        "For mathematics papers: a single unfixable error in the main theorem proof warrants "
        "rejection regardless of other contributions. Papers with correct but minor results "
        "may still score low on significance. An elegant proof of a known result may score "
        "high if the technique is sufficiently novel."
    ),
)
