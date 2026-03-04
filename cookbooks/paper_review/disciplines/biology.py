# -*- coding: utf-8 -*-
"""Biology & Life Sciences discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

BIOLOGY = DisciplineConfig(
    id="biology",
    name="Biology & Life Sciences",
    venues=[
        "Cell",
        "Nature",
        "Science",
        "eLife",
        "PLOS Biology",
        "Nature Cell Biology",
        "Nature Genetics",
        "Molecular Cell",
        "Current Biology",
        "PNAS",
        "Developmental Cell",
        "Journal of Cell Biology",
        "Genome Biology",
    ],
    reviewer_context=(
        "You specialize in biology and life sciences, including molecular biology, "
        "cell biology, genetics, genomics, developmental biology, neuroscience, "
        "ecology, and evolutionary biology. You are experienced evaluating "
        "experimental designs, statistical analyses, and biological interpretations."
    ),
    evaluation_dimensions=[
        "Biological Significance: Does this work provide fundamental new insights into "
        "biological processes, mechanisms, or evolutionary relationships?",
        "Originality: Is this a genuinely new discovery, mechanism, or methodology? "
        "How does it advance the field beyond prior work?",
        "Experimental Rigor: Are appropriate controls included? Are experiments "
        "independently replicated? Are sample sizes sufficient for the claims made?",
        "Statistical Analysis: Are statistical methods appropriate for the data type? "
        "Are sample sizes, n values, and error statistics clearly reported?",
        "Reproducibility: Are methods described in sufficient detail to repeat? "
        "Are reagents, cell lines, and animal models fully specified?",
        "Clarity: Are figures clear and properly labeled? Is the manuscript logically "
        "structured? Are key findings clearly stated?",
        "Ethics: Is animal/human subject ethics approval mentioned? Are biosafety "
        "considerations addressed where relevant?",
        "Citations and Related Work: Are key precedents and competing hypotheses cited?",
    ],
    correctness_categories=[
        "Statistical Errors - Inappropriate statistical tests, wrong n values, "
        "incorrect error bar types (SD vs SEM vs 95% CI), missing multiple comparisons correction",
        "Data Inconsistencies - Numbers in text, figures, and tables that do not match",
        "Logical Contradictions - Conclusions or interpretations inconsistent with data shown",
        "Methodological Errors - Critical controls missing, inappropriate model organism or "
        "cell line for the question being asked",
        "Causal Inference Errors - Claiming mechanism or causality from correlational data",
        "Image/Figure Errors - Duplicated panels, inconsistent scale bars, or impossible data",
    ],
    correctness_context=(
        "Pay special attention to: adequacy of controls, consistency between "
        "figure data and text claims, appropriate statistical methodology, "
        "and whether mechanistic conclusions are supported by the experiments shown."
    ),
    scoring_notes=(
        "For biology papers: missing key controls (e.g., knockdown rescue, isotype controls) "
        "is a significant weakness. Overinterpretation of correlational data as mechanistic "
        "evidence lowers the score. Irreproducible reagents or unclear methods are major concerns."
    ),
)
