# -*- coding: utf-8 -*-
"""Economics & Finance discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

ECONOMICS = DisciplineConfig(
    id="economics",
    name="Economics & Finance",
    venues=[
        "American Economic Review (AER)",
        "Quarterly Journal of Economics (QJE)",
        "Journal of Political Economy (JPE)",
        "Review of Economic Studies (RES)",
        "Econometrica",
        "Journal of Finance",
        "Review of Financial Studies",
        "Journal of Financial Economics",
        "Journal of Economic Perspectives",
        "RAND Journal of Economics",
        "Economic Journal",
    ],
    reviewer_context=(
        "You specialize in economics and finance, including microeconomics, macroeconomics, "
        "econometrics, labor economics, industrial organization, international economics, "
        "public finance, asset pricing, corporate finance, and behavioral economics."
    ),
    evaluation_dimensions=[
        "Economic Significance: Does this paper address an important economic question? "
        "Are the findings likely to influence policy, theory, or further research?",
        "Identification Strategy: Is the causal identification strategy credible? "
        "Are endogeneity concerns properly addressed (IV, RDD, DiD, natural experiment)?",
        "Theoretical Contribution: If theoretical, is the model well-specified? "
        "Are equilibrium conditions and assumptions clearly stated and justified?",
        "Empirical Rigor: Are data sources credible and well-described? "
        "Are sample selection and variable construction clearly explained? "
        "Is the econometric specification appropriate?",
        "Robustness: Are results robust to alternative specifications, samples, "
        "and functional forms? Are placebo tests or falsification checks provided?",
        "Originality: Does this open a new area of inquiry, provide a novel identification "
        "strategy, or substantially revise existing understanding?",
        "Clarity: Is the argument logically structured? Are tables and figures informative? "
        "Is the economic intuition clearly explained?",
        "Citations and Related Work: Is the paper properly situated in the existing literature?",
    ],
    correctness_categories=[
        "Econometric Errors - Incorrect application of statistical methods, wrong standard error "
        "computation, invalid instrument conditions, misspecified models",
        "Mathematical Errors - Incorrect derivations in theoretical models, wrong proofs of propositions",
        "Data Inconsistencies - Summary statistics or regression results that contradict each other "
        "or stated sample sizes",
        "Logical Contradictions - Theoretical predictions inconsistent with the model's assumptions",
        "Identification Errors - Using instruments that clearly fail exclusion restriction, "
        "or DiD setups with violated parallel trends",
        "Causal Claims Without Identification - Claiming causal effects from observational regressions "
        "without credible identification strategy",
    ],
    correctness_context=(
        "Pay special attention to: validity of the identification strategy, correctness of "
        "econometric specifications, consistency of empirical results with theoretical predictions, "
        "and whether stated sample sizes match reported degrees of freedom."
    ),
    scoring_notes=(
        "For economics papers: weak identification (low first-stage F-statistics, invalid instruments) "
        "is a major reason to reject. Results that are not robust to minor specification changes "
        "lower the score significantly. Theoretical models without empirical grounding or "
        "motivation should be evaluated primarily on their mathematical and logical contribution."
    ),
)
