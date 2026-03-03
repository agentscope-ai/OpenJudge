# -*- coding: utf-8 -*-
"""Social Sciences discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

SOCIAL_SCIENCES = DisciplineConfig(
    id="social_sciences",
    name="Social Sciences",
    venues=[
        "American Sociological Review (ASR)", "American Journal of Sociology (AJS)",
        "American Political Science Review (APSR)", "World Politics",
        "International Organization", "Administrative Science Quarterly (ASQ)",
        "Social Forces", "Annual Review of Sociology",
        "Journal of Communication", "Public Opinion Quarterly",
        "Demography", "Social Science & Medicine",
    ],
    reviewer_context=(
        "You specialize in social sciences, including sociology, political science, "
        "communication, public administration, demography, and related fields. "
        "You are experienced evaluating both quantitative (survey, experimental, "
        "computational) and qualitative (ethnographic, interview, case study) research."
    ),
    evaluation_dimensions=[
        "Theoretical Contribution: Does the paper advance sociological or political theory? "
        "Are the theoretical claims clearly derived and tested?",

        "Research Design: Is the methodology (quantitative, qualitative, or mixed) "
        "appropriate for the research question? Are sampling strategies justified?",

        "Empirical Rigor: For quantitative work: are statistical methods appropriate, "
        "are confounders controlled, are effect sizes reported? "
        "For qualitative work: is the analytic approach systematic and transparent?",

        "Generalizability: Are the boundaries of generalizability clearly stated? "
        "Is the sample representative of the intended population?",

        "Originality: Does this provide new theoretical insight, study a novel population, "
        "or bring new data to bear on an important question?",

        "Clarity: Is the argument logically presented? Are concepts clearly defined? "
        "Are findings presented in a way accessible to the broader social science audience?",

        "Ethics: Are research ethics (IRB approval, informed consent, data privacy) "
        "described? Are positionality and potential biases acknowledged in qualitative work?",

        "Citations and Related Work: Is the paper properly grounded in relevant literature "
        "across subfields?",
    ],
    correctness_categories=[
        "Statistical Errors - Incorrect statistical models for the outcome type "
        "(e.g., OLS on a binary outcome), wrong standard errors, improper causal interpretation",
        "Data Inconsistencies - Sample sizes, percentages, or statistics that differ "
        "between text, tables, and figures",
        "Logical Contradictions - Conclusions not supported by the evidence presented",
        "Causal Inference Errors - Claiming causal effects from cross-sectional or "
        "correlational data without appropriate identification",
        "Measurement Validity Errors - Using proxies or indicators that clearly do not "
        "measure the intended theoretical concept",
        "Sampling/Generalizability Errors - Generalizing to populations not represented "
        "in the sample",
    ],
    correctness_context=(
        "Pay special attention to: appropriateness of statistical models for the data type, "
        "whether causal language is justified by the research design, consistency of "
        "reported statistics across the manuscript, and whether the sample supports "
        "the stated scope of conclusions."
    ),
    scoring_notes=(
        "For social science papers: purely descriptive work without theoretical contribution "
        "scores lower. Qualitative papers should be evaluated on analytic rigor and transparency, "
        "not by quantitative standards. Mixed-methods papers should justify why each method "
        "is necessary and how they complement each other."
    ),
)
