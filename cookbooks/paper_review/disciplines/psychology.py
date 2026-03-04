# -*- coding: utf-8 -*-
"""Psychology & Cognitive Science discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

PSYCHOLOGY = DisciplineConfig(
    id="psychology",
    name="Psychology & Cognitive Science",
    venues=[
        "Psychological Science",
        "Journal of Personality and Social Psychology (JPSP)",
        "Psychological Review",
        "Cognition",
        "Journal of Experimental Psychology: General",
        "Perspectives on Psychological Science",
        "Annual Review of Psychology",
        "Nature Human Behaviour",
        "Trends in Cognitive Sciences",
        "Journal of Cognitive Neuroscience",
        "Psychonomic Bulletin & Review",
    ],
    reviewer_context=(
        "You specialize in psychology and cognitive science, including experimental psychology, "
        "social psychology, cognitive psychology, developmental psychology, clinical psychology, "
        "neuropsychology, and cognitive neuroscience. You are experienced with both "
        "experimental and survey-based research designs."
    ),
    evaluation_dimensions=[
        "Theoretical Contribution: Does the paper advance theoretical understanding "
        "of psychological or cognitive phenomena? Are the hypotheses clearly derived "
        "from existing theory?",
        "Study Design: Is the experimental or survey design appropriate for the hypotheses? "
        "Are confounds and demand characteristics adequately controlled?",
        "Statistical Power and Validity: Are sample sizes sufficient (power analysis reported)? "
        "Is the study pre-registered? Are effect sizes reported alongside p-values?",
        "Replicability: Are stimuli, measures, and procedures described in enough detail "
        "to replicate? Are materials available as supplementary?",
        "Originality: Does this challenge an established finding, provide a new mechanism, "
        "or extend prior work in a meaningful way?",
        "Clarity: Are hypotheses clearly stated before presenting results? "
        "Are figures and tables interpretable? Is the Discussion appropriately tempered?",
        "Ethics: Is IRB approval mentioned? Is participant consent and deception/debriefing "
        "protocol described where applicable?",
        "Citations and Related Work: Are competing theoretical accounts and prior replications cited?",
    ],
    correctness_categories=[
        "Statistical Errors - Incorrect statistical tests for the data type, misinterpreted "
        "p-values, inflated effect sizes due to optional stopping, incorrect degrees of freedom",
        "Data Inconsistencies - Sample sizes, means, or statistics that differ between text, " "tables, and figures",
        "Logical Contradictions - Conclusions that are not supported by the data or that "
        "contradict stated hypotheses",
        "Measurement Errors - Using scales or instruments not validated for the population studied",
        "Causal Inference Errors - Drawing causal conclusions from correlational or cross-sectional designs",
        "HARKing (Hypothesizing After Results Known) - Evidence that hypotheses were formulated "
        "after seeing the data without disclosure",
    ],
    correctness_context=(
        "Pay special attention to: adequacy of sample sizes (power analysis), "
        "consistency of statistics across text and tables, whether pre-registration "
        "is consistent with the analyses reported, and whether causal language is "
        "appropriate for the study design."
    ),
    scoring_notes=(
        "For psychology papers: underpowered studies (N < 50 per cell without justification) "
        "are a significant weakness. Unregistered studies with p-values just below 0.05 "
        "should be evaluated skeptically. Failure to report effect sizes lowers the score. "
        "Successful replications of important findings are valuable even without novel contributions."
    ),
)
