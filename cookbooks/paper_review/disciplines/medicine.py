# -*- coding: utf-8 -*-
"""Medicine & Health Sciences discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

MEDICINE = DisciplineConfig(
    id="medicine",
    name="Medicine & Health Sciences",
    venues=[
        "The Lancet", "New England Journal of Medicine (NEJM)", "JAMA",
        "BMJ", "Nature Medicine", "Cell", "PLOS Medicine",
        "Annals of Internal Medicine", "Circulation", "CHEST",
        "Journal of Clinical Oncology", "Radiology",
    ],
    reviewer_context=(
        "You specialize in medicine and health sciences, including clinical research, "
        "epidemiology, pharmacology, public health, and biomedical sciences. "
        "You are experienced with RCTs, cohort studies, meta-analyses, and clinical guidelines."
    ),
    evaluation_dimensions=[
        "Clinical Significance: Does the work address an important medical question? "
        "Are the findings likely to change clinical practice or advance patient care?",

        "Methodological Rigor: Is the study design appropriate (RCT, cohort, case-control, etc.)? "
        "Are sample sizes adequate? Is randomization and blinding properly implemented?",

        "Statistical Validity: Are statistical methods appropriate? Is multiple comparisons "
        "correction applied? Are confidence intervals and p-values correctly interpreted?",

        "Reproducibility: Are inclusion/exclusion criteria, interventions, outcomes, "
        "and analysis plans pre-registered or clearly specified?",

        "Clarity: Is the manuscript structured following CONSORT/STROBE/PRISMA guidelines? "
        "Are tables and figures clearly presented?",

        "Ethics and Patient Safety: Is IRB/ethics approval mentioned? Is informed consent "
        "described? Are potential harms or adverse events reported?",

        "Originality: Does this add new knowledge beyond what is already established? "
        "Is it sufficiently different from previous systematic reviews or trials?",

        "Limitations: Have the authors honestly acknowledged study limitations, "
        "confounders, and generalizability constraints?",
    ],
    correctness_categories=[
        "Statistical Errors - Incorrect statistical tests, misinterpreted p-values, "
        "wrong confidence intervals, improper multiple comparisons handling",
        "Methodological Contradictions - Study design claims inconsistent with actual methods",
        "Data Inconsistencies - Numbers in text, tables, and figures that do not match",
        "Logical Contradictions - Conclusions not supported by the presented results",
        "Causal Inference Errors - Claiming causality from observational data without justification",
        "Outcome Reporting Errors - Switching primary/secondary endpoints, selective reporting",
    ],
    correctness_context=(
        "Pay special attention to: statistical methodology, whether the study design "
        "supports the conclusions drawn, consistency of numbers across text and tables, "
        "and proper reporting of adverse events or outcomes."
    ),
    scoring_notes=(
        "For medical papers: unregistered trials, missing ethics approval, or "
        "inadequate patient safety reporting are major reasons to reject. "
        "Underpowered studies with negative results should be evaluated for their "
        "methodological contribution rather than the outcome alone."
    ),
)
