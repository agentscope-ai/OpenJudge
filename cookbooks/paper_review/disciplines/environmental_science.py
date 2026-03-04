# -*- coding: utf-8 -*-
"""Environmental Science discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

ENVIRONMENTAL_SCIENCE = DisciplineConfig(
    id="environmental_science",
    name="Environmental Science",
    venues=[
        "Nature Climate Change",
        "Nature Sustainability",
        "Global Change Biology",
        "Environmental Science & Technology",
        "Environmental Health Perspectives",
        "Ecology Letters",
        "Global Environmental Change",
        "One Earth",
        "Earth System Science Data",
        "Atmospheric Chemistry and Physics",
        "Water Research",
        "Journal of Cleaner Production",
    ],
    reviewer_context=(
        "You specialize in environmental science, including climate science, ecology, "
        "atmospheric science, hydrology, environmental chemistry, conservation biology, "
        "sustainability science, and environmental health. You are experienced evaluating "
        "field studies, remote sensing data, climate models, and life cycle assessments."
    ),
    evaluation_dimensions=[
        "Environmental Significance: Does this work address a critical environmental problem? "
        "Are findings relevant to policy, conservation, or mitigation efforts?",
        "Methodological Rigor: Are field sampling, remote sensing, or modeling methods "
        "appropriate and well-described? Are uncertainty estimates provided?",
        "Data Quality: Are data sources credible and properly validated? "
        "Are spatiotemporal scales appropriate for the conclusions drawn?",
        "Model Validity: If models are used, are they validated against observations? "
        "Are assumptions, limitations, and uncertainty ranges clearly stated?",
        "Originality: Does this provide new data from underrepresented regions, "
        "new mechanistic understanding, or a novel methodological approach?",
        "Interdisciplinary Integration: Does the paper appropriately integrate "
        "physical, biological, chemical, and/or social dimensions where relevant?",
        "Clarity: Are maps, figures, and data visualizations clear? "
        "Is the manuscript accessible to a broad environmental science readership?",
        "Ethics and Policy Relevance: Are data accessibility and reproducibility addressed? "
        "Are policy implications discussed responsibly?",
    ],
    correctness_categories=[
        "Statistical and Uncertainty Errors - Incorrect statistical approaches for "
        "spatiotemporal data, missing uncertainty quantification, improper trend analysis",
        "Data Inconsistencies - Numbers in text, tables, and figures that do not match; "
        "inconsistent units across the paper",
        "Model Application Errors - Climate or ecological models applied outside their "
        "validated range, incorrect parameter values",
        "Logical Contradictions - Conclusions not supported by the data or analysis presented",
        "Causal Inference Errors - Attributing environmental changes to a single cause "
        "without ruling out confounders",
        "Temporal/Spatial Scale Errors - Drawing global conclusions from local data, "
        "or vice versa, without justification",
    ],
    correctness_context=(
        "Pay special attention to: consistency of units across the paper, "
        "appropriate uncertainty quantification, whether model outputs are validated "
        "against observations, and whether spatial/temporal scale of analysis "
        "supports the stated conclusions."
    ),
    scoring_notes=(
        "For environmental science papers: missing uncertainty estimates for modeled quantities "
        "is a significant weakness. Data from a single site extrapolated globally lowers the score. "
        "Failure to make data publicly available when feasible is a concern for reproducibility."
    ),
)
