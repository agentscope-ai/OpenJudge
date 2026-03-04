# -*- coding: utf-8 -*-
"""Chemistry discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

CHEMISTRY = DisciplineConfig(
    id="chemistry",
    name="Chemistry",
    venues=[
        "Journal of the American Chemical Society (JACS)",
        "Angewandte Chemie",
        "Nature Chemistry",
        "Nature",
        "Chemical Science",
        "ACS Central Science",
        "Journal of Physical Chemistry Letters",
        "Organic Letters",
        "Inorganic Chemistry",
        "ACS Nano",
        "Nano Letters",
    ],
    reviewer_context=(
        "You specialize in chemistry, including organic, inorganic, physical, "
        "analytical, and computational chemistry, as well as materials chemistry "
        "and chemical biology. You are experienced evaluating synthetic procedures, "
        "spectroscopic characterization, and mechanistic studies."
    ),
    evaluation_dimensions=[
        "Scientific Significance: Does this work advance synthetic methodology, "
        "mechanistic understanding, or materials design in a meaningful way?",
        "Originality: Is the chemistry genuinely novel? Is this the first synthesis "
        "of a class of compounds, a new reaction type, or a new mechanistic insight?",
        "Experimental Rigor: Are reaction conditions, yields, and characterization data "
        "(NMR, MS, IR, X-ray, etc.) fully reported? Is the synthesis reproducible?",
        "Mechanistic Validity: Are proposed mechanisms supported by experimental evidence "
        "(isotope labeling, kinetics, DFT calculations)? Are alternative mechanisms considered?",
        "Characterization Completeness: Are all new compounds fully characterized with "
        "appropriate spectroscopic data? Is purity established?",
        "Clarity: Is the manuscript well organized? Are reaction schemes clear? "
        "Is the Supporting Information sufficient?",
        "Safety and Ethics: Are hazardous reagents and procedures identified? "
        "Are waste disposal and environmental considerations mentioned?",
        "Citations and Related Work: Are precedents in the literature appropriately cited?",
    ],
    correctness_categories=[
        "Chemical Structure Errors - Incorrect molecular structures, wrong stereochemistry, "
        "or inconsistent IUPAC naming",
        "Stoichiometry and Mass Balance Errors - Reaction equations that are not balanced, " "incorrect molar ratios",
        "Spectroscopic Data Inconsistencies - NMR, MS, or IR data that contradicts the " "proposed structure",
        "Thermodynamic/Kinetic Errors - Incorrect application of thermodynamic principles, " "invalid kinetic models",
        "Logical Contradictions - Conclusions not supported by the experimental evidence presented",
        "Data Inconsistencies - Numbers in text, tables, and schemes that do not match",
    ],
    correctness_context=(
        "Pay special attention to: structural correctness of depicted molecules, "
        "consistency of spectroscopic characterization data with proposed structures, "
        "mass balance in reaction schemes, and whether mechanistic claims are "
        "supported by the experimental data."
    ),
    scoring_notes=(
        "For chemistry papers: inadequate characterization of new compounds is a major reason "
        "to reject. Reactions reported without yield data or reproducibility information "
        "are incomplete. Missing spectroscopic data for key compounds lowers the score significantly."
    ),
)
