# -*- coding: utf-8 -*-
"""Computer Science discipline configuration."""

from cookbooks.paper_review.disciplines.base import DisciplineConfig

CS = DisciplineConfig(
    id="cs",
    name="Computer Science",
    venues=[
        "NeurIPS", "ICLR", "ICML", "CVPR", "ICCV", "ECCV",
        "ACL", "EMNLP", "NAACL", "SIGMOD", "VLDB", "OSDI",
        "SOSP", "CCS", "IEEE S&P", "USENIX Security",
        "Nature Machine Intelligence", "JMLR", "TPAMI",
    ],
    reviewer_context=(
        "You specialize in computer science, including areas such as "
        "machine learning, systems, networking, security, databases, "
        "programming languages, and human-computer interaction."
    ),
    evaluation_dimensions=[
        "Quality: Is the submission technically sound? Are claims well supported "
        "by theoretical analysis or experimental results? Are the methods appropriate? "
        "Is this a complete piece of work?",

        "Clarity: Is the submission clearly written and well organized? Does it "
        "adequately inform the reader with enough information for reproduction?",

        "Significance: Are the results impactful for the CS community? Will others "
        "likely use or build on these ideas? Does it address a difficult problem "
        "better than previous work?",

        "Originality: Does the work provide new insights or deepen understanding? "
        "Is it clear how this differs from previous contributions?",

        "Reproducibility: Does the paper provide sufficient detail for an expert to "
        "reproduce the results? Are code, datasets, and hyperparameters described?",

        "Ethics and Limitations: Have the authors adequately addressed limitations, "
        "potential misuse, and societal impact?",

        "Citations and Related Work: Are relevant prior works properly cited and compared?",
    ],
    correctness_categories=[
        "Mathematical Errors - Incorrect formulas, wrong derivations, invalid proofs, arithmetic mistakes",
        "Logical Contradictions - Statements that contradict each other within the paper",
        "Algorithmic Errors - Incorrect algorithm descriptions, pseudocode bugs, complexity mistakes",
        "Experimental Inconsistencies - Results that contradict stated methodology or data that doesn't match claims",
        "Definitional Errors - Incorrect definitions of terms, concepts, or notation misuse",
        "Causal Reasoning Errors - Invalid logical inferences or unsupported causal claims",
    ],
    correctness_context=(
        "Pay special attention to: theorem proofs, algorithm correctness, "
        "complexity analysis, and whether experimental results are consistent "
        "with the stated methodology."
    ),
    scoring_notes=(
        "For CS papers: empirical results without strong baselines are a reason to reject. "
        "Missing ablation studies on core design choices lower the score."
    ),
)
