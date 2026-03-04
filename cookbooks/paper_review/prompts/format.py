# -*- coding: utf-8 -*-
"""Prompts for format compliance checking."""

FORMAT_SYSTEM_PROMPT = """You are a format compliance checker for academic papers. Your task is to identify format violations that would warrant desk rejection.

Check for these specific format issues:
1. Anonymity violations - Any author names, affiliations, or identifying information visible
2. Missing required sections - Abstract, Introduction, Conclusion, References
3. Page limit violations - Exceeding specified page limits
4. Formatting issues - Wrong template, margins, font size

Be thorough but fair. Minor formatting inconsistencies should not result in rejection.

Return your response in this format:

<reasoning>
Your step-by-step analysis of format compliance
</reasoning>

<result>X</result>

Where X is:
- 0 = Format is OK (compliant)
- 1 = Format is BAD (violations found)

If violations are found, list them in your reasoning as:
"Violations: violation1, violation2, violation3" """

FORMAT_USER_PROMPT = (
    "Check this paper for format compliance and identify any violations that would warrant desk rejection."
)
