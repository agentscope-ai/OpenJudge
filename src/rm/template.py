from typing import List, Type

from pydantic import BaseModel
from src.rm.base import BaseTemplate, Rule
from src.rm.schema import EvaluationContext


class ParserTemplate(BaseTemplate):
    @classmethod
    def format(cls, desc: str, context: str, output_schema: Type[EvaluationContext]) -> str:
        prompt = """# Task Description
{desc}
# Context
{context}
# Output Format
{output_schema}
"""
        return prompt.format(desc=desc, context=context, output_schema=output_schema.format())


class EvaluationTemplate(BaseTemplate):
    @classmethod
    def format(
        cls,
        desc: str,
        principles: str,
        actual_output: str,
        output_schema: Type[EvaluationContext],
        examples: List[str] | None = None,
        context: str | None = None,
        expected_output: str | None = None,
        rules: List[Rule] | None = None,
    ) -> str:
        prompt = f"""# Task Description
{desc}
# Principles
{principles}
"""
        if rules:
            rules_str = ""
            for idx, rule in enumerate(rules):
                rules_str += f"## Rule {idx + 1}\n description: {rule.desc} \n score: {rule.score} \n"
            prompt += f"# Rules\n{rules_str}\n"

        if examples:
            prompt += f"# Examples\n{examples}\n"

        if context:
            prompt += f"# Context\n{context}\n"

        prompt += f"# Actual Output\n{actual_output}\n"

        if expected_output:
            prompt += f"# Expected Output\n{expected_output}\n"

        prompt += f"# Evaluation Format\n{output_schema.format()}\n"
        return prompt


class RuleScoreTemplate(BaseTemplate):
    @classmethod
    def format(cls, desc: str, context: str, rules: List[Rule], output_schema: Type[EvaluationContext]) -> str:
        rules_str = ""
        for idx, rule in enumerate(rules):
            rules_str += f"## Rule {idx + 1}\n description: {rule.desc} \n score: {rule.score} \n"

        prompt = """# Task Description
{desc}
# Rules
{rules}
# Context
{context}
# Output Format
{output_schema}
"""
        return prompt.format(desc=desc, context=context, rules=rules_str, output_schema=output_schema.format())
