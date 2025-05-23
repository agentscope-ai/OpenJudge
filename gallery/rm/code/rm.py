from pathlib import Path
import re
from typing import Dict, List, Self, Type
from pydantic import Field
from src.rm._context import RMContext
from src.rm.module import LLMRewardModule, RuleRewardModule
from src.rm.schema import LLMLLMModuleOutput, Rule
from src.rm.template import BaseTemplate, EvaluationTemplate


class TaskTemplate(BaseTemplate):
    @classmethod
    def format(cls, desc: str, context: str, output: Type[LLMLLMModuleOutput]) -> str:
        prompt = """# Task Description
{desc}
# Context
{context}
# Output Format
{output}
"""
        return prompt.format(desc=desc, context=context, output=output.format())


class EvaluationTemplate(BaseTemplate):
    @classmethod
    def format(
        cls,
        desc: str,
        principles: str,
        actual_output: str,
        output: Type[LLMLLMModuleOutput],
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

        prompt += f"# Evaluation Format\n{output.format()}\n"
        return prompt


class RuleScoreTemplate(BaseTemplate):
    @classmethod
    def format(cls, desc: str, context: str, rules: List[Rule], output: Type[LLMLLMModuleOutput]) -> str:
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
{output}
"""
        return prompt.format(desc=desc, context=context, rules=rules_str, output=output.format())


class Claims(LLMLLMModuleOutput):
    claims: str = Field(default=..., description="claims in the context")


class ViolatedPrinciples(LLMLLMModuleOutput):
    principles: List[int] = Field(default=..., description="indices of the violated principles, a list")

    @classmethod
    def parse(cls, text: str) -> Self:
        pattern = r'<([^>]+)>(.*?)</\1>'
        matches = re.findall(pattern, text)
        contents = {match[0]: match[1] for match in matches}
        contents["principles"] = eval(contents["principles"])
        return cls(**contents)


class Score(LLMLLMModuleOutput):
    score: int | float = Field(default=..., description="score by rule")


class BestOfN(LLMLLMModuleOutput):
    best: int = Field(default=..., description="the index of best answer")


class DataRM(RuleRewardModule):
    def run(self, sample: DataSample):
        """
        call pointwise parser
        """
        # TODO parallel
        sample.input.context[self.name] = {
            "context": f"""query: {input.data.history[-1].content}
retrieve context: {input.data.contexts[-1].context}
""" 
        }
        for output in sample.output:
            output.context[self.name] = {
                "actual_output": output.answer.content
            }
        


class LLMRM(LLMRewardModule):
    principles: List[str] | str | None = Field(default=None, description="evaluation priciples")
    examples: List[str] | None = Field(default=None, description="evaluation examples")
    template: Type[BaseTemplate] | str | Dict = Field(default=EvaluationTemplate, description="genreal template of llm evaluation prompt")
    rules: List[Rule] | None = Field(default=None, description="score rules")

    def format(self, **kwargs) -> str:
        return self.template.format(desc=self.desc, principles=self.principles, examples=self.examples, output=self.output, rules=self.rules, **kwargs)
    
    def _run(self, **kwargs) -> RMContext:
        query = self.format(**kwargs)
        response = self.llm.simple_chat(query=query)
        output = self.output.parse(response)
        return output


class LengtRM(RuleRewardModule):
    """
    rule parser: check the length of actual_output
    """
    def _run(self, actual_output: str):
        return dict(actual_output_length=len(actual_output))


class FormatRM(RuleRewardModule):
    """
    rule parser: check the format of actual_output
    """
    def _run(self, actual_output: str) -> dict:
        for title in ['intro', 'body', 'conclusion']:
            if f"# {title}" not in actual_output:
                return {
                    "format_standardized": False
                }
        return {"format_standardized": True}
    
