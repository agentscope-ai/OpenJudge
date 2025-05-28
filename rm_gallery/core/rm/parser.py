from pathlib import Path
from typing import Dict, List, Type
from pydantic import Field
from src.data.data_schema import EvaluationSample
from src.task.base import LLMTask, Rule, RuleTask
from src.task.template import BaseTemplate, EvaluationTemplate


class DataTask(RuleTask):
    def run_pointwise(self, sample: EvaluationSample):
        """
        call pointwise parser
        """
        # TODO parallel
        sample.evaluation_contexts[self.name] = {
            "context": f"""query: {sample.input[-1].content}
retrieve context: {sample.contexts[-1].context}

""" 
        }

        for output in sample.outputs:
            output.evaluation_contexts[self.name] = {
                "actual_output": output.content
            }


class LLMEvaluation(LLMTask):
    principles: List[str] | str | None = Field(default=None, description="evaluation priciples")
    examples: List[str] | None = Field(default=None, description="evaluation examples")
    template: Type[BaseTemplate] | str | Dict = Field(default=EvaluationTemplate, description="genreal template of llm evaluation prompt")
    rules: List[Rule] | None = Field(default=None, description="score rules")

    def format(self, **kwargs) -> str:
        return self.template.format(desc=self.desc, principles=self.principles, examples=self.examples, output_schema=self.output_schema, rules=self.rules, **kwargs)


class LengthParser(RuleTask):
    """
    rule parser: check the length of actual_output
    """
    def _run(self, actual_output: str):
        return dict(actual_output_length=len(actual_output))


class FormatParser(RuleTask):
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
    
