from typing import Dict, List, Type
from pydantic import Field
from src.data.data_schema import EvaluationContext
from src.rm.base import BaseParser, LLMParser, Rule
from src.rm.schema import Score
from src.rm.template import BaseTemplate, RuleScoreTemplate


class BaseScorer(BaseParser):
    weight: Dict[str, float] | float = Field(default=...)


class LLMScorer(LLMParser):
    rules: List[Rule] = Field(default=...)
    output_schema: Type[EvaluationContext] = Field(default=Score)
    template: Type[BaseTemplate] = RuleScoreTemplate

    def format(self, **kwargs) -> str:
        return self.template.format(desc=self.desc, rules=self.rules, output_schema=self.output_schema, **kwargs)
