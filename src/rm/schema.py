import re
from typing import List, Self
from pydantic import BaseModel, Field
from src.data.data_schema import EvaluationContext


class LLMEvaluationContext(EvaluationContext):
    reason: str = Field(default=..., description="analysis process", alias="think")


class Claims(LLMEvaluationContext):
    claims: str = Field(default=..., description="claims in the context")


class ViolatedPrinciples(LLMEvaluationContext):
    principles: List[int] = Field(default=..., description="indices of the violated principles, a list")

    @classmethod
    def parse(cls, text: str) -> Self:
        pattern = r'<([^>]+)>(.*?)</\1>'
        matches = re.findall(pattern, text)
        contents = {match[0]: match[1] for match in matches}
        contents["principles"] = eval(contents["principles"])
        return cls(**contents)


class Score(LLMEvaluationContext):
    score: int | float = Field(default=..., description="score by rule")


class BestOfN(LLMEvaluationContext):
    best: int = Field(default=..., description="the index of best answer")
