from typing import Type

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.rm.module import LLMReward, PointWiseReward
from rm_gallery.core.rm.registry import RewardRegistry
from rm_gallery.core.rm.schema import RewardDimensionWithScore, RewardResult
from rm_gallery.core.rm.template import BasePromptTemplate, ReasoningTemplate


class HelpfulnessTemplate(ReasoningTemplate):
    """
    A template class for evaluating the helpfulness of an answer.
    """

    helpfulness: str = Field(default=..., description="is answer helpful? Yes or No.")

    @classmethod
    def format(cls, desc: str, query: str, answer: str) -> str:
        return f"""# Task Description
        {desc}
        # Answer
        {answer}
        # Output Format
        {cls.schema()}
        """


@RewardRegistry.register("helpfulness")
class HelpfulnessReward(LLMReward, PointWiseReward):
    """
    A reward module class for evaluating the helpfulness of an answer using an LLM (Large Language Model).
    """

    name: str = Field(default=...)
    desc: str | None = Field(
        default="Your task is to judge whether the answer is helpfull"
    )
    weight: float = Field(default=1.0, description="weight")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BasePromptTemplate] | str | dict = Field(default=HelpfulnessTemplate)

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        return {
            "desc": self.desc,
            "query": sample.input[-1].content,
            "answer": sample.output[-1].answer.content,
        }

    def _after_call(
        self, response: HelpfulnessTemplate, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name="helpfulness",
                    score=1 if response.helpfulness == "Yes" else 0,
                    reason=response.reason,
                    weight=self.weight,
                )
            ],
        )
