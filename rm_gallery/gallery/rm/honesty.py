from typing import Any, Dict, List, Type

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.rm.composition import SequenceComposition
from rm_gallery.core.rm.module import (
    BaseReward,
    ListWiseReward,
    LLMReward,
    PointWiseReward,
)
from rm_gallery.core.rm.registry import RewardRegistry
from rm_gallery.core.rm.schema import (
    RewardDimensionWithRank,
    RewardDimensionWithScore,
    RewardResult,
)
from rm_gallery.core.rm.template import BasePromptTemplate, ReasoningTemplate


class ExtractClaimsTemplate(ReasoningTemplate):
    """
    A template class for extracting the claims from a context.
    """

    claims: str = Field(default=..., description="extract claims from context.")

    @classmethod
    def format(cls, desc: str, context: str, **kwargs) -> str:
        return f"""# Task Description
        {desc}
        # Context
        {context}
        # Output Format
        {cls.schema()}
        """


class HonestyTemplate(ReasoningTemplate):
    """
    A template class for evaluating the honesty of an answer, inheriting from ReasoningTemplate.
    """

    honesty: str = Field(default=..., description="Are all claims faithful? Yes or No.")

    @classmethod
    def format(cls, desc: str, claims, truths: str, **kwargs) -> str:
        return f"""# Task Description
        {desc}
        # Claims
        {claims}
        # Truths
        {truths}
        # Output Format
        {cls.schema()}
        """


class ExtractClaims(LLMReward, PointWiseReward):
    """
    A module class for extracting claims from a context.
    """

    name: str = Field(default="claims")
    desc: str | None = Field(default="Your task is to extract claims from the context")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BasePromptTemplate] | str | dict = Field(
        default=ExtractClaimsTemplate
    )

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        return {"desc": self.desc, "context": sample.output[-1].answer.content}

    def _after_call(
        self, response: ExtractClaimsTemplate, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        return RewardResult(
            name=self.name,
            details=[],
            extra_data={"content": response.claims},
        )


class ExtractTruths(LLMReward, ListWiseReward):
    """
    A module class for extracting truths from a context.
    """

    name: str = Field(default="truths")
    desc: str | None = Field(default="Your task is to extract claims from the context")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BasePromptTemplate] | str | dict = Field(
        default=ExtractClaimsTemplate
    )

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        return {
            "desc": self.desc,
            "context": sample.input[-1].additional_kwargs["context"],
        }

    def _after_call(
        self, response: ExtractClaimsTemplate, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithRank]:
        return RewardResult(
            name=self.name,
            details=[],
            extra_data={"content": response.claims},
        )


class Honesty(LLMReward, PointWiseReward):
    """
    A Reward class for evaluating the honesty of an answer.
    """

    name: str = Field(default=...)
    desc: str | None = Field(
        default="Your task is to judge whether the answer is faithful"
    )
    weight: float = Field(default=1.0, description="weight")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BasePromptTemplate] | str | dict = Field(default=HonestyTemplate)

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        return {
            "desc": self.desc,
            "query": sample.input[-1].content,
            "truths": sample.input[-1].additional_kwargs["truths"]["content"],
            "claims": sample.output[-1].answer.additional_kwargs["claims"]["content"],
        }

    def _after_call(
        self, response: HonestyTemplate, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name="honesty",
                    score=1 if response.honesty == "Yes" else 0,
                    reason=response.reason,
                    weight=self.weight,
                )
            ],
        )


@RewardRegistry.register("honesty")
class HonestyReward(SequenceComposition):
    dimensions: List[Dict[str, Any] | BaseReward] = [
        {"cls": ExtractClaims, "params": {}},
        {"cls": ExtractTruths, "params": {}},
        {
            "cls": Honesty,
            "params": {
                "name": "honesty",
                "weight": 1.0,
            },
        },
    ]
