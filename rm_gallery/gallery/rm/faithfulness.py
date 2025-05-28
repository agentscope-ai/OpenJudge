from concurrent.futures import ThreadPoolExecutor
from typing import List, Type

from pydantic import Field

from rm_gallery.core.data.schema import DataOutput, DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.rm.module import (
    BaseRewardModule,
    ListModule,
    LLMModule,
    PointModule,
)
from rm_gallery.core.rm.template import BaseTemplate, ReasoningTemplate
from rm_gallery.core.utils.registry import RewardRegistry


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


class FaithfulnessTemplate(ReasoningTemplate):
    """
    A template class for evaluating the faithfulness of an answer, inheriting from ReasoningTemplate.
    """

    faithfulness: str = Field(
        default=..., description="Are all claims faithful? Yes or No."
    )

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


class ExtractClaims(LLMModule, PointModule):
    """
    A module class for extracting claims from a context.
    """

    name: str = Field(default=...)
    desc: str | None = Field(default="Your task is to extract claims from the context")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BaseTemplate] | str | dict = Field(default=ExtractClaimsTemplate)

    def _before_call(
        self, input: List[ChatMessage], output: DataOutput, **kwargs
    ) -> dict:
        return {"desc": self.desc, "context": output.answer.content}

    def _after_call(
        self, response: ExtractClaimsTemplate, output: DataOutput, **kwargs
    ):
        output.answer.additional_kwargs["claims"] = response.claims


class ExtractTruths(LLMModule, ListModule):
    """
    A module class for extracting truths from a context.
    """

    name: str = Field(default=...)
    desc: str | None = Field(default="Your task is to extract claims from the context")
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BaseTemplate] | str | dict = Field(default=ExtractClaimsTemplate)

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        return {
            "desc": self.desc,
            "context": sample.input[-1].additional_kwargs["context"],
        }

    def _after_call(
        self, response: ExtractClaimsTemplate, sample: DataSample, **kwargs
    ):
        sample.input[-1].additional_kwargs["truths"] = response.claims


class Faithfulness(LLMModule, PointModule):
    """
    A Reward class for evaluating the faithfulness of an answer.
    """

    name: str = Field(default=...)
    desc: str | None = Field(
        default="Your task is to judge whether the answer is faithful"
    )
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BaseTemplate] | str | dict = Field(default=FaithfulnessTemplate)

    def _before_call(
        self, input: List[ChatMessage], output: DataOutput, **kwargs
    ) -> dict:
        return {
            "desc": self.desc,
            "query": input[-1].content,
            "truths": input[-1].additional_kwargs["truths"],
            "claims": output.answer.additional_kwargs["claims"],
        }

    def _after_call(self, response: FaithfulnessTemplate, output: DataOutput, **kwargs):
        output.answer.reward.set_reward(
            dimension="faithfulness",
            value=1 if response.faithfulness == "Yes" else 0,
            reason=response.reason,
        )


@RewardRegistry.register("faithfulness")
class FaithfulnessReward(LLMModule, BaseRewardModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._extract_claims_module = ExtractClaims(
            name="extract_claims",
            llm=self.llm,
        )
        self._extract_truths_module = ExtractTruths(
            name="extract_truths",
            llm=self.llm,
        )
        self._faithfulness_module = Faithfulness(
            name="faithfulness",
            llm=self.llm,
        )

    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None):
        if thread_pool:
            future_claim = thread_pool.submit(
                self._extract_claims_module.run, sample=sample, thread_pool=thread_pool
            )
            future_truth = thread_pool.submit(
                self._extract_truths_module.run, sample=sample, thread_pool=thread_pool
            )

            future_claim.result()
            future_truth.result()
            self._faithfulness_module.run(sample=sample, thread_pool=thread_pool)
        else:
            self._extract_claims_module.run(sample=sample)
            self._extract_truths_module.run(sample=sample)
            self._faithfulness_module.run(sample=sample)
