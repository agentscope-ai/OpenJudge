from typing import List, Type

from pydantic import Field

from rm_gallery.core.data.schema import DataOutput
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.rm.module import LLMModule, PointModule
from rm_gallery.core.rm.template import BaseTemplate, ReasoningTemplate
from rm_gallery.core.utils.registry import RewardRegistry


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
class HelpfulnessReward(LLMModule, PointModule):
    """
    A reward module class for evaluating the helpfulness of an answer using an LLM (Large Language Model).
    """

    name: str = Field(default=...)
    desc: str | None = Field(
        default="Your task is to judge whether the answer is helpfull"
    )
    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BaseTemplate] | str | dict = Field(default=HelpfulnessTemplate)

    def _before_call(
        self, input: List[ChatMessage], output: DataOutput, **kwargs
    ) -> dict:
        return {
            "desc": self.desc,
            "query": input[-1].content,
            "answer": output.answer.content,
        }

    def _after_call(self, response: HelpfulnessTemplate, output: DataOutput, **kwargs):
        output.answer.reward.set_reward(
            dimension="helpfulness",
            value=1 if response.helpfulness == "Yes" else 0,
            reason=response.reason,
        )
