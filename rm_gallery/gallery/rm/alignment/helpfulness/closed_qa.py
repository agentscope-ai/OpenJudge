from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

SCENARIO = "Closed QA: Search for direct answers to specific questions in given text sources (i.e. given context, given options)."

PRINCIPLES = []
DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""


@RewardRegistry.register("closed_qa_listwise_reward")
class ClosedQAListWiseReward(BaseHelpfulnessListWiseReward):
    """Closed QA: Search for direct answers to specific questions in given text sources (i.e. given context, given options)."""

    name: str = Field(default="closed_qa_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
