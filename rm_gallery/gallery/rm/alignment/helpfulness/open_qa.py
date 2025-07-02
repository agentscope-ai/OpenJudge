from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

SCENARIO = "Open QA: Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions."
PRINCIPLES = []
DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""


@RewardRegistry.register("open_qa_listwise_reward")
class OpenQAListWiseReward(BaseHelpfulnessListWiseReward):
    """Open QA: Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions."""

    name: str = Field(default="open_qa_listwise_reward")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
