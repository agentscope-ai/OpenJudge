from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

SCENARIO = "Reasoning: Involves processing and analyzing text to draw inferences, make predictions, or solve problems, requiring an understanding of underlying concepts and relationships within the text."
PRINCIPLES = []
DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""


@RewardRegistry.register("reasoning_listwise_reward")
class ReasoningListWiseReward(BaseHelpfulnessListWiseReward):
    """Reasoning: Involves processing and analyzing text to draw inferences, make predictions, or solve problems, requiring an understanding of underlying concepts and relationships within the text."""

    name: str = Field(default="reasoning_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
