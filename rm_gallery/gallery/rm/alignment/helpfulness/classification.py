from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

PRINCIPLES = []


SCENARIO = "Classification: Entails assigning predefined categories or labels to text based on its content."

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""


@RewardRegistry.register("classification_listwise_reward")
class ClassificationListWiseReward(BaseHelpfulnessListWiseReward):
    """Classification: Entails assigning predefined categories or labels to text based on its content."""

    name: str = Field(
        default="classification_listwise_reward", description="reward name"
    )
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
