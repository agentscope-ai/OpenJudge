from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Brainstorming: Generating text to come up with new ideas or solutions, with an emphasis on creativity and driving thinking."

PRINCIPLES = [
    "Relevance to Core Request: Ensure ideas directly address the explicit requirements and intent of the query without extraneous details.",
    "Balanced Speculative Creativity: Maintain imaginative freedom while tethering concepts to logical extrapolation or feasibility.",
]


@RewardRegistry.register("brainstorming_listwise_reward")
class BrainstormingListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
