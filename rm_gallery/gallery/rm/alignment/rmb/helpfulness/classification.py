from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Classification Correctness: Accurately categorize based on the question.",
    "Logical Coherence: Explicitly connect textual evidence to conclusions through clear, rational reasoning without contradictions or unsupported leaps.",
]

SCENARIO = "Classification: Entails assigning predefined categories or labels to text based on its content."


@RewardRegistry.register("classification_listwise_reward")
class ClassificationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
