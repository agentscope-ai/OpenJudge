from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."
PRINCIPLES = [
    "Adherence to Constraints and Comprehensive Coverage: Prioritize compliance with specified requirements, ensuring all essential elements are included to meet the scope of the task without missing critical details.",
    "Structural Coherence and Organization: Maintain logical progression and clear organization to enhance readability and alignment with the task’s purpose."
    "Originality Within Boundaries: Balance creative innovation with adherence to factual/technical accuracy or genre conventions to avoid clichés or formulaic outputs",
]


@RewardRegistry.register("generation_listwise_reward")
class GenerationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
