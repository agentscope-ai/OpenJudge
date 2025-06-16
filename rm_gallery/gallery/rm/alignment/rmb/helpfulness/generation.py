from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."
PRINCIPLES = [
    "Adherence to Instruction Constraints: Prioritize exact compliance with all specified requirements (e.g., formatting, keyword limits, structural rules) to avoid disqualification or misalignment with user intent.",
    "Comprehensive Coverage: Ensure inclusion of all essential elements (e.g., concepts, themes, plot points) to fulfill the scope of the task without omitting critical details.",
    "Structural Coherence: Maintain logical progression and clear organization to enhance readability, navigability, and alignment with the task’s purpose.",
    "Originality Within Boundaries: Balance creative innovation with adherence to factual/technical accuracy or genre conventions to avoid clichés or formulaic outputs.",
    "Emotional and Thematic Resonance: Integrate human elements, symbolic depth, or thematic consistency to elevate engagement and connect with the audience’s intellect or emotions.",
]


@RewardRegistry.register("generation_listwise_reward")
class GenerationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
