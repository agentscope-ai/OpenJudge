from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Comprehensive Coverage of Core Content: A superior summary captures all critical elements, themes, and details central to the source material without omitting key information.",
    "Avoidance of Irrelevant or Tangential Information: Focuses exclusively on the primary subject, eliminating extraneous details that distract from the core narrative or argument.",
    "Logical Structure and Coherence: Information is organized in a clear, hierarchical, or chronological sequence to ensure readability and logical progression of ideas.",
]

SCENARIO = "Summarization: The text is compressed into a short form, retaining the main information, which is divided into extraction (directly selected from the original text) and production (rewriting the information)."


@RewardRegistry.register("summarization_listwise_reward")
class SummarizationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
