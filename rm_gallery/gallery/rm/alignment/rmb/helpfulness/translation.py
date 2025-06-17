from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Accuracy in Meaning Preservation: Ensure the translation precisely retains the original intent, nuances, and core message without distortion or omission.",
    "Fluency and Natural Flow: Produce translations that sound idiomatic and grammatically correct in the target language, avoiding awkward phrasing or unnatural structures.",
    "Proper Terminology Usage: Use domain-specific or technical terms consistently and accurately to preserve the original text's precision and professionalism.",
]

SCENARIO = "Translation: Converting text from one language to another."


@RewardRegistry.register("translation_listwise_reward")
class TranslationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
