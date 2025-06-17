from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Code: Involves generating, understanding, or modifying programming language code within text."
PRINCIPLES = [
    "Technical Accuracy in Implementation: Ensure strict adherence to language-specific syntax, APIs, and platform conventions to avoid logical or runtime errors.",
    "Clarity in Communication and Structure: Provide step-by-step explanations, well-documented code, and structured outputs to enhance readability and maintainability.",
]


@RewardRegistry.register("code_listwise_reward")
class CodeListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
