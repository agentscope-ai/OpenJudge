from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

PRINCIPLES = [
    "Accuracy in Translation: Faithfully convey the original text's meaning, intent, and nuances without distortion, omission, or addition.",
    "Contextual Appropriateness: Preserve the original context, tone, and purpose while adapting to target language conventions and specified formatting requirements.",
]

SCENARIO = "Translation: Converting text from one language to another."

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance.These principles can serve as supplementary knowledge for your judgment. If you find any of the principles helpful for the current problem, feel free to use them as supplements.If all answers meet all principles, you can judge and choose one answer by yourself.
"""


@RewardRegistry.register("translation_listwise_reward")
class TranslationListWiseReward(BaseHelpfulnessListWiseReward):
    """Translation: Converting text from one language to another."""

    name: str = Field(default="translation_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC, description="task description")
