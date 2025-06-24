from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Prioritize contextual accuracy over superficial criteria: Emphasize deeper cultural, industry, or functional context (e.g., K-pop's globalized identity, genre fluidity) rather than relying on surface-level attributes like language, nationality, or isolated errors.",
    "Systematic and evidence-based analysis: Apply predefined criteria methodically (e.g., all classification rules, hierarchical ordering) while grounding judgments in explicit textual or structural evidence to ensure objectivity and comprehensiveness.",
    "Maintain structural and categorical precision: Ensure strict alignment with classification frameworks, hierarchical sequences, or formatting requirements (e.g., Stellaris terminology, ecological order) without conflating categories or omitting required elements.",
]


SCENARIO = "Classification: Entails assigning predefined categories or labels to text based on its content."

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance.These principles can serve as supplementary knowledge for your judgment. If you find any of the principles helpful for the current problem, feel free to use them as supplements.
"""


@RewardRegistry.register("classification_listwise_reward")
class ClassificationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
