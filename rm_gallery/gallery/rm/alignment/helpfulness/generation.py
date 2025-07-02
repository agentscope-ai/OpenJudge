from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

SCENARIO = "Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."
PRINCIPLES = [
    "Adherence to Instructional Specificity: Prioritize addressing all explicit requirements (e.g., format, content scope, tone) with precise alignment to ensure completeness and fidelity to the task's intent.",
    "Depth and Originality in Content: Deliver nuanced, actionable insights or creative elements that exceed generic responses through specific examples, contextual relevance, and imaginative elaboration.",
    "Structural Coherence and Logical Flow: Maintain organized progression (e.g., clear hierarchy, thematic sequencing) to enhance readability while avoiding contradictions or deviations from established frameworks.",
]

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance.These principles can serve as supplementary knowledge for your judgment. If you find any of the principles helpful for the current problem, feel free to use them as supplements.
"""


@RewardRegistry.register("generation_listwise_reward")
class GenerationListWiseReward(BaseHelpfulnessListWiseReward):
    """Generation: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."""

    name: str = Field(default="generation_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
