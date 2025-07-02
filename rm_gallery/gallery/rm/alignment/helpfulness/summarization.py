from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

PRINCIPLES = [
    "Comprehensive Coverage of Core Content: A superior summary captures all critical elements, themes, and details central to the source material without omitting key information.",
    "Avoidance of Irrelevant or Tangential Information: Focuses exclusively on the primary subject, eliminating extraneous details that distract from the core narrative or argument.",
    "Logical Structure and Coherence: Information is organized in a clear, hierarchical, or chronological sequence to ensure readability and logical progression of ideas.",
]

SCENARIO = "Summarization: The text is compressed into a short form, retaining the main information, which is divided into extraction (directly selected from the original text) and production (rewriting the information)."

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any principle, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.
"""


@RewardRegistry.register("summarization_listwise_reward")
class SummarizationListWiseReward(BaseHelpfulnessListWiseReward):
    """Summarization: The text is compressed into a short form, retaining the main information, which is divided into extraction (directly selected from the original text) and production (rewriting the information)."""

    name: str = Field(
        default="summarization_listwise_reward", description="reward name"
    )
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC, description="task description")
