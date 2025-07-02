from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance. You must check each candidate answer in turn to see if it violates any principle, and provide reasons for any violations you find. These reasons should be used as references for ranking the answers.
You may organize your reasoning as you see fit, but keep your thought process as concise as possible.
"""
SCENARIO = ""
PRINCIPLES = [
    "Strict Adherence to Explicit Formatting and Structural Requirements: Prioritize exact compliance with all specified formatting, structural, and technical constraints (e.g., punctuation, indentation, bullet points, word counts) as the primary criterion for evaluating completions.",
    "Clarity, Logical Progression, and Thematic Consistency: Ensure content is coherent, logically structured, and maintains alignment with the scenario's core premise, fulfilling implicit demands for depth, relevance, and narrative or analytical consistency.",
]


@RewardRegistry.register("precise_if_listwise_reward")
class PreciseIFListWiseReward(BaseHelpfulnessListWiseReward):
    """Precise Instruction Following : Follows precise instructions, such as ‘Answer without the letter u’."""

    name: str = Field(default="precise_if_listwise_reward")
    desc: str = Field(default=DESC)
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
