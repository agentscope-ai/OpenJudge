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
    "Direct Relevance to Core Query: Prioritize completions that explicitly address the specific question, task, or scenario posed in the query without introducing tangential concepts, unnecessary details, or unrelated analysis."
]


@RewardRegistry.register("focus_listwise_reward")
class FocusListWiseReward(BaseHelpfulnessListWiseReward):
    """Focus: Detects high-quality, on-topic answers to general user queries"""

    name: str = Field(default="focus_listwise_reward")
    desc: str = Field(default=DESC)
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
