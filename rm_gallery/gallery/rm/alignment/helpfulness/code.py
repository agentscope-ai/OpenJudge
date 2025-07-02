from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""
SCENARIO = "Code: Involves generating, understanding, or modifying programming language code within text."
PRINCIPLES = []


@RewardRegistry.register("code_listwise_reward")
class CodeListWiseReward(BaseHelpfulnessListWiseReward):
    """Code: Involves generating, understanding, or modifying programming language code within text."""

    name: str = Field(default="code_listwise_reward")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
