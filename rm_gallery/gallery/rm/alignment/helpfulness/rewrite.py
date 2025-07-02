from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

PRINCIPLES = []

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""

SCENARIO = """Rewrite: the assistant aims to modifies existing text to alter its style while preserving the original information and intent."""


@RewardRegistry.register("rewrite_listwise_reward")
class RewriteListWiseReward(BaseHelpfulnessListWiseReward):
    """Rewrite: the assistant aims to modifies existing text to alter its style while preserving the original information and intent."""

    name: str = Field(default="rewrite_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
