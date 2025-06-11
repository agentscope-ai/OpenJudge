from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import (
    BaseListWisePrincipleReward,
    BasePointWisePrincipleReward,
)
from rm_gallery.core.reward.registry import RewardRegistry

REWRITE_PRINCIPLES = [
    "Creative Writing: Involves the generation of original content.",
    "Professional Content Generation: Involves the generation of professional content.",
]

SCENARIO = """Rewriting: Creating new textual content, from articles to stories, with an emphasis on originality and creativity."""


@RewardRegistry.register("rewrite_pointwise")
class RewritePointWiseReward(BasePointWisePrincipleReward):
    scenario: str = Field(default=SCENARIO)
    principles: List[str] = Field(default=REWRITE_PRINCIPLES)


@RewardRegistry.register("rewrite_listwise")
class RewriteListWiseReward(BaseListWisePrincipleReward):
    scenario: str = Field(default=SCENARIO)
    principles: List[str] = Field(default=REWRITE_PRINCIPLES)
