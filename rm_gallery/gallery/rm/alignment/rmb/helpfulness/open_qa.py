from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Open QA: Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions."
PRINCIPLES = [
    "Accuracy and Reliance on Authoritative Sources: Ensure that all information presented is precise and verified, using reputable and evidence-based references to maintain factual correctness and avoid misinformation."
    "Clarity and Organization: Facilitate understanding and readability by structuring content logically and with clear formatting, such as headings and bullet points."
    "Focus and Relevance: Address the user's specific query directly and avoid speculation, ensuring all information is evidence-based and practically useful without tangential details."
]


@RewardRegistry.register("open_qa_listwise_reward")
class OpenQAListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
