from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Closed QA: Search for direct answers to specific questions in given text sources (i.e. given context, given options)."

PRINCIPLES = [
    "Accuracy in Answering: Prioritize factually correct responses directly addressing the question with verified information.",
    "Logical and Structured Reasoning: Provide clear, step-by-step explanations without gaps or contradictions to ensure transparency.",
    "Adherence to Task Requirements: Strictly follow explicit instructions (e.g., formatting, scope, answer choices) to meet the task's expectations.",
    "Avoidance of Errors: Prevent computational, logical, or interpretive mistakes that compromise the validity of the response.",
]


@RewardRegistry.register("closed_qa_listwise_reward")
class ClosedQAListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
