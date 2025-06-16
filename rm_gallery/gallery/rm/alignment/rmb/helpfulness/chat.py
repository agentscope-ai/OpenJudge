from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Contextual Relevance and Understanding: Prioritize responses that directly align with the user's specific context, intent, and conversational history to maintain meaningful and targeted interactions.",
    "Clarity and Specificity: Deliver precise, concrete details while avoiding ambiguity or vague language to ensure the user receives actionable and understandable information.",
    "Structured Organization: Present information in a logical, coherent sequence or format to enhance readability and facilitate the user's ability to follow and apply the content.",
    "Balanced and Honest Communication: Maintain transparency about limitations, acknowledge multiple perspectives, and avoid overpromising to foster trust and realistic expectations.",
    "User-Centric Focus with Instruction Adherence: Prioritize the user's explicit needs, emotional state, and formatting requirements while avoiding deviations from their stated goals or constraints.",
]

SCENARIO = "Chat: Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction."


@RewardRegistry.register("chat_listwise_reward")
class ChatListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
