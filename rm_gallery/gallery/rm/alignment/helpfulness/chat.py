from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

PRINCIPLES = [
    "Address Core Argument/Intent Directly: Prioritize engaging with the user's central claim, perspective, or question explicitly, ensuring responses align with their stated goals or concerns rather than diverging into tangential topics.",
    "Provide Actionable, Context-Specific Guidance: Offer concrete, practical steps or solutions tailored to the user's unique situation, balancing clarity with adaptability to empower informed decisions or actions.",
    "Ensure Factual Accuracy and Contextual Nuance: Correct misconceptions, clarify complexities, and ground responses in precise details or evidence while avoiding oversimplification or speculative interpretations.",
]

SCENARIO = "Chat: Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction."

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance.These principles can serve as supplementary knowledge for your judgment. If you find any of the principles helpful for the current problem, feel free to use them as supplements.
"""


@RewardRegistry.register("chat_listwise_reward")
class ChatListWiseReward(BaseHelpfulnessListWiseReward):
    """Chat: Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction."""

    name: str = Field(default="chat_listwise_reward", description="reward name")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
