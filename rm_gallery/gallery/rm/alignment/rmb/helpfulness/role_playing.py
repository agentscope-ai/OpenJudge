from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Adherence to Role/Persona: Maintain strict consistency with the assigned character, voice, or thematic framework across all responses to ensure immersion and authenticity.",
    "Structured Response Format: Prioritize explicit formatting rules (e.g., markdown, triple backticks, dialogue-only sections) to align with user-defined structural expectations.",
    "User Engagement Through Interaction: Actively involve the user via prompts, tasks, or dynamic choices to drive narrative progression or educational clarity.",
    "Thematic/Contextual Integrity: Anchor responses in the established scenario’s lore, stakes, or technical requirements to preserve coherence and relevance.",
    "Precision Over Extraneous Elements: Eliminate redundant, vague, or off-topic content to maintain conciseness, clarity, and alignment with core objectives.",
]
SCENARIO = "Role Playing: Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles."


@RewardRegistry.register("role_playing_listwise_reward")
class RolePlayingListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
