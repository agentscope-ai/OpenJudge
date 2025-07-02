from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

# PRINCIPLES = [
#     "Refusing harmful requests directly: The assistant must immediately decline prompts involving harmful, unethical, or illegal actions (e.g., distributing proprietary code, enabling privacy violations, or facilitating dangerous activities) to prevent misuse and uphold ethical/legal compliance.",
#     "Role Consistency: Maintain the assigned character's traits, voice, and thematic alignment throughout interactions to ensure authenticity and immersion.",
#     "Adherence to Instructional Guidelines: Strictly follow all specified structural, formatting, and procedural requirements in the prompt to meet functional expectations.",
#     "Interactive and Immersive Engagement: Prioritize dynamic user involvement, contextual richness, and narrative tension to sustain engagement while respecting role boundaries.",
# ]
PRINCIPLES = [
    "Character and Contextual Fidelity: Prioritize maintaining the assigned character's persona, motivations, and world-building consistency while strictly adhering to the scenario's established rules, terminology, and thematic boundaries to ensure immersive authenticity."
]
SCENARIO = "Role Playing: Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles."

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
I will also provide you with a set of principles, listed under the heading #Principles. These principles are ordered from highest to lowest importance.These principles can serve as supplementary knowledge for your judgment, though not necessarily required. First, think independently. Use these principles only when unsure about certain answers, selecting specific ones based on the questions and answers.
"""


@RewardRegistry.register("role_playing_listwise_reward")
class RolePlayingListWiseReward(BaseHelpfulnessListWiseReward):
    """Role Playing: Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles."""

    name: str = Field(default="role_playing_listwise_reward")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC, description="task description")
