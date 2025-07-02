from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListWiseReward

DESC = """
Your role is that of a professional evaluation expert. I will provide you with a question and several candidate answers. Your task is to select the single best answer from the candidates.
"""
SCENARIO = "Brainstorming: Generating text to come up with new ideas or solutions, with an emphasis on creativity and driving thinking."

PRINCIPLES = [
    "Creative Relevance and Contextual Alignment: Prioritize completions that balance novel ideas with direct ties to the scenario's core context, ensuring ideas are both imaginative and grounded in the specific problem or theme.",
    "Practical Feasibility and Actionable Detail: Favor completions that offer concrete, implementable solutions or insights, avoiding abstract or overly speculative suggestions that lack real-world applicability.",
    "Structural Coherence and Logical Organization: Prefer completions that present ideas in a clear, logically sequenced framework (e.g., categorized sections, step-by-step processes) to enhance readability and development potential.",
]


@RewardRegistry.register("brainstorming_listwise_reward")
class BrainstormingListWiseReward(BaseHelpfulnessListWiseReward):
    """Brainstorming: Generating text to come up with new ideas or solutions, with an emphasis on creativity and driving thinking."""

    name: str = Field(default="brainstorming_listwise_reward")
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
    desc: str = Field(default=DESC)
