from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

PRINCIPLES = [
    "Evidence-Based Specificity: Prioritize completions that identify concrete textual examples (e.g., repeated phrases, grammatical patterns) to ground conclusions in observable data rather than abstract claims.",
    "Logical Coherence: Favor completions that explicitly connect textual evidence to conclusions through clear, rational reasoning without contradictions or unsupported leaps.",
    "Nuanced Interpretation: Prefer completions that acknowledge both AI-like and human-like traits in text, avoiding binary distinctions and exploring subtleties like mimicry vs. inherent characteristics.",
    "Multidimensional Analysis: Select completions that evaluate multiple facets of text (e.g., structure, tone, factual accuracy, stylistic quirks) for holistic assessment rather than isolated features.",
    "Clarity and Structure: Choose completions organized with logical flow (e.g., cause-effect chains, numbered arguments) that enhance readability and persuasive impact through deliberate presentation.",
]

SCENARIO = "Classification: Entails assigning predefined categories or labels to text based on its content."


@RewardRegistry.register("classification_listwise_reward")
class ClassificationListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
