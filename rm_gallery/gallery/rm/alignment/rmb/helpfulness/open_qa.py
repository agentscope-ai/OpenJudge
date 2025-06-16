from typing import List

from pydantic import Field

from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward

SCENARIO = "Open QA: Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions."
PRINCIPLES = [
    "Accuracy and Factual Correctness: Prioritize precise, verified information aligned with credible sources or established knowledge to avoid errors or misinformation.",
    "Clarity and Structured Organization: Present information logically with clear formatting (e.g., headings, bullet points) to enhance readability and comprehension.",
    "Avoidance of Speculation and Unverified Claims: Refrain from unconfirmed theories or assumptions, emphasizing evidence-based explanations and transparency about limitations.",
    "Direct Relevance to the Query: Focus on addressing the user’s specific needs without tangential or redundant details, ensuring practical utility.",
    "Use of Authoritative Sources and Evidence: Anchor claims in reputable references (e.g., academic studies, official guidelines) to validate reliability and depth.",
]


@RewardRegistry.register("open_qa_listwise_reward")
class OpenQAListWiseReward(BaseHelpfulnessListwiseReward):
    scenario: str = Field(default=SCENARIO, description="assistant scenario")
    principles: List[str] = Field(default=PRINCIPLES)
