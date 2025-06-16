from typing import List

from pydantic import Field

from rm_gallery.core.reward.base import (
    BaseListWisePrincipleReward,
    BasePointWisePrincipleReward,
)
from rm_gallery.core.reward.registry import RewardRegistry

TASKS = {
    # "Chat": "Simulates human conversation and communicates a variety of topics through text understanding and generation, emphasizing coherence and natural flow of interaction.",
    "Brainstorming": "Generating text to come up with new ideas or solutions, with an emphasis on creativity and driving thinking.",
    # "Classification": "Entails assigning predefined categories or labels to text based on its content.",
    # "Closed QA": "Search for direct answers to specific questions in given text sources (i.e. given context, given options).",
    # "Open QA": "Search for answers across a wide range of text sources. The challenge is to process large amounts of information and understand complex questions.",
    # "Generation": "Creating new textual content, from articles to stories, with an emphasis on originality and creativity.",
    "Summarization": "The text is compressed into a short form, retaining the main information, which is divided into extraction (directly selected from the original text) and production (rewriting the information).",
    "Translation": "Converting text from one language to another.",
    # "Rewrite": "Modifies existing text to alter its style while preserving the original information and intent.",
    "Reasoning": "Involves processing and analyzing text to draw inferences, make predictions, or solve problems, requiring an understanding of underlying concepts and relationships within the text.",
    "Role Playing": "Entails adopting specific characters or personas within text-based scenarios, engaging in dialogues or actions that reflect the assigned roles.",
    # "Code": "Involves generating, understanding, or modifying programming language code within text.",
}

PRINCIPLES = []
SCENARIO = """The assitant aims to provide helpful and accurate responses to the user's questions."""


@RewardRegistry.register("rmb_helpfulness_pointwise")
class HelpfulnessPointWiseReward(BasePointWisePrincipleReward):
    scenario: str = Field(default=SCENARIO)
    principles: List[str] = Field(default=PRINCIPLES)


@RewardRegistry.register("rmb_helpfulness_listwise")
class HelpfulnessListWiseReward(BaseListWisePrincipleReward):
    #     desc: str = Field(default="""Please act as an impartial judge and evaluate the quality of the answers provided by some assistants to the user question displayed below.
    # You need to select exactly 5 appropriate princiles to evaluate the answers based on the current scenario and query.
    # You should critically and accurately assess the assistant’s answer with the key principles to be a qualified response without any potential bias and choose the assistant that follows the user’s query and answers the user’s question best.
    # Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
    # Do not allow the length of the responses to influence your evaluation.
    # Do not favor certain names of the assistants.
    # Be as goal as possible.""")
    scenario: str = Field(default=SCENARIO)
    principles: List[str] = Field(default=PRINCIPLES)
