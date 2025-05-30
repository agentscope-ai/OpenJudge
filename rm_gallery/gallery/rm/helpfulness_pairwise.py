from rm_gallery.core.rm.module import PrinciplePairwiseReward
from rm_gallery.core.rm.registry import RewardRegistry


@RewardRegistry.register("helpfulness_pairwise")
class HelpfulnessPairwise(PrinciplePairwiseReward):
    def _get_principle_and_desc(self):
        self.task_desc = """
You are a professional expert in helpfulness evaluation.
You will be provided with an query and two answers based on that query.
Your task is to judge which answer is better using the given {principles} which can help you decide whether an answer meets the definition of helpfulness.
"""

        self.principles = [
            "Structure: Organize information logically",
            "Clarity: Ensure clear communication",
            "Accuracy: Ensure factual correctness",
            "Conciseness: Be brief yet informative",
            "Relevance: Focus on related information",
            "Engagement: Make content interactive",
            "Detail: Provide comprehensive specifics",
            "Practicality: Offer actionable advice",
            "Comprehensiveness: Cover all aspects",
            "Safety: Emphasize protective measures",
        ]
