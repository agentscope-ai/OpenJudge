import json
from typing import Any, Dict, List

from examples.train.pairwise.template import PairwiseComparisonTemplate
from rm_gallery.core.train.dataset import BaseTrainDataset
from rm_gallery.gallery.rm.alignment.base import HelpfulnessPairWiseReward


class HelpfulnessPairwiseTrainDataset(BaseTrainDataset):
    """Specialized dataset for pairwise comparison evaluation tasks"""

    def __init__(self, *args, **kwargs):
        self.helpfulness_reward = HelpfulnessPairWiseReward(
            name="helpfulness_pairwise_train",
            template=PairwiseComparisonTemplate,
            examples=self._get_examples(),
            llm=None,
        )

        super().__init__(*args, **kwargs)

    @staticmethod
    def _normalize_row(row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON string fields to python objects"""
        for key in ["input", "output", "metadata"]:
            if key in row_dict and isinstance(row_dict[key], str):
                try:
                    row_dict[key] = json.loads(row_dict[key])
                except json.JSONDecodeError:
                    pass
        return row_dict

    def _get_examples(self) -> List[str]:
        """Get training examples for the pairwise comparison evaluation as a list"""
        return [
            """Example 1:
Query: Explain the concept of machine learning to a beginner.
Response A: Machine learning is a type of artificial intelligence (AI) that allows computers to learn and make decisions without being explicitly programmed for every task. It works by using algorithms that can analyze data, identify patterns, and make predictions or decisions based on what they've learned. For example, when you get movie recommendations on Netflix, that's machine learning at work - the system learns from your viewing history and suggests movies you might like.

Response B: Machine learning is when computers use math to look at data and make guesses. It's like teaching a computer to recognize cats in pictures by showing it thousands of cat photos until it gets good at spotting cats.

<think>
Response A provides a more comprehensive and beginner-friendly explanation. It defines machine learning clearly, explains how it works, and provides a relatable example with Netflix recommendations. Response B is overly simplified and doesn't give enough context or explanation for a beginner to truly understand the concept.
</think>
<preference>A</preference>""",
            """Example 2:
Query: What are the health benefits of regular exercise?
Response A: Exercise is good for you and helps you stay healthy. It makes your heart work better and you feel good.

Response B: Regular exercise provides numerous health benefits including: 1) Cardiovascular health - strengthens the heart muscle and improves circulation, 2) Weight management - helps burn calories and maintain healthy body weight, 3) Mental health - releases endorphins that reduce stress and improve mood, 4) Bone health - weight-bearing exercises strengthen bones and prevent osteoporosis, 5) Immune system - moderate exercise boosts immune function, and 6) Sleep quality - regular activity promotes better sleep patterns. It's recommended to get at least 150 minutes of moderate exercise per week.

<think>
Response B is significantly better than Response A. Response A is very basic and lacks specific information. Response B provides detailed, comprehensive information about multiple health benefits of exercise, includes specific recommendations, and is well-structured with clear points. It's much more informative and helpful.
</think>
<preference>B</preference>""",
        ]

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template with thinking enabled for pairwise comparison evaluation"""
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Construct prompt for pairwise comparison without DataSample conversion"""
        example = self._normalize_row(example)

        # Extract query and two responses
        query = ""
        if isinstance(example.get("input"), list) and len(example["input"]):
            query = example["input"][0].get("content", "")

        responses = []
        if isinstance(example.get("output"), list):
            for out in example["output"][:2]:
                ans = out.get("answer", {})
                if isinstance(ans, dict):
                    responses.append(ans.get("content", ""))

        if len(responses) < 2:
            # fallback to metadata raw_data
            md = example.get("metadata", {})
            if isinstance(md, dict):
                raw = md.get("raw_data", {})
                responses = [raw.get("response_1", ""), raw.get("response_2", "")]

        response_a, response_b = (responses + ["", ""])[:2]

        prompt = PairwiseComparisonTemplate.format(
            desc="Please choose the better response.",
            principles="",
            examples="",
            query=query,
            response_a=response_a,
            response_b=response_b,
            enable_thinking=False,
        )

        return [{"role": "user", "content": prompt}]

    def _extract_ground_truth(self, row_dict: Dict[str, Any]) -> str:
        """Extract ground truth for pairwise comparison evaluation"""
        row_dict = self._normalize_row(row_dict)
        try:
            metadata = row_dict.get("metadata", {})
            if isinstance(metadata, dict):
                return metadata.get("preferred", "tie")
        except:
            return "tie"

    def _get_data_source(self, row_dict: Dict[str, Any]) -> str:
        """Get data source for pairwise comparison evaluation"""
        row_dict = self._normalize_row(row_dict)
        return row_dict.get("source", "helpsteer2_pairwise")
