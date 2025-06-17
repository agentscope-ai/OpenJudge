import re
from typing import List

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult
from rm_gallery.core.utils.tokenizer import get_tokenizer


class AccuracyReward(BasePointWiseReward):
    """
    Calculate accuracy (exact match rate) between generated content and reference answer.

    This reward evaluates if the generated content matches the reference answer exactly.
    A score of 1.0 indicates an exact match, while 0.0 indicates no match.
    """

    name: str = Field(default="accuracy", description="Accuracy reward")

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Calculate accuracy score.

        Args:
            sample: Data sample containing generated content and reference answer

        Returns:
            RewardResult: Reward result containing accuracy score
        """
        generated = sample.output[0].answer.content.strip()
        reference = sample.output[0].answer.label.get("reference", "").strip()

        # Calculate accuracy (1.0 for exact match, 0.0 otherwise)
        accuracy = 1.0 if generated == reference else 0.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=accuracy,
                    reason=f"Generated content {'matches' if accuracy == 1.0 else 'does not match'} reference exactly",
                )
            ],
            extra_data={
                "generated": generated,
                "reference": reference,
                "accuracy": accuracy,
            },
        )


class F1ScoreReward(BasePointWiseReward):
    """
    Calculate F1 score between generated content and reference answer at word level.

    This reward computes precision, recall and F1 score by comparing word overlap
    between generated and reference texts. Uses configurable tokenizer to support
    multilingual content including Chinese and English.
    """

    name: str = Field(default="f1_score", description="F1 score reward")
    tokenizer_type: str = Field(
        default="tiktoken",
        description="Tokenizer type: 'tiktoken', 'jieba', or 'simple'",
    )
    encoding_name: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name (for tiktoken tokenizer)",
    )
    chinese_only: bool = Field(
        default=False,
        description="Whether to keep only Chinese characters (for jieba tokenizer)",
    )

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize tokenizer
        self._tokenizer = get_tokenizer(
            tokenizer_type=self.tokenizer_type,
            encoding_name=self.encoding_name,
            chinese_only=self.chinese_only,
        )

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Calculate F1 score.

        Args:
            sample: Data sample containing generated content and reference answer

        Returns:
            RewardResult: Reward result containing F1 score
        """
        generated = sample.output[0].answer.content.strip()
        reference = sample.output[0].answer.label.get("reference", "").strip()

        # Tokenize using unified tokenizer
        generated_preprocessed = self._tokenizer.preprocess_text(
            generated, to_lower=True
        )
        reference_preprocessed = self._tokenizer.preprocess_text(
            reference, to_lower=True
        )

        generated_tokens = set(self._tokenizer.tokenize(generated_preprocessed))
        reference_tokens = set(self._tokenizer.tokenize(reference_preprocessed))

        # Calculate precision, recall and F1 score
        if not generated_tokens and not reference_tokens:
            precision = recall = f1 = 1.0
        elif not generated_tokens or not reference_tokens:
            precision = recall = f1 = 0.0
        else:
            intersection = generated_tokens.intersection(reference_tokens)
            precision = len(intersection) / len(generated_tokens)
            recall = len(intersection) / len(reference_tokens)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=f1,
                    reason=f"F1 score: {f1:.3f} (Precision: {precision:.3f}, Recall: {recall:.3f})",
                )
            ],
            extra_data={
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "generated_tokens": list(generated_tokens),
                "reference_tokens": list(reference_tokens),
                "tokenizer_type": self.tokenizer_type,
                "tokenizer_name": self._tokenizer.name,
            },
        )


class RougeReward(BasePointWiseReward):
    """
    ROUGE-L similarity evaluation
    """

    name: str = Field(default="rouge", description="ROUGE similarity reward")

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Calculate ROUGE-L score

        Args:
            sample: Data sample containing generated content

        Returns:
            RewardResult: Reward result containing ROUGE-L score
        """
        generated = sample.output[0].answer.content.strip().lower()
        reference = sample.output[0].answer.label.get("reference", "").strip().lower()

        # Tokenization
        generated_tokens = generated.split()
        reference_tokens = reference.split()

        if not generated_tokens and not reference_tokens:
            rouge_l = 1.0
        elif not generated_tokens or not reference_tokens:
            rouge_l = 0.0
        else:
            # Calculate LCS length
            lcs_len = self._lcs_length(generated_tokens, reference_tokens)

            # Calculate ROUGE-L
            if len(generated_tokens) == 0 or len(reference_tokens) == 0:
                rouge_l = 0.0
            else:
                precision = lcs_len / len(generated_tokens)
                recall = lcs_len / len(reference_tokens)
                rouge_l = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=rouge_l,
                    reason=f"ROUGE-L score: {rouge_l:.3f}",
                )
            ],
            extra_data={
                "rouge_l": rouge_l,
                "generated_length": len(generated_tokens),
                "reference_length": len(reference_tokens),
                "lcs_length": self._lcs_length(generated_tokens, reference_tokens)
                if generated_tokens and reference_tokens
                else 0,
            },
        )


class NumberAccuracyReward(BasePointWiseReward):
    """
    Check numerical calculation accuracy.

    This reward verifies if the numbers in the generated content match
    the numbers in the reference content within a specified tolerance.
    """

    name: str = Field(default="number_accuracy", description="Number accuracy reward")
    tolerance: float = Field(default=1e-6, description="Numerical comparison tolerance")

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        # Match integers and floating point numbers
        number_pattern = r"-?\d+\.?\d*"
        numbers = re.findall(number_pattern, text)
        return [float(n) for n in numbers if n]

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Check numerical accuracy.

        Args:
            sample: Data sample containing numerical values

        Returns:
            RewardResult: Reward result containing numerical accuracy score
        """
        generated = sample.output[0].answer.content
        reference = sample.output[0].answer.label.get("reference", "")

        generated_numbers = self._extract_numbers(generated)
        reference_numbers = self._extract_numbers(reference)

        if not reference_numbers:
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=0.0,
                        reason="No reference numbers to compare",
                    )
                ],
                extra_data={
                    "generated_numbers": generated_numbers,
                    "reference_numbers": reference_numbers,
                },
            )

        if not generated_numbers:
            return RewardResult(
                name=self.name,
                details=[
                    RewardDimensionWithScore(
                        name=self.name,
                        score=0.0,
                        reason="No numbers found in generated content",
                    )
                ],
                extra_data={
                    "generated_numbers": generated_numbers,
                    "reference_numbers": reference_numbers,
                },
            )

        # Compare numbers (match in order)
        correct = 0
        total = min(len(generated_numbers), len(reference_numbers))

        for i in range(total):
            if abs(generated_numbers[i] - reference_numbers[i]) <= self.tolerance:
                correct += 1

        accuracy = correct / len(reference_numbers) if reference_numbers else 0.0

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=accuracy,
                    reason=f"Number accuracy: {correct}/{len(reference_numbers)} numbers correct",
                )
            ],
            extra_data={
                "accuracy": accuracy,
                "correct_numbers": correct,
                "total_reference_numbers": len(reference_numbers),
                "generated_numbers": generated_numbers,
                "reference_numbers": reference_numbers,
            },
        )
