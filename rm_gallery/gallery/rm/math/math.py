from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult


class MathVerifyReward(BasePointWiseReward):
    """
    Verify mathematical expressions using math_verify library
    """

    name: str = Field(default="math_verify", description="Math verification reward")
    timeout_score: float = Field(default=0.0, description="Score to assign on timeout")

    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Verify mathematical expressions

        Args:
            sample: Data sample containing mathematical content

        Returns:
            RewardResult: Reward result containing verification score
        """
        generated = sample.output[0].answer.content.strip()
        reference = sample.output[0].answer.label.get("reference", "").strip()

        # Use the compute_score function to verify the math
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

        score = 0.0

        # Wrap the ground truth in \boxed{} format for verification
        reference_boxed = "\\boxed{" + reference + "}"

        try:
            score, details = verify_func([reference_boxed], [generated])
        except Exception:
            pass

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=details,
                )
            ],
            extra_data={
                "generated": generated,
                "reference": reference,
                "score": score,
            },
        )
