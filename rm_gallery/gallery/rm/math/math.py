from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult


@RewardRegistry.register("math_verify_reward")
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

        score = 0.0
        reason = "Verification failed or timed out"

        try:
            # Parse the reference (gold) answer
            # Use both LatexExtractionConfig and ExprExtractionConfig for maximum flexibility
            gold_parsed = parse(
                reference,
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            )

            # Parse the generated answer
            pred_parsed = parse(
                generated,
                extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            )

            # If both parsing succeeded and we have results
            if gold_parsed and pred_parsed:
                # Use the first parsed result from each
                gold_expr = gold_parsed[0]
                pred_expr = pred_parsed[0]

                # Verify if they match
                if verify(gold_expr, pred_expr):
                    score = 1.0
                    reason = f"({gold_parsed}, {pred_parsed})"
                else:
                    score = 0.0
                    reason = f"({gold_parsed}, {pred_parsed})"
            else:
                score = 0.0
                reason = f"Parsing failed - gold: {gold_parsed}, pred: {pred_parsed}"

        except Exception as e:
            score = self.timeout_score
            reason = f"Exception occurred: {str(e)}"

        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name,
                    score=score,
                    reason=str(reason),
                )
            ],
            extra_data={
                "generated": generated,
                "reference": reference,
                "score": score,
            },
        )
