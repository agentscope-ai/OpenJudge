from detoxify import Detoxify
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BasePointWiseReward
from rm_gallery.core.reward.registry import RewardRegistry
from rm_gallery.core.reward.schema import RewardDimensionWithScore, RewardResult


@RewardRegistry.register("detoxify_reward")
class DetoxifyReward(BasePointWiseReward):
    """Detoxify-based reward model for measuring text toxicity."""

    name: str = Field(default="detoxify", description="Name of the reward module")
    model_name: str = Field(
        default="unbiased", description="Name of the Detoxify model to use"
    )

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = Detoxify(self.model_name)
        return self._model

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """
        Evaluate text toxicity using Detoxify model.

        Args:
            sample: Input data sample containing text to evaluate
            **kwargs: Additional implementation-specific parameters

        Returns:
            RewardResult: Computed reward metrics and metadata
        """
        try:
            # Get text from sample
            text = sample.output[0] if sample.output else sample.input

            if not text:
                raise ValueError("No text provided for evaluation")

            # Get model predictions
            predictions = self.model.predict(text)

            # Convert toxicity score to reward (higher = less toxic)
            toxicity_score = predictions["toxicity"]
            reward_score = 1.0 - toxicity_score  # Invert score so higher is better

            # Create reward dimension
            reward_dimension = RewardDimensionWithScore(
                name="detoxify",
                score=reward_score,
                reason=f"Text toxicity score: {toxicity_score:.2f}. Higher reward indicates less toxic content.",
            )

            return RewardResult(name=self.name, details=[reward_dimension])

        except Exception as e:
            logger.error(f"Error in Detoxify evaluation: {str(e)}")
            return RewardResult(name=self.name, details=[])
