from typing import Dict

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.composition import RouterComposition
from rm_gallery.gallery.rm.alignment.base import BaseHelpfulnessListwiseReward
from rm_gallery.gallery.rm.alignment.harmlessness.safety import SafetyListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.focus import FocusListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.math import MathListWiseReward
from rm_gallery.gallery.rm.alignment.helpfulness.precise_if import (
    PreciseIFListWiseReward,
)
from rm_gallery.gallery.rm.alignment.honesty.factuality import FactualityListWiseReward


class RewardBench2Router(RouterComposition):
    """
    RouterComposition subclass implementing category-based routing for reward models

    Attributes:
        router (Dict[str, dict]): Dictionary mapping category identifiers to reward model classes
    """

    # Reward model routing configuration
    # Maps category identifiers to corresponding reward model implementations
    rewards: Dict[str, dict] = {
        "safety": {"cls": SafetyListWiseReward},
        "focus": {"cls": FocusListWiseReward},
        "math": {"cls": MathListWiseReward},
        "factuality": {"cls": FactualityListWiseReward},
        "precis_if": {"cls": PreciseIFListWiseReward},
        "general": {"cls": BaseHelpfulnessListwiseReward},
    }

    def _condition(self, sample: DataSample) -> str:
        """
        Extract routing condition from data sample

        Args:
            sample (DataSample): Input data sample containing category path metadata

        Returns:
            str: Normalized category identifier extracted from the third level of category path,
                or 'general' as fallback value
        """
        # Extract third-level category from path and normalize to lowercase
        # Example: "Safety/Content/Toxicity" -> "toxicity"

        try:
            cond = sample.metadata["raw_data"]["subset"].lower()
        except Exception:
            # Fallback to general reward model when path extraction fails
            cond = "general"

        if cond not in self.rewards:
            cond = "general"
        return cond
