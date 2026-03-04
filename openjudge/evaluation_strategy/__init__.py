"""Evaluation strategies"""

from .average_evaluation_strategy import AverageEvaluationStrategy
from .base_evaluation_strategy import BaseEvaluationStrategy
from .direct_evaluation_strategy import DirectEvaluationStrategy
from .voting_evaluation_strategy import (
    CLOSEST_TO_MEAN,
    MAX,
    MIN,
    VotingEvaluationStrategy,
)

__all__ = [
    "BaseEvaluationStrategy",
    "CLOSEST_TO_MEAN",
    "DirectEvaluationStrategy",
    "MAX",
    "MIN",
    "VotingEvaluationStrategy",
    "AverageEvaluationStrategy",
]
