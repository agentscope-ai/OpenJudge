from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field


class RewardDimension(BaseModel):
    """
    Base class for reward dimensions containing common attributes.

    Attributes:
        name (str): Identifier name for the reward dimension
        reason (str): Explanation of how the reward value was determined
    """

    name: str = Field(default=..., description="name")
    # weight: float = Field(default=..., description="weight")
    reason: str = Field(default=..., description="reason")


class RewardDimensionWithScore(RewardDimension):
    """
    Pointwise/Stepwise reward dimension with a numerical score.

    Attributes:
        score (float): Numerical value representing the reward magnitude
    """

    score: float = Field(default=..., description="score")


class RewardDimensionWithRank(RewardDimension):
    """
    ListWise/Pointwise reward dimension with ranking values.

    Attributes:
        rank (List[float]): Collection of ranking scores for different positions

    Methods:
        __getitem__: Returns a scored reward dimension for a specific rank position
    """

    rank: List[float] = Field(default_factory=list, description="rank")

    def __getitem__(self, index: int) -> RewardDimensionWithScore:
        """
        Access a specific position's reward information.

        :param index: Position in the ranking list to retrieve
        :type index: int
        :returns: Reward information with score for the specified position
        :rtype: RewardDimensionWithScore
        """
        return RewardDimensionWithScore(
            name=self.name,
            # weight=self.weight,
            reason=self.reason,
            score=self.rank[index],
        )


# Type variable for generic programming, allows handling both score-based and rank-based rewards
T = TypeVar("T", RewardDimensionWithScore, RewardDimensionWithRank)


class RewardResult(BaseModel, Generic[T]):
    """
    Container for reward calculation results with generic type support.

    Attributes:
        name (str): Identifier of the reward module that generated this result
        details (List[T]): Collection of detailed reward information items
        extra_data (dict): Additional metadata or context information
    """

    name: str = Field(default=..., description="reward module name")
    details: List[T] = Field(default_factory=list, description="reward details")
    extra_data: dict = Field(default_factory=dict, description="extra data")
