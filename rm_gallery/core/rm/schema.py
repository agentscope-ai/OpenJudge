from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field


class RewardDimension(BaseModel):
    name: str = Field(default=..., description="name")
    weight: float = Field(default=..., description="weight")
    reason: str = Field(default=..., description="reason")


class RewardDimensionWithScore(RewardDimension):
    """
    Pointwise/Stepwise reward
    """

    score: float = Field(default=..., description="score")


class RewardDimensionWithRank(RewardDimension):
    """
    Listwise/Pointwise reward
    """

    rank: List[float] = Field(default_factory=list, description="rank")

    def __getitem__(self, index: int) -> RewardDimensionWithScore:
        return RewardDimensionWithScore(
            name=self.name,
            weight=self.weight,
            reason=self.reason,
            score=self.rank[index],
        )


T = TypeVar("T", RewardDimensionWithScore, RewardDimensionWithRank)


class RewardResult(BaseModel, Generic[T]):
    name: str = Field(default=..., description="reward module name")
    details: List[T] = Field(default_factory=list, description="reward details")
    extra_data: dict = Field(default_factory=dict, description="extra data")
