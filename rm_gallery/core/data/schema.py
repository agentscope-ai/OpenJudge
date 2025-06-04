from datetime import datetime
from typing import Any, Dict, List, Optional, Self, Union

from pydantic import BaseModel, Field

from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.rm.schema import RewardDimensionWithScore


class Reward(BaseModel):
    """Reward for the data sample"""

    score: float = Field(default=0.0, description="score")
    details: List[RewardDimensionWithScore] = Field(
        default_factory=list, description="details"
    )

    # @property
    # def total_score(self) -> float:
    #     """Get the total score of the reward"""
    #     return sum(
    #         dimension.score * dimension.weight for dimension in self.details
    #     ) / sum(dimension.weight for dimension in self.details)


class Step(ChatMessage):
    """Step in the process"""

    label: Optional[Dict[str, Any]] = Field(default={}, description="label")
    reward: Reward = Field(default=Reward(), description="reward")


class DataOutput(BaseModel):
    """Data output"""

    answer: Step = Field(default=...)
    steps: Optional[List[Step]] = Field(default=None, description="steps")


class DataSample(BaseModel):
    """Data sample"""

    unique_id: str = Field(..., description="Unique identifier for the data")
    input: List[ChatMessage] = Field(default_factory=list, description="input")
    output: List[DataOutput] = Field(default_factory=list, description="output")
    task_category: Optional[str] = Field(default=None, description="task category")
    source: Optional[str] = Field(default=None, description="source")
    created_at: datetime = Field(default_factory=datetime.now, description="createdAt")
    metadata: Optional[Dict] = Field(default=None, description="metadata")

    def update(self, sample: "DataSample") -> Self:
        self.input[-1].additional_kwargs.update(sample.input[-1].additional_kwargs)
        for i, output in enumerate(self.output):
            output.answer.additional_kwargs.update(
                sample.output[i].answer.additional_kwargs
            )
            output.answer.reward.details.extend(sample.output[i].answer.reward.details)

            if output.steps:
                for j, step in output.steps:
                    step.additional_kwargs.update(
                        sample.output[i].steps[j].additional_kwargs
                    )
                    step.reward.details.extend(sample.output[i].steps[j].reward.details)
        return self

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class BaseDataSet(BaseModel):
    """Base dataset class for managing collections of data"""

    datas: List[DataSample] = Field(
        default_factory=list, description="List of data items"
    )
    name: str = Field(..., description="dataset name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="metadata")

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.datas)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[DataSample, List[DataSample]]:
        """Support index-based access to data"""
        return self.datas[index]

    def get_data_samples(self) -> List[DataSample]:
        """Get all evaluation samples from the dataset"""
        return [data for data in self.datas]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format"""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "datas": [data.dict() for data in self.datas],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDataSet":
        """Create dataset from dictionary format"""
        return cls(
            name=data["name"],
            metadata=data.get("metadata", {}),
            datas=[DataSample(**item) for item in data["datas"]],
        )

    class Config:
        arbitrary_types_allowed = True
