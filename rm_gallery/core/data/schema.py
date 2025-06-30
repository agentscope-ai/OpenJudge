"""
Core data schema definitions for the reward modeling data pipeline.
Provides structured data models for samples, rewards, and datasets with validation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from rm_gallery.core.model.message import ChatMessage
from rm_gallery.core.reward.schema import RewardDimensionWithScore


class Reward(BaseModel):
    """
    Reward evaluation result for data samples with detailed scoring breakdown.

    Stores both overall score and dimension-specific details for comprehensive
    reward evaluation tracking and analysis.

    Attributes:
        score: Overall aggregated reward score (typically weighted average)
        details: List of individual reward dimensions with their scores
    """

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
    """
    Individual reasoning step in a multi-step process with evaluation metadata.

    Extends ChatMessage to include step-specific labels and reward evaluations.
    Used for tracking intermediate steps in complex reasoning tasks.

    Attributes:
        label: Additional labeling information for the step
        reward: Reward evaluation specific to this step
    """

    label: Optional[Dict[str, Any]] = Field(default={}, description="label")
    reward: Reward = Field(default=Reward(), description="reward")


class DataOutput(BaseModel):
    """
    Output response structure containing the final answer and optional reasoning steps.

    Encapsulates both the final response and any intermediate reasoning steps
    for comprehensive evaluation and analysis.

    Attributes:
        answer: Final response step with complete answer
        steps: Optional list of intermediate reasoning steps
    """

    answer: Step = Field(default=...)
    steps: Optional[List[Step]] = Field(default=None, description="steps")


class DataSample(BaseModel):
    """
    Complete data sample structure for reward modeling training and evaluation.

    Represents a single interaction with input context, multiple possible outputs,
    and associated metadata for comprehensive reward model training.

    Attributes:
        unique_id: Unique identifier for tracking and deduplication
        input: Conversation context as list of chat messages
        output: List of possible responses with evaluations
        task_category: Optional categorization for task-specific analysis
        source: Origin dataset or system that generated this sample
        created_at: Timestamp for temporal tracking
        metadata: Additional context and debugging information
    """

    unique_id: str = Field(..., description="Unique identifier for the data")
    input: List[ChatMessage] = Field(default_factory=list, description="input")
    output: List[DataOutput] = Field(default_factory=list, description="output")
    task_category: Optional[str] = Field(default=None, description="task category")
    source: Optional[str] = Field(default=None, description="source")
    created_at: datetime = Field(default_factory=datetime.now, description="createdAt")
    metadata: Optional[Dict] = Field(default=None, description="metadata")

    def update(self, sample: "DataSample") -> "DataSample":
        """
        Merge another sample's data into this sample for combining evaluations.

        Updates additional_kwargs and reward details from the source sample
        while preserving the original structure.

        Args:
            sample: Source sample to merge data from

        Returns:
            Self with updated data for method chaining
        """
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
    """
    Container for managing collections of data samples with metadata.

    Provides standardized interface for dataset operations including indexing,
    iteration, and serialization for consistent data handling across the pipeline.

    Attributes:
        datasamples: Collection of data samples in the dataset
        name: Human-readable identifier for the dataset
        metadata: Additional information about dataset origin and processing
    """

    datasamples: List[DataSample] = Field(
        default_factory=list, description="List of data items"
    )
    name: str = Field(..., description="dataset name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="metadata")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            Integer count of data samples
        """
        return len(self.datasamples)

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[DataSample, List[DataSample]]:
        """
        Enable index-based and slice-based access to dataset samples.

        Args:
            index: Integer index or slice object for data access

        Returns:
            Single DataSample for integer index, list for slice
        """
        return self.datasamples[index]

    def get_data_samples(self) -> List[DataSample]:
        """
        Retrieve all data samples from the dataset.

        Returns:
            Complete list of DataSample objects in the dataset
        """
        return [data for data in self.datasamples]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataset to dictionary format for serialization.

        Returns:
            Dictionary representation with name, metadata, and serialized samples
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "datasamples": [data.model_dump() for data in self.datasamples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDataSet":
        """
        Create dataset instance from dictionary representation.

        Args:
            data: Dictionary with dataset structure and sample data

        Returns:
            New BaseDataSet instance with restored data
        """
        return cls(
            name=data["name"],
            metadata=data.get("metadata", {}),
            datasamples=[DataSample(**item) for item in data["datasamples"]],
        )

    class Config:
        arbitrary_types_allowed = True
