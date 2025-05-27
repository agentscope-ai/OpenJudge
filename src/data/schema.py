from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union, Tuple
from datetime import datetime
from enum import Enum
from abc import abstractmethod

from src.base_module import BaseModule


class MessageRole(str, Enum):
    """Message role"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class DataModuleType(Enum):
    """data module type"""
    BUILD = "builder"
    LOAD = "loader"
    GENERATE = "generator"
    PROCESS = "processor"
    ANNOTATION = "annotator"

class Reward(BaseModel):
    """Reward for the data sample"""
    total_score: float = Field(..., description="totalScore")
    rewards_detail: List[Dict[str, Any]] = Field(default_factory=list,description="list of rewards")
    
    def get_reward_info(self, dimension: str) -> Dict[str, Any]:
        """Get reward value and reason for a specific dimension"""
        for reward in self.rewards_detail:
            if reward["dimension"] == dimension:
                return {
                    "value": reward.get("value", 0.0),
                    "reason": reward.get("reason"),
                }
        return {"value": 0.0, "reason": None}
    
    def set_reward(self, dimension: str, value: Any, reason: str = None) -> None:
        """Set reward for a specific dimension with optional reason"""
        reward_dict = {"dimension": dimension, "value": value}
        if reason:
            reward_dict["reason"] = reason
            
        # Update existing reward or add new one
        for reward in self.rewards_detail:
            if reward["dimension"] == dimension:
                reward.update(reward_dict)
                break
        else:
            self.rewards_detail.append(reward_dict)
        
        # Update total score if this is the first reward
        if len(self.rewards_detail) == 1:
            self.total_score = value

class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    name: Optional[str] = None
    content: Optional[Any] = ""  # support str for llm or list of dict for multi-modal model
    additional_kwargs: dict = Field(default_factory=dict)
    time_created: datetime = Field(default_factory=lambda: datetime.now(),
                                   description="Timestamp marking the message creation time")

    def __str__(self) -> str:
        return f"{self.time_created.strftime('%Y-%m-%d %H:%M:%S')} {self.role.value}: {self.content}"

    @staticmethod
    def convert_from_strings(self, messages: List[str], system_message: str) -> str:
        """
        turn vanilla strings to structure messages for fast debugging
        """
        result_messages = [ChatMessage(role=MessageRole.SYSTEM, content=system_message), ]

        toggle_roles = [MessageRole.USER, MessageRole.ASSISTANT]
        for index, msg in enumerate(messages):
            result_messages.append(ChatMessage(role=toggle_roles[index%2], content=msg))
            
        return result_messages

    @staticmethod
    def convert_to_strings(self, messages: List["ChatMessage"]) -> Tuple[List[str], str]:
        """
        turn structure messages to vanilla strings for fast debugging
        """
        vanilla_messages = []
        system_message = ""

        for index, msg in enumerate(messages):
            if msg.role == MessageRole.SYSTEM:
                system_message += msg.content
            else:
                vanilla_messages.append(msg.content)

        return vanilla_messages, system_message
    

class Step(ChatMessage):
    """Step in the process"""
    label: Optional[Dict[str, Any]] = Field(default=None,description="label")
    reward: Optional[Reward] = Field(default=None,description="reward")

class DataOutput(BaseModel):
    """Data output"""
    answer: Step = Field(default=...)
    steps: Optional[List[Step]] = Field(default=None,description="steps")

class DataSample(BaseModel):
    """Data sample"""
    unique_id: str = Field(..., description="Unique identifier for the data")
    input: List[ChatMessage] = Field(default_factory=list,description="input")
    output: List[DataOutput] = Field(default_factory=list,description="output")
    label: Optional[Dict[str, Any]] = Field(default=None,description="label")
    reward: Optional[Reward] = Field(default=None,description="reward")
    domain: Optional[str] = Field(default=None,description="domain")
    source: Optional[str] = Field(default=None,description="source")
    created_at: datetime = Field(default_factory=datetime.now,description="createdAt")
    metadata: Optional[Dict] = Field(default=None,description="metadata")

    # update data
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid field: {key}")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BaseDataSet(BaseModel):
    """Base dataset class for managing collections of data"""
    datas: List[DataSample] = Field(default_factory=list, description="List of data items")
    name: str = Field(..., description="dataset name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="metadata")

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.datas)

    def __getitem__(self, index: Union[int, slice]) -> Union[DataSample, List[DataSample]]:
        """Support index-based access to data"""
        return self.datas[index]

    def get_data_samples(self) -> List[DataSample]:
        """Get all evaluation samples from the dataset"""
        return [data.data_sample for data in self.datas]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format"""
        return {
            "name": self.name,
            "metadata": self.metadata,
            "datas": [data.dict() for data in self.datas]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseDataSet':
        """Create dataset from dictionary format"""
        return cls(
            name=data["name"],
            metadata=data.get("metadata", {}),
            datas=[DataSample(**item) for item in data["datas"]]
        )

    class Config:
        arbitrary_types_allowed = True