import re
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Self, Any

class Reward(BaseModel):
    total_score: float = Field(..., description="total score")
    rewards_detail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of dimension-specific rewards including reason"
    )
    
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


class DataInfo(BaseModel):
    domain: str = Field(..., description="domain")
    source: Optional[str] = Field(
        default=None, serialization_alias="source"
    )


class EvaluationContext(BaseModel):
    @classmethod
    def parse(cls, text: str) -> Self:
        pattern = r'<([^>]+)>(.*?)</\1>'
        matches = re.findall(pattern, text)
        contents = {match[0]: match[1] for match in matches}
        return cls(**contents)

    @classmethod
    def format(cls) -> str:
        schema_str = ""
        for key, property in cls.model_json_schema(by_alias=True)["properties"].items():
            schema_str += f"<{key}>{property["description"]}</{key}>"
        return schema_str


class ContentDict(BaseModel):
    """Content with its corresponding reward"""
    role: str = Field(..., description="role")
    content: Optional[str] = Field(default=None, description="Output content")
    content_label: Optional[str] = Field(default=None, description="content label")
    rewards: Optional[Reward] = Field(default=None, description="Reward for this output")
    # TODO: add extra schema
    extra_metadata: Optional[Dict] = Field(default=None, serialization_alias="extraMetadata")
    evaluation_contexts: Dict[str, EvaluationContext] = Field(default={})


class ContextDict(BaseModel):
    """Context dict"""
    context_type: str = Field(..., description="context type")
    context: Optional[str] = Field(default=None, description="context")
    extra_metadata: Optional[Dict] = Field(default=None, serialization_alias="extraMetadata")


class EvaluationSample(BaseModel):
    input: List[ContentDict] = Field(
        default_factory=list,
        serialization_alias="input"
    )
    outputs: List[ContentDict] = Field(
        default_factory=list,
        serialization_alias="outputs"
    )
    contexts: List[ContextDict] = Field(
        default_factory=list,
        serialization_alias="contexts"
    )
    data_info: Optional[DataInfo] = Field(
        default=None, serialization_alias="dataInfo"
    )
    extra_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="extraMetadata"
    )
    evaluation_contexts: Dict[str, EvaluationContext] = Field(default={})
    # TODO: support custom schema