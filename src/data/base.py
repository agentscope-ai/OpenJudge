from typing import Sequence, Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
from datetime import datetime

from src.data.data_schema import EvaluationSample


class BaseData(BaseModel):
    """Base data class, parent class for all specific data types"""
    unique_id: str = Field(..., description="Unique identifier for the data")
    evaluation_sample: EvaluationSample = Field(
        default_factory=EvaluationSample,
        serialization_alias="evaluationSample",
    )
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    extra_metadata: Optional[Dict] = Field(
        default=None, serialization_alias="extraMetadata"
    )

    # 更新数据的接口
    def update(self, other, **kwargs):
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



    def update(self,other):
        pass


class BaseDataSet(BaseModel):
    """Base dataset class for managing collections of data"""
    datas: Sequence[BaseData] = Field(default_factory=list, description="List of data items")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    version: str = Field("1.0.0", description="Dataset version")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.datas)

    def __getitem__(self, index: Union[int, slice]) -> Union[BaseData, List[BaseData]]:
        """Support index-based access to data"""
        return self.datas[index]

    def evaluate(self, dimension: str) -> Dict[str, float]:
        """Evaluate the dataset on a specific dimension"""
        pass

    def get_evaluation_samples(self) -> List[EvaluationSample]:
        """Get all evaluation samples from the dataset"""
        return [data.samples for data in self.datas]

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "extra_metadata": self.extra_metadata,
            "datas": [data.dict() for data in self.datas]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseDataSet':
        """Create dataset from dictionary format"""
        return cls(
            name=data["name"],
            description=data.get("description"),
            version=data.get("version", "1.0.0"),
            extra_metadata=data.get("extra_metadata", {}),
            datas=[BaseData(**item) for item in data["datas"]]
        )

    class Config:
        arbitrary_types_allowed = True