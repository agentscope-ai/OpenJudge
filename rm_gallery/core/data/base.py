from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from rm_gallery.core.base_module import BaseModule
from rm_gallery.core.data.schema import BaseDataSet, DataSample


class BaseData(BaseModel):
    """Base data class, parent class for all specific data types"""
    unique_id: str = Field(..., description="Unique identifier for the data")
    evaluation_sample: DataSample = Field(
        default_factory=DataSample,
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

    def get_module_info(self) -> Dict[str, Any]:
        """get module info"""
        config_dict = self.config.model_dump() if self.config else None
        return {
            "type": self.module_type.value,
            "name": self.name,
            "config": config_dict,
        }
