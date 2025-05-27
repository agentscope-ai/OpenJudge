
from typing import Dict, Iterable, List, Self
from pydantic import BaseModel, Field

from src.data.data_schema import DataSample, InputSample, OutputSample, Step
from src.rm.schema import LLMModuleOutput


class StepContext(Step):
    context: Dict[str, LLMModuleOutput | dict] = Field(default={})


class OutputContext(OutputSample):
    context: Dict[str, LLMModuleOutput | dict] = Field(default={})
    steps: List[StepContext] = Field(default=[])


class InputContext(InputSample):
    context: Dict[str, LLMModuleOutput | dict] = Field(default={})


class RMContext(DataSample):
    input: InputContext = Field(default=...)
    output: List[OutputContext] = Field(default=...)

    @classmethod
    def from_sample(cls, sample: DataSample) -> Self:
        """
        Creates a new instance of the current class based on a DataSample object.

        Parameters:
        - sample (DataSample): A DataSample instance containing the data for the new instance.

        Returns:
        - Self: A new instance of the current class initialized with the provided data.
        """
        return cls(
            **sample.model_dump()
        )

class ModuleLevel(str, Enum):
    POINTWISE = "POINTWISE"
    LISTWISE = "LISTWISE"
    STEPWISE = "STEPWISE"


class InputVar(BaseModel):
    name: str = Field(default=...)
    path: str = Field(default=...)
    level: ModuleLevel = Field(default=ModuleLevel.POINTWISE)
    default: Any | None = Field(default=None)
