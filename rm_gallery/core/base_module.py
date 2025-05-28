from abc import abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseModule")


class BaseModule(BaseModel):
    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...
