from abc import ABC
from src.data.base import BaseData


from abc import abstractmethod
from enum import IntEnum
from typing import Any, TypeVar, List, Callable

from pydantic import BaseModel, Field

from _del.executor.queue import ThreadSafeQueue

T = TypeVar("T", bound="BaseModule")

class BaseModule(BaseModel):


    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...















