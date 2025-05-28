from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, TypeVar, List, Callable
from pydantic import BaseModel, Field


T = TypeVar("T", bound="BaseModule")


class BaseModule(BaseModel):

    @abstractmethod
    def run(self, **kwargs) -> Any:
        ...















