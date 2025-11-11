"""
Base Module Classes

Define base abstract classes for all RM-Gallery modules.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseModule(BaseModel, ABC):
    """
    Base class for all RM-Gallery modules

    This class combines Pydantic's BaseModel with ABC to provide:
    - Data validation through Pydantic
    - Abstract method enforcement through ABC
    - Common interface for all modules through the run method

    All concrete modules should inherit from this class and implement the run method.

    Attributes:
        model_config: Pydantic configuration allowing arbitrary types

    Example:
        >>> class MyModule(BaseModule):
        ...     name: str = "my_module"
        ...
        ...     def run(self, **kwargs) -> Any:
        ...         # Implement specific logic
        ...         return {"result": "success"}
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Run the module with given parameters

        This is the core method that all subclasses must implement.

        Args:
            **kwargs: Arbitrary keyword arguments specific to each module

        Returns:
            Any: Module execution result

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement run()")
