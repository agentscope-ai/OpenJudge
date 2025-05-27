from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from loguru import logger
import importlib

from .base import BaseData
from .data_processor import DataJuicerOperator
from src.base_module import BaseModule

T = TypeVar('T', bound=BaseData)

class BaseOperator(BaseModule):
    """
    Base class for all data processing operators
    """
    pass

class Operator(Generic[T], BaseOperator):
    """
    Base class for all data processing operators
    """
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        
    @abstractmethod
    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process the entire dataset
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class OperatorFactory(BaseModule):
    """
    Factory class for creating operators from configuration
    """
    _operator_registry: Dict[str, Callable[[Dict[str, Any]], Operator]] = {}
    _registry_logged: bool = False

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator for registering operator creation functions.
        
        Args:
            operator_type: The type identifier for the operator
            
        Returns:
            A decorator function that registers the operator creation function
        """
        def decorator(func: Callable[[Dict[str, Any]], Operator]) -> Callable:
            cls._operator_registry[name] = func
            return func
        return decorator

    @classmethod
    def create_operator(cls, operator_config: Dict[str, Any]) -> Operator:
        """
        Create an operator from configuration
        """
        op_type = operator_config.get('type')
        name = operator_config.get('name', op_type)
        config = operator_config.get('config', {})
            
        logger.info(f"Creating operator with name: {name}")
        
        # Check registry first
        if name in cls._operator_registry:
            return cls._operator_registry[name](operator_config)
        
        # Only handle data_juicer operators
        if op_type != 'data_juicer':
            raise ValueError(f"Unknown operator type: {op_type}")
            
        module_path = 'data_juicer.ops.filter'
        class_name = lambda n: ''.join(word.capitalize() for word in n.split('_'))
        
        operator_name = class_name(name)
        operator_module = importlib.import_module(f"{module_path}.{name.lower()}")
        operator_class = getattr(operator_module, operator_name)
        return DataJuicerOperator(name=operator_name, juicer_op_class=operator_class, config=config)