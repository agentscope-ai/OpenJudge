import importlib
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import DataSample

T = TypeVar("T", bound=DataSample)


class BaseOperator(BaseModule, Generic[T]):
    """Base class for all data processing operators"""

    name: str = Field(..., description="operator name")
    config: Dict[str, Any] = Field(default_factory=dict, description="operator config")

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(name=name, config=config or {}, **kwargs)

    @abstractmethod
    def process_dataset(self, items: List[T]) -> List[T]:
        """Process the entire dataset"""
        pass

    def run(self, **kwargs):
        """Run method implementation for operator"""
        items = kwargs.get("items", [])
        return self.process_dataset(items)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class OperatorFactory:
    """Factory for creating operators from configuration"""

    _operator_registry: Dict[str, Callable[[Dict[str, Any]], BaseOperator]] = {}
    _external_operators: Dict[str, type] = {}

    # Operator type mapping
    _operator_types = {"filter": "filter", "group": "group", "map": "map"}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator for registering operator creation functions or classes"""

        def decorator(func_or_class):
            # Check if it's a class (subclass of BaseOperator)
            if isinstance(func_or_class, type) and issubclass(
                func_or_class, BaseOperator
            ):
                # Create a factory function for the class
                def class_factory(operator_config: Dict[str, Any]) -> BaseOperator:
                    op_name = operator_config.get("name", name)
                    config = operator_config.get("config", {})
                    return func_or_class(name=op_name, config=config)

                cls._operator_registry[name] = class_factory
                return func_or_class
            else:
                # It's a function, register as-is
                cls._operator_registry[name] = func_or_class
                return func_or_class

        return decorator

    @classmethod
    def create_operator(cls, operator_config: Dict[str, Any]) -> BaseOperator:
        """Create an operator from configuration"""
        op_type = operator_config.get("type")
        name = operator_config.get("name", op_type)
        config = operator_config.get("config", {})

        # Check registry first
        if name in cls._operator_registry:
            return cls._operator_registry[name](operator_config)

        # Handle built-in operator types
        if op_type in cls._operator_types:
            return RegisteredOperator(name=name, operator_type=op_type, config=config)
        elif op_type == "data_juicer":
            return cls._create_data_juicer_filter_operator(name, config)
        else:
            raise ValueError(f"Unknown operator type: {op_type}")

    @classmethod
    def _create_data_juicer_filter_operator(
        cls, name: str, config: Dict[str, Any]
    ) -> BaseOperator:
        """Create data_juicer operator"""
        try:
            # Import data_juicer filter module
            import data_juicer.ops.filter as dj_filters

            # Convert snake_case name to PascalCase class name
            class_name = "".join(word.capitalize() for word in name.split("_"))

            # Try to get the operator class from data_juicer.ops.filter
            if hasattr(dj_filters, class_name):
                operator_class = getattr(dj_filters, class_name)
                return DataJuicerOperator(
                    name=class_name, juicer_op_class=operator_class, config=config
                )
            else:
                # Fallback: try to import from specific module (for backward compatibility)
                module_path = "data_juicer.ops.filter"
                operator_module = importlib.import_module(
                    f"{module_path}.{name.lower()}"
                )
                operator_class = getattr(operator_module, class_name)
                return DataJuicerOperator(
                    name=class_name, juicer_op_class=operator_class, config=config
                )

        except ImportError as e:
            raise ImportError(
                f"Failed to import data_juicer operator '{name}': {e}. "
                f"Please ensure py-data-juicer is installed: pip install py-data-juicer"
            )
        except AttributeError as e:
            raise AttributeError(
                f"Data_juicer operator '{class_name}' not found. "
                f"Available operators can be found in data_juicer.ops.filter module. Error: {e}"
            )


class RegisteredOperator(BaseOperator[T]):
    """Operator that uses the operator registry"""

    operator_type: str = Field(..., description="operator type")

    def __init__(
        self,
        name: str,
        operator_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            name=name, config=config, operator_type=operator_type, **kwargs
        )

    def process_dataset(self, items: List[T]) -> List[T]:
        try:
            if self.name in OperatorFactory._operator_registry:
                operator = OperatorFactory._operator_registry[self.name](
                    {
                        "type": self.operator_type,
                        "name": self.name,
                        "config": self.config,
                    }
                )
                return operator.process_dataset(items)

            logger.warning(f"No registered operator found for name: {self.name}")
            return items
        except Exception as e:
            logger.error(
                f"Error in {self.operator_type} operation {self.name}: {str(e)}"
            )
            return items


class DataJuicerOperator(BaseOperator[T]):
    """Adapter for data-juicer operators"""

    juicer_op: Any = Field(..., description="Data Juicer operator instance")

    def __init__(
        self,
        name: str,
        juicer_op_class: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        juicer_op = juicer_op_class(**config) if config else juicer_op_class()
        super().__init__(name=name, config=config, juicer_op=juicer_op, **kwargs)

    def process_dataset(self, items: List[T]) -> List[T]:
        """Process dataset using data-juicer operators"""
        try:
            all_texts = []
            text_to_item_indices = {}

            for i, item in enumerate(items):
                # Extract texts from input history
                if item.input and item.input:
                    for input_item in item.input:
                        if input_item.content:
                            all_texts.append(input_item.content)
                            text_to_item_indices.setdefault(
                                input_item.content, []
                            ).append(i)

                # Extract texts from output answers
                if item.output:
                    for output_item in item.output:
                        if output_item.answer and output_item.answer.content:
                            all_texts.append(output_item.answer.content)
                            text_to_item_indices.setdefault(
                                output_item.answer.content, []
                            ).append(i)

            if not all_texts:
                return items

            # Process with data-juicer
            sample = {
                "text": all_texts,
                "__dj__stats__": [{} for _ in range(len(all_texts))],
            }

            processed_sample = self.juicer_op.compute_stats_batched(sample)
            keep_indices = list(self.juicer_op.process_batched(processed_sample))

            # Determine which items to keep
            items_to_keep = set()
            for i, (text, keep) in enumerate(zip(all_texts, keep_indices)):
                if keep:
                    items_to_keep.update(text_to_item_indices[text])

            return [items[i] for i in range(len(items)) if i in items_to_keep]

        except Exception as e:
            logger.error(
                f"Error in dataset-level processing with operator {self.name}: {str(e)}"
            )
            return items
