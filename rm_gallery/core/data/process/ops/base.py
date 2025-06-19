"""
Data Processing Operator Framework - extensible system for data transformation and filtering operations.

Provides base classes, registry, and factory system for creating modular data processing
operators that can be combined into flexible processing pipelines.
"""

import importlib
from abc import abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from loguru import logger
from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import DataSample

T = TypeVar("T", bound=DataSample)


class BaseOperator(BaseModule, Generic[T]):
    """
    Abstract base class for all data processing operators in the pipeline framework.

    Operators are modular processing units that transform, filter, or modify datasets
    in a standardized way. Each operator processes a list of data samples and returns
    a modified list, enabling flexible composition into processing pipelines.

    Attributes:
        name: Unique identifier for the operator instance
        config: Configuration parameters specific to the operator
    """

    name: str = Field(..., description="operator name")
    config: Dict[str, Any] = Field(default_factory=dict, description="operator config")

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize operator with name and configuration.

        Args:
            name: Unique identifier for the operator
            config: Operator-specific configuration parameters
            **kwargs: Additional initialization parameters
        """
        super().__init__(name=name, config=config or {}, **kwargs)

    @abstractmethod
    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process the entire dataset with operator-specific logic.

        This is the main processing method that must be implemented by all
        concrete operators. It receives a list of data samples and returns
        a modified list after applying the operator's transformation or filtering.

        Args:
            items: List of data samples to process

        Returns:
            List of processed data samples (may be filtered or transformed)
        """
        pass

    def run(self, **kwargs):
        """
        Run method implementation for operator interface compatibility.

        Args:
            **kwargs: Runtime parameters including 'items' list

        Returns:
            Result of process_dataset method
        """
        items = kwargs.get("items", [])
        return self.process_dataset(items)

    def __str__(self) -> str:
        """
        String representation for debugging and logging.

        Returns:
            Human-readable operator description
        """
        return f"{self.__class__.__name__}({self.name})"


class OperatorFactory:
    """
    Factory class for creating and registering data processing operators.

    Provides centralized operator creation from configuration dictionaries,
    supports built-in operator types, and enables registration of custom operators
    through decorator pattern or direct registration.
    """

    _operator_registry: Dict[str, Callable[[Dict[str, Any]], BaseOperator]] = {}
    _external_operators: Dict[str, type] = {}

    # Operator type mapping
    _operator_types = {"filter": "filter", "group": "group", "map": "map"}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator for registering operator creation functions or classes.

        Supports both function-based and class-based operator registration.
        For classes, automatically creates a factory function.

        Args:
            name: Unique operator identifier for registry lookup

        Returns:
            Decorator function that registers and returns the original object

        Example:
            @OperatorFactory.register("my_filter")
            class MyFilterOperator(BaseOperator):
                ...
        """

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
        """
        Create operator instance from configuration dictionary.

        Supports registered operators, built-in types, and external library
        operators (like data_juicer) through automatic discovery and instantiation.

        Args:
            operator_config: Configuration dictionary containing:
                - type: Operator type identifier
                - name: Operator instance name
                - config: Operator-specific parameters

        Returns:
            Configured operator instance ready for pipeline integration

        Raises:
            ValueError: If operator type is unknown or unsupported
            ImportError: If external operator dependencies are missing
        """
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
        """
        Create operator adapter for data_juicer library operators.

        Automatically discovers and wraps data_juicer filter operators for
        integration into the processing pipeline framework.

        Args:
            name: data_juicer operator name (snake_case)
            config: Operator configuration parameters

        Returns:
            DataJuicerOperator wrapper instance

        Raises:
            ImportError: If data_juicer library is not installed
            AttributeError: If specified operator class is not found
        """
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
    """
    Generic operator wrapper that delegates to registry-based implementations.

    Used for operators registered in the factory registry, providing a uniform
    interface while delegating actual processing to registered functions or classes.

    Attributes:
        operator_type: Type classification of the operator
    """

    operator_type: str = Field(..., description="operator type")

    def __init__(
        self,
        name: str,
        operator_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize registered operator wrapper.

        Args:
            name: Operator instance name
            operator_type: Type classification for the operator
            config: Operator configuration parameters
            **kwargs: Additional initialization parameters
        """
        super().__init__(
            name=name, config=config, operator_type=operator_type, **kwargs
        )

    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process dataset by delegating to registered operator implementation.

        Args:
            items: List of data samples to process

        Returns:
            Processed data samples from registered operator
        """
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
    """
    Adapter class for integrating data-juicer library operators into the pipeline framework.

    Wraps data-juicer operators to provide standardized interface and automatic
    text extraction/processing for compatibility with DataSample structures.

    Attributes:
        juicer_op: Instantiated data-juicer operator for actual processing
    """

    juicer_op: Any = Field(..., description="Data Juicer operator instance")

    def __init__(
        self,
        name: str,
        juicer_op_class: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize data-juicer operator adapter.

        Args:
            name: Operator instance name
            juicer_op_class: data-juicer operator class to wrap
            config: Configuration parameters for the juicer operator
            **kwargs: Additional initialization parameters
        """
        juicer_op = juicer_op_class(**config) if config else juicer_op_class()
        super().__init__(name=name, config=config, juicer_op=juicer_op, **kwargs)

    def process_dataset(self, items: List[T]) -> List[T]:
        """
        Process dataset using data-juicer operators with automatic text extraction.

        Extracts text content from DataSample structures, applies data-juicer
        filtering, and returns samples that pass the filter criteria.

        Args:
            items: List of DataSample objects to process

        Returns:
            Filtered list of DataSample objects that pass data-juicer criteria
        """
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
