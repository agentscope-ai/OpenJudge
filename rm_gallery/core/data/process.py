"""
Data Processor Module - Unified data processing functionality
"""

from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic, Union
from abc import ABC, abstractmethod
from loguru import logger
import importlib
from pydantic import Field

from rm_gallery.core.data.schema import DataSample, BaseDataSet
from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.base_module import BaseModule

T = TypeVar('T', bound=DataSample)

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
        items = kwargs.get('items', [])
        return self.process_dataset(items)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

class OperatorFactory:
    """Factory for creating operators from configuration"""
    
    _operator_registry: Dict[str, Callable[[Dict[str, Any]], BaseOperator]] = {}
    
    # Operator type mapping
    _operator_types = {
        'filter': 'filter',
        'group': 'group', 
        'map': 'map'
    }

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator for registering operator creation functions"""
        def decorator(func: Callable[[Dict[str, Any]], BaseOperator]) -> Callable:
            cls._operator_registry[name] = func
            return func
        return decorator

    @classmethod
    def create_operator(cls, operator_config: Dict[str, Any]) -> BaseOperator:
        """Create an operator from configuration"""        
        op_type = operator_config.get('type')
        name = operator_config.get('name', op_type)
        config = operator_config.get('config', {})
        
        # Check registry first
        if name in cls._operator_registry:
            return cls._operator_registry[name](operator_config)
        
        # Handle built-in operator types
        if op_type in cls._operator_types:
            return RegisteredOperator(name=name, operator_type=op_type, config=config)
        elif op_type == 'data_juicer':
            return cls._create_data_juicer_filter_operator(name, config)
        else:
            raise ValueError(f"Unknown operator type: {op_type}")
    
    @classmethod
    def _create_data_juicer_filter_operator(cls, name: str, config: Dict[str, Any]) -> BaseOperator:
        """Create data_juicer operator"""
        try:
            # Import data_juicer filter module
            import data_juicer.ops.filter as dj_filters
            
            # Convert snake_case name to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            
            # Try to get the operator class from data_juicer.ops.filter
            if hasattr(dj_filters, class_name):
                operator_class = getattr(dj_filters, class_name)
                return DataJuicerOperator(name=class_name, juicer_op_class=operator_class, config=config)
            else:
                # Fallback: try to import from specific module (for backward compatibility)
                module_path = 'data_juicer.ops.filter'
                operator_module = importlib.import_module(f"{module_path}.{name.lower()}")
                operator_class = getattr(operator_module, class_name)
                return DataJuicerOperator(name=class_name, juicer_op_class=operator_class, config=config)
                
        except ImportError as e:
            raise ImportError(f"Failed to import data_juicer operator '{name}': {e}. "
                            f"Please ensure py-data-juicer is installed: pip install py-data-juicer")
        except AttributeError as e:
            raise AttributeError(f"Data_juicer operator '{class_name}' not found. "
                               f"Available operators can be found in data_juicer.ops.filter module. Error: {e}")
    
class RegisteredOperator(BaseOperator[T]):
    """Operator that uses the operator registry"""
    
    operator_type: str = Field(..., description="operator type")
    
    def __init__(self, name: str, operator_type: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(name=name, config=config, operator_type=operator_type, **kwargs)

    def process_dataset(self, items: List[T]) -> List[T]:
        try:
            if self.name in OperatorFactory._operator_registry:
                operator = OperatorFactory._operator_registry[self.name]({
                    'type': self.operator_type,
                    'name': self.name,
                    'config': self.config
                })
                return operator.process_dataset(items)
            
            logger.warning(f"No registered operator found for name: {self.name}")
            return items
        except Exception as e:
            logger.error(f"Error in {self.operator_type} operation {self.name}: {str(e)}")
            return items

class DataJuicerOperator(BaseOperator[T]):
    """Adapter for data-juicer operators"""
    
    juicer_op: Any = Field(..., description="Data Juicer operator instance")
    
    def __init__(self, name: str, juicer_op_class: Any, config: Optional[Dict[str, Any]] = None, **kwargs):
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
                            text_to_item_indices.setdefault(input_item.content, []).append(i)
                
                # Extract texts from output answers
                if item.output:
                    for output_item in item.output:
                        if output_item.answer and output_item.answer.content:
                            all_texts.append(output_item.answer.content)
                            text_to_item_indices.setdefault(output_item.answer.content, []).append(i)

            if not all_texts:
                return items

            # Process with data-juicer
            sample = {
                'text': all_texts,
                '__dj__stats__': [{} for _ in range(len(all_texts))]
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
            logger.error(f"Error in dataset-level processing with operator {self.name}: {str(e)}")
            return items

class DataProcessModule(BaseDataModule):
    """Data process module - process data"""
    
    operators: List[BaseOperator] = Field(default_factory=list, description="operators list")
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, 
                 operators: Optional[List[BaseOperator]] = None, **kwargs):
        super().__init__(
            module_type=DataModuleType.PROCESS,
            name=name,
            config=config,
            operators=operators or [],
            **kwargs
        )
    
    def run(self, input_data: Union[BaseDataSet, List[DataSample]], **kwargs) -> BaseDataSet:
        """Process data through the operator pipeline"""
        try:
            data_samples = self._prepare_data(input_data)
            processed_data = data_samples
            
            logger.info(f"Processing {len(data_samples)} items with {len(self.operators)} operators")
            
            # Apply operators sequentially
            for i, operator in enumerate(self.operators):
                try:
                    logger.info(f"Applying operator {i+1}/{len(self.operators)}: {operator.name}")
                    processed_data = operator.process_dataset(processed_data)
                    logger.info(f"Operator {operator.name} completed: {len(processed_data)} items remaining")
                except Exception as e:
                    logger.error(f"Error in operator {operator.name}: {str(e)}")
                    continue
            
            # Create output dataset
            output_dataset = BaseDataSet(
                name=f"{self.name}_processed",
                extra_metadata={
                    "original_count": len(data_samples),
                    "processed_count": len(processed_data),
                    "operators_applied": [op.name for op in self.operators]
                },
                datas=processed_data
            )
            
            logger.info(f"Processing completed: {len(data_samples)} -> {len(processed_data)} items")
            return output_dataset
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise e
            
    def _prepare_data(self, input_data: Union[BaseDataSet, List[DataSample]]) -> List[DataSample]:
        """Prepare data for processing"""
        if isinstance(input_data, BaseDataSet):
            return list(input_data.datas)
        return input_data
    
    def get_operators_info(self) -> List[Dict[str, Any]]:
        """Get information about all operators"""
        return [
            {
                "name": op.name,
                "type": op.__class__.__name__,
                "config": op.config
            }
            for op in self.operators
        ]

def create_process_module(name: str, config: Optional[Dict[str, Any]] = None,
                         operators: Optional[List[BaseOperator]] = None) -> DataProcessModule:
    """Create data process module factory function"""
    return DataProcessModule(name=name, config=config, operators=operators) 