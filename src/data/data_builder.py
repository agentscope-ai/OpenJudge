from typing import Dict, Any
from loguru import logger
from pathlib import Path
import yaml
import random

from src.base_module import BaseModule
from .base import BaseData, BaseDataSet
from .data_load_strategy import DataLoadStrategyRegistry
from .data_processor import (
    DataPipeline, OperatorFactory,
)


class DataBuilder(BaseModule):
    """
    DataBuilder is responsible for building evaluation datasets from various sources.
    It supports loading data from local files and remote sources (like Huggingface),
    and processing data through a executor of operators.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data builder with a YAML configuration file path
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        self.config = self._load_yaml_config()
        self.pipeline = DataPipeline(name=self.config['dataset'].get('name', 'unnamed_pipeline'))
        self._init_operators()
        
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)    
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load YAML configuration: {str(e)}")
            
    def _init_operators(self):
        """Initialize data operators based on configuration"""
        if 'processors' not in self.config.get('dataset', {}):
            logger.info("No processors configured")
            return
            
        processor_configs = self.config['dataset']['processors']
        logger.info(f"Initializing {len(processor_configs)} processors")
        
        for proc_config in processor_configs:
            try:
                logger.info(f"Creating operator: {proc_config}")
                operator = OperatorFactory.create_operator(proc_config)
                if operator:
                    self.pipeline.add_operator(operator)
                    logger.info(f"Added operator: {operator}")
            except Exception as e:
                logger.error(f"Failed to create operator {proc_config.get('type')}: {str(e)}")
    
    def build_dataset(self) -> BaseDataSet:
        """
        Build a BaseDataSet from the configuration and process it through the executor
        
        Returns:
            BaseDataSet: The constructed and processed dataset
        """
        if 'dataset' not in self.config:
            raise ValueError("Configuration must contain a 'dataset' section")
            
        dataset_config = self.config['dataset']
        configs = dataset_config.get('configs', {})
        
        # get necessary configs
        data_type = configs.get('type')
        data_source = configs.get('source', '*')
        dimension = configs.get('dimension', '*')

        logger.info(f"Building dataset with type: {data_type}, source: {data_source}, dimension: {dimension}")

        if 'dimension' not in configs:
            logger.warning("Missing 'dimension' in config")
            configs['dimension'] = 'common'
        
        # Get the appropriate data load strategy
        strategy_class = DataLoadStrategyRegistry.get_strategy_class(
            data_type=data_type,
            data_source=data_source,
            dimension=dimension
        )
        
        if not strategy_class:
            raise ValueError(f"No suitable data load strategy found for type: {data_type}")
            
        # Initialize and use the strategy to load data
        strategy = strategy_class(configs)
        data_items = strategy.load_data()
        logger.info(f"Loaded {len(data_items)} data items")
        
        # make sure all data items are BaseData objects
        valid_data_items = []
        for item in data_items:
            if isinstance(item, BaseData):
                valid_data_items.append(item)
            else:
                logger.warning(f"Skipping invalid data item: {type(item)}")
        
        logger.info(f"Found {len(valid_data_items)} valid data items")
        
        # Apply limit if specified
        if 'limit' in configs:
            valid_data_items = random.sample(valid_data_items, configs['limit'])
            logger.info(f"Applied limit of {configs['limit']}, now have {len(valid_data_items)} items")
            
        # Process all items through executor
        try:
            logger.info("Processing items through executor")
            processed_items = self.pipeline.run(valid_data_items)
            logger.info(f"Pipeline processing complete. Input: {len(valid_data_items)} items, Output: {len(processed_items)} items")
        except Exception as e:
            logger.error(f"Error processing data through executor: {str(e)}")
            processed_items = []
        
        # Create and return the dataset
        return BaseDataSet(
            name=dataset_config.get('name', 'unnamed_dataset'),
            description=dataset_config.get('description'),
            version=dataset_config.get('version', '1.0.0'),
            extra_metadata=dataset_config.get('extra_metadata', {}),
            datas=processed_items if processed_items is not None else []
        )

def load_dataset_from_yaml(config_path: str) -> BaseDataSet:
    """
    Convenience function to load a dataset from a YAML configuration file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        BaseDataSet: The loaded and processed dataset
    """
    builder = DataBuilder(config_path)
    return builder.build_dataset()