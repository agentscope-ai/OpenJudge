"""
Data Build Module - core data build module, driving the entire data pipeline
"""
from typing import Dict, Any, List, Union, Optional
from loguru import logger
from pathlib import Path
import yaml
from pydantic import Field

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import DataSample, BaseDataSet
from rm_gallery.core.data.load import DataLoadModule, create_load_module
from rm_gallery.core.data.process import OperatorFactory, DataProcessModule, create_process_module
from rm_gallery.core.data.ops import register_all_operators
# Ensure operators are registered when module is imported
register_all_operators()

class DataBuildModule(BaseDataModule):
    """Data build module - driving the entire data pipeline"""
    
    load_module: Optional[DataLoadModule] = Field(default=None)
    process_module: Optional[DataProcessModule] = Field(default=None)
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, **modules):
        super().__init__(
            module_type=DataModuleType.BUILD,
            name=name,
            config=config,
            **modules
        )
    
    def run(self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs) -> BaseDataSet:
        """Run data build pipeline"""
        try:
            current_data = input_data
            logger.info(f"Starting data build pipeline: {self.name}")
            
            # Define pipeline stages
            stages = [
                ("Loading", self.load_module),
                ("Processing", self.process_module)
            ]
            
            for stage_name, module in stages:
                if module:
                    logger.info(f"Stage: {stage_name}")
                    current_data = module.run(current_data)
                    logger.info(f"{stage_name} completed: {len(current_data)} items")
            
            logger.info(f"Pipeline completed: {len(current_data)} items processed")
            return current_data
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise e

def create_build_module(name: str, config: Optional[Dict[str, Any]] = None, **modules) -> DataBuildModule:
    """Factory function to create data build module"""
    return DataBuildModule(name=name, config=config, **modules)

def create_build_module_from_yaml(config_path: str) -> DataBuildModule:
    """Create data build module from YAML configuration"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Support new dataset structure
    if 'dataset' in config:
        return _create_from_dataset_config(config['dataset'])
    else:
        raise ValueError("Invalid configuration file")
    
def _create_from_dataset_config(dataset_config: Dict[str, Any]) -> DataBuildModule:
    """Create build module from dataset configuration"""
    dataset_name = dataset_config.get('name', 'dataset')
    modules = {}
    
    # Create load module
    configs = dataset_config.get('configs', {})
    if configs:
        modules['load_module'] = create_load_module(
            name=f"{dataset_name}-loader",
            config={'description': f"Load {dataset_name} data"},
            load_strategy_type=configs.get('type', 'local'),
            data_source=configs.get('source', '*'),
            dimension=configs.get('dimension', '*'),
            load_config={'path': configs.get('path'), 'limit': configs.get('limit')}
        )
    
    # Create process module
    processors = dataset_config.get('processors', [])
    if processors:
        operators = []
        for proc_config in processors:
            try:
                operators.append(OperatorFactory.create_operator(proc_config))
            except Exception as e:
                logger.error(f"Failed to create operator {proc_config}: {str(e)}")
        
        modules['process_module'] = create_process_module(
            name=f"{dataset_name}-processor",
            config={'description': f"Process {dataset_name} data"},
            operators=operators
        )
    return create_build_module(
        name=dataset_name,
        config={'description': f"Build module for {dataset_name}"},
        **modules
    )