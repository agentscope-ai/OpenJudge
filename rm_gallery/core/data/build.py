"""
Data Build Module - core data build module, driving the entire data pipeline
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.annotation.annotation import (
    DataAnnotator,
    create_annotation_module,
)
from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.export import DataExport, create_export_module
from rm_gallery.core.data.load.base import DataLoad, create_load_module
from rm_gallery.core.data.process.ops.base import OperatorFactory
from rm_gallery.core.data.process.process import DataProcess, create_process_module
from rm_gallery.core.data.schema import BaseDataSet, DataSample
from rm_gallery.core.utils.file import read_yaml


class DataBuild(BaseDataModule):
    """Data build module - driving the entire data pipeline"""

    load_module: Optional[DataLoad] = Field(default=None)
    process_module: Optional[DataProcess] = Field(default=None)
    annotation_module: Optional[DataAnnotator] = Field(default=None)
    export_module: Optional[DataExport] = Field(default=None)

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **modules,
    ):
        super().__init__(
            module_type=DataModuleType.BUILD,
            name=name,
            config=config,
            metadata=metadata,
            **modules,
        )

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs
    ) -> BaseDataSet:
        """Run data build pipeline"""
        try:
            current_data = input_data
            logger.info(f"Starting data build pipeline: {self.name}")

            # Define pipeline stages
            stages = [
                ("Loading", self.load_module),
                ("Processing", self.process_module),
                ("Annotation", self.annotation_module),
                ("Export", self.export_module),
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


def create_build_module(
    name: str, config: Optional[Dict[str, Any]] = None, **modules
) -> DataBuild:
    """Factory function to create data build module"""
    return DataBuild(name=name, config=config, **modules)


def create_build_module_from_yaml(config_path: str) -> DataBuild:
    """Create data build module from YAML configuration"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = read_yaml(config_path)

    # Support new dataset structure
    if "dataset" in config:
        return _create_from_dataset_config(config["dataset"])
    else:
        raise ValueError("Invalid configuration file")


def _create_from_dataset_config(dataset_config: Dict[str, Any]) -> DataBuild:
    """Create build module from dataset configuration"""
    dataset_name = dataset_config.get("name", "dataset")
    metadata = dataset_config.get("metadata", {})
    modules = {}

    # Create load module
    load_config = dataset_config.get("configs", {})
    if load_config:
        modules["load_module"] = create_load_module(
            name=dataset_name,
            load_strategy_type=load_config.get("type", "local"),
            data_source=load_config.get("source", "*"),
            config=load_config,
            metadata=metadata,
        )

    # Create process module
    processors = dataset_config.get("processors", [])
    if processors:
        operators = []
        for proc_config in processors:
            try:
                operators.append(OperatorFactory.create_operator(proc_config))
            except Exception as e:
                logger.error(f"Failed to create operator {proc_config}: {str(e)}")

        modules["process_module"] = create_process_module(
            name=dataset_name, operators=operators, metadata=metadata
        )

    # Create annotation module
    annotation_config = dataset_config.get("annotation", {})
    if annotation_config:
        modules["annotation_module"] = create_annotation_module(
            name=dataset_name,
            label_config=annotation_config.get("label_config"),
            template_name=annotation_config.get("template_name"),
            project_title=annotation_config.get("project_title"),
            project_description=annotation_config.get("project_description"),
            server_url=annotation_config.get("server_url"),
            api_token=annotation_config.get("api_token"),
            export_processor=annotation_config.get("export_processor"),
            metadata=metadata,
        )

    # Create export module
    export_config = dataset_config.get("export", {})
    if export_config:
        modules["export_module"] = create_export_module(
            name=dataset_name,
            config=export_config,
            metadata=metadata,
        )

    return create_build_module(
        name=dataset_name,
        config={"description": f"Build module for {dataset_name}"},
        **modules,
    )
