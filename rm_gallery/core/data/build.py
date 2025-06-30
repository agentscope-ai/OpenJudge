"""
Data Build Module - core data pipeline orchestrator for end-to-end data processing.
Coordinates loading, processing, annotation, and export stages with flexible configuration.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.annotation.annotation import DataAnnotator, create_annotator
from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.export import DataExporter, create_exporter
from rm_gallery.core.data.load.base import DataLoader, create_loader
from rm_gallery.core.data.process.ops.base import OperatorFactory
from rm_gallery.core.data.process.process import DataProcessor, create_processor
from rm_gallery.core.data.schema import BaseDataSet, DataSample
from rm_gallery.core.utils.file import read_yaml


class DataBuilder(BaseDataModule):
    """
    Main pipeline orchestrator that coordinates all data processing stages.

    Manages the complete data workflow from raw input to final export format,
    executing each stage in sequence while maintaining data integrity and logging.

    Attributes:
        load_module: Optional data loading component for ingesting external data
        process_module: Optional processing component for filtering and transforming data
        annotation_module: Optional annotation component for adding labels and metadata
        export_module: Optional export component for outputting data in target formats
    """

    load_module: Optional[DataLoader] = Field(default=None)
    process_module: Optional[DataProcessor] = Field(default=None)
    annotation_module: Optional[DataAnnotator] = Field(default=None)
    export_module: Optional[DataExporter] = Field(default=None)

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **modules,
    ):
        """
        Initialize the data build pipeline with specified modules.

        Args:
            name: Unique identifier for the pipeline instance
            config: Pipeline-level configuration parameters
            metadata: Additional metadata for tracking and debugging
            **modules: Keyword arguments for individual pipeline modules
        """
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
        """
        Execute the complete data processing pipeline with all configured stages.

        Processes data through sequential stages: loading → processing → annotation → export.
        Each stage is optional and only executed if the corresponding module is configured.

        Args:
            input_data: Initial dataset, list of samples, or None for load-only pipelines
            **kwargs: Additional runtime parameters passed to individual modules

        Returns:
            Final processed dataset after all stages complete

        Raises:
            Exception: If any pipeline stage fails, with detailed error logging
        """
        try:
            current_data = input_data
            logger.info(f"Starting data build pipeline: {self.name}")

            # Define pipeline stages with human-readable names
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


def create_builder(
    name: str, config: Optional[Dict[str, Any]] = None, **modules
) -> DataBuilder:
    """
    Factory function to create a data build module with specified configuration.

    Args:
        name: Unique identifier for the pipeline
        config: Pipeline configuration parameters
        **modules: Individual module instances to include in the pipeline

    Returns:
        Configured DataBuilder instance ready for execution
    """
    return DataBuilder(name=name, config=config, **modules)


def create_builder_from_yaml(config_path: str) -> DataBuilder:
    """
    Create a data build module from YAML configuration file.

    Supports comprehensive pipeline configuration including data sources,
    processing operators, annotation settings, and export formats.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Fully configured DataBuilder instance based on YAML specification

    Raises:
        FileNotFoundError: If configuration file does not exist
        ValueError: If configuration format is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = read_yaml(config_path)

    # Support new dataset structure
    if "dataset" in config:
        return _create_from_dataset_config(config["dataset"])
    else:
        raise ValueError("Invalid configuration file")


def _create_from_dataset_config(dataset_config: Dict[str, Any]) -> DataBuilder:
    """
    Create build module from dataset configuration with automatic module instantiation.

    Parses dataset configuration to create appropriate modules for each pipeline stage.
    Handles load strategies, processing operators, annotation settings, and export formats.

    Args:
        dataset_config: Dictionary containing complete dataset configuration

    Returns:
        Configured DataBuilder instance with all specified modules

    Raises:
        Exception: If module creation fails for any configured component
    """
    dataset_name = dataset_config.get("name", "dataset")
    metadata = dataset_config.get("metadata", {})
    modules = {}

    # Create load module from data source configuration
    load_config = dataset_config.get("configs", {})
    if load_config:
        modules["load_module"] = create_loader(
            name=dataset_name,
            load_strategy_type=load_config.get("type", "local"),
            data_source=load_config.get("source", "*"),
            config=load_config,
            metadata=metadata,
        )

    # Create process module from operators configuration
    processors = dataset_config.get("processors", [])
    if processors:
        operators = []
        for proc_config in processors:
            try:
                operators.append(OperatorFactory.create_operator(proc_config))
            except Exception as e:
                logger.error(f"Failed to create operator {proc_config}: {str(e)}")

        modules["process_module"] = create_processor(
            name=dataset_name, operators=operators, metadata=metadata
        )

    # Create annotation module from annotation configuration
    annotation_config = dataset_config.get("annotation", {})
    if annotation_config:
        modules["annotation_module"] = create_annotator(
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

    # Create export module from export configuration
    export_config = dataset_config.get("export", {})
    if export_config:
        modules["export_module"] = create_exporter(
            name=dataset_name,
            config=export_config,
            metadata=metadata,
        )

    return create_builder(
        name=dataset_name,
        config={"description": f"Build module for {dataset_name}"},
        **modules,
    )
