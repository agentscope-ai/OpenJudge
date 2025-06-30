"""
Base data module framework providing abstract interfaces for data pipeline components.
Defines common structure and behavior for all data processing modules in the system.
"""

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import BaseDataSet, DataSample


class DataModuleType(Enum):
    """
    Enumeration of supported data module types for categorizing processing components.

    Each type represents a distinct stage in the data pipeline:
    - BUILD: Orchestrates the entire data pipeline workflow
    - LOAD: Ingests data from external sources
    - GENERATE: Creates new data samples programmatically
    - PROCESS: Transforms and filters existing data
    - ANNOTATION: Adds labels and metadata to data
    - EXPORT: Outputs data to various target formats
    """

    BUILD = "builder"
    LOAD = "loader"
    PROCESS = "processor"
    ANNOTATION = "annotator"
    EXPORT = "exporter"


class BaseDataModule(BaseModule):
    """
    Abstract base class for all data processing modules in the pipeline.

    Provides common interface and metadata management for data operations.
    All concrete data modules must inherit from this class and implement the run method.

    Attributes:
        module_type: Type classification of the data module from DataModuleType enum
        name: Unique identifier for the module instance
        config: Module-specific configuration parameters
        metadata: Additional metadata for tracking and debugging
    """

    module_type: DataModuleType = Field(..., description="module type")
    name: str = Field(..., description="module name")
    config: Optional[Dict[str, Any]] = Field(None, description="module config")
    metadata: Optional[Dict[str, Any]] = Field(None, description="metadata")

    @abstractmethod
    def run(self, input_data: Union[BaseDataSet, List[DataSample]], **kwargs):
        """
        Execute the module's data processing logic.

        Args:
            input_data: Input dataset or list of data samples to process
            **kwargs: Additional runtime parameters specific to the module

        Returns:
            Processed data in the form of BaseDataSet or List[DataSample]

        Raises:
            NotImplementedError: If not implemented by concrete subclass
        """
        pass

    def get_module_info(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive module information for debugging and monitoring.

        Returns:
            Dict containing module type, name, configuration, and metadata
            Used for pipeline introspection and debugging
        """
        config_dict = self.config.model_dump() if self.config else None
        return {
            "type": self.module_type.value,
            "name": self.name,
            "config": config_dict,
            "metadata": self.metadata,
        }
