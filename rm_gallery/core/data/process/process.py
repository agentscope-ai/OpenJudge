"""
Data Processing Module - unified data processing framework with operator pipeline architecture.

Provides flexible data transformation capabilities through sequential operator application,
supporting filtering, mapping, and custom processing operations on datasets.
"""

from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.process.ops.base import BaseOperator
from rm_gallery.core.data.schema import BaseDataSet, DataSample


class DataProcessor(BaseDataModule):
    """
    Main data processing module that applies operator pipelines to datasets.

    Orchestrates sequential application of processing operators to transform
    and filter data samples while preserving metadata and providing detailed logging.

    Attributes:
        operators: List of processing operators to apply in sequence

    Input: BaseDataSet or List[DataSample] containing raw data
    Output: BaseDataSet with processed data and combined metadata
    """

    operators: List[BaseOperator] = Field(
        default_factory=list, description="operators list"
    )

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        operators: Optional[List[BaseOperator]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize data processing module with operator pipeline.

        Args:
            name: Unique identifier for the processing module
            config: Processing configuration parameters
            operators: List of operators to apply in sequence
            metadata: Additional metadata for tracking and debugging
            **kwargs: Additional initialization parameters
        """
        super().__init__(
            module_type=DataModuleType.PROCESS,
            name=name,
            config=config,
            operators=operators or [],
            metadata=metadata,
            **kwargs,
        )

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample]], **kwargs
    ) -> BaseDataSet:
        """
        Execute the data processing pipeline with sequential operator application.

        Applies each operator in sequence to the dataset, maintaining data integrity
        and providing comprehensive logging of transformations and filtering results.

        Args:
            input_data: Dataset or list of samples to process
            **kwargs: Additional runtime parameters

        Returns:
            BaseDataSet with processed data and combined metadata including
            processing statistics and operator information

        Raises:
            Exception: If processing pipeline fails at any stage
        """
        try:
            data_samples = self._prepare_data(input_data)
            processed_data = data_samples

            # Preserve original dataset metadata if available
            original_metadata = {}
            if isinstance(input_data, BaseDataSet):
                original_metadata = input_data.metadata or {}

            logger.info(
                f"Processing {len(data_samples)} items with {len(self.operators)} operators"
            )

            # Apply operators sequentially
            for i, operator in enumerate(self.operators):
                try:
                    logger.info(
                        f"Applying operator {i + 1}/{len(self.operators)}: {operator.name}"
                    )
                    processed_data = operator.process_dataset(processed_data)
                    logger.info(
                        f"Operator {operator.name} completed: {len(processed_data)} items remaining"
                    )
                except Exception as e:
                    logger.error(f"Error in operator {operator.name}: {str(e)}")
                    continue

            # Merge original metadata with processing metadata
            combined_metadata = original_metadata.copy()
            combined_metadata.update(
                {
                    "original_count": len(data_samples),
                    "processed_count": len(processed_data),
                    "operators_applied": [op.name for op in self.operators],
                }
            )

            # Create output dataset with preserved metadata
            output_dataset = BaseDataSet(
                name=f"{self.name}_processed",
                metadata=combined_metadata,
                datasamples=processed_data,
            )

            logger.info(
                f"Processing completed: {len(data_samples)} -> {len(processed_data)} items"
            )
            return output_dataset

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise e

    def _prepare_data(
        self, input_data: Union[BaseDataSet, List[DataSample]]
    ) -> List[DataSample]:
        """
        Prepare input data for processing by extracting samples from dataset wrapper.

        Args:
            input_data: Input dataset or sample list

        Returns:
            List of DataSample objects ready for operator processing
        """
        if isinstance(input_data, BaseDataSet):
            return list(input_data.datasamples)
        return input_data

    def get_operators_info(self) -> List[Dict[str, Any]]:
        """
        Retrieve information about all configured operators for debugging and monitoring.

        Returns:
            List of dictionaries containing operator metadata including
            name, type, and configuration details
        """
        return [
            {"name": op.name, "type": op.__class__.__name__, "config": op.config}
            for op in self.operators
        ]


def create_processor(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    operators: Optional[List[BaseOperator]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataProcessor:
    """
    Factory function to create data processing module with specified configuration.

    Args:
        name: Unique identifier for the processing module
        config: Processing configuration parameters
        operators: List of operators to include in the pipeline
        metadata: Additional metadata for tracking and debugging

    Returns:
        Configured DataProcessor instance ready for pipeline integration
    """
    return DataProcessor(
        name=name, config=config, operators=operators, metadata=metadata
    )
