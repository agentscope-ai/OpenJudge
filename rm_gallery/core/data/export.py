"""
Data Export Module - export processed datasets to various formats with flexible configuration.
Supports multiple output formats, train/test splitting, and preserves original directory structure.
"""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import BaseDataSet, DataSample


class DataExporter(BaseDataModule):
    """
    Data export module for outputting processed datasets to various target formats.

    Supports multiple export formats (JSON, JSONL, Parquet), optional train/test splitting,
    and preservation of original directory structure for organized output management.

    Configuration options:
        - output_dir: Target directory for exported files
        - formats: List of export formats (json, jsonl, parquet)
        - split_ratio: Optional train/test split ratios
        - preserve_structure: Whether to maintain original directory structure
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the data export module with configuration.

        Args:
            name: Unique identifier for the export module
            config: Export configuration including formats, output directory, and split settings
            metadata: Additional metadata for tracking and debugging
            **kwargs: Additional initialization parameters
        """
        super().__init__(
            module_type=DataModuleType.EXPORT,
            name=name,
            config=config,
            metadata=metadata,
            **kwargs,
        )

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs
    ) -> BaseDataSet:
        """
        Execute the data export pipeline with configured formats and settings.

        Processes input data through optional train/test splitting, then exports
        to specified formats while optionally preserving directory structure.

        Args:
            input_data: Dataset or list of samples to export, or None for empty export
            **kwargs: Additional runtime parameters

        Returns:
            Original input dataset unchanged (passthrough for pipeline chaining)

        Raises:
            Exception: If export process fails at any stage
        """
        try:
            if input_data is None:
                logger.warning("No input data provided for export")
                return BaseDataSet(name="empty_export", datasamples=[])

            # Convert to BaseDataSet if needed
            if isinstance(input_data, list):
                dataset = BaseDataSet(name=self.name, datasamples=input_data)
            else:
                dataset = input_data

            # Get export configuration
            export_config = self.config or {}
            output_dir = Path(export_config.get("output_dir", "./exports"))
            formats = export_config.get("formats", ["json"])
            split_ratio = export_config.get(
                "split_ratio", None
            )  # e.g., {"train": 0.8, "test": 0.2}
            preserve_structure = export_config.get("preserve_structure", False)
            filename_prefix = self.name

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            # Split dataset if requested
            if split_ratio:
                train_data, test_data = self._split_dataset(
                    dataset.datasamples, split_ratio
                )
                datasets_to_export = {
                    "train": BaseDataSet(
                        name=f"{dataset.name}_train",
                        datasamples=train_data,
                        metadata=dataset.metadata,
                    ),
                    "test": BaseDataSet(
                        name=f"{dataset.name}_test",
                        datasamples=test_data,
                        metadata=dataset.metadata,
                    ),
                }
            else:
                datasets_to_export = {"full": dataset}

            # Export data
            if preserve_structure:
                # Export with preserved directory structure
                for split_name, split_dataset in datasets_to_export.items():
                    self._export_with_structure(
                        split_dataset, output_dir, formats, filename_prefix, split_name
                    )
            else:
                # Export in traditional way (all data in single files)
                for split_name, split_dataset in datasets_to_export.items():
                    for format_type in formats:
                        self._export_format(
                            split_dataset,
                            output_dir,
                            format_type,
                            filename_prefix,
                            split_name,
                        )

            logger.info(
                f"Successfully exported {len(dataset.datasamples)} samples to {output_dir}"
            )
            return dataset

        except Exception as e:
            logger.error(f"Error during data export: {str(e)}")
            raise

    def _export_with_structure(
        self,
        dataset: BaseDataSet,
        output_dir: Path,
        formats: List[str],
        filename_prefix: str,
        split_name: str,
    ):
        """
        Export data while preserving original directory structure from source files.

        Groups samples by their source file paths and recreates the directory
        structure in the output location for organized data management.

        Args:
            dataset: Dataset to export with preserved structure
            output_dir: Base output directory for structured export
            formats: List of export formats to generate
            filename_prefix: Prefix for generated filenames
            split_name: Split identifier (train/test/full)
        """
        # Group samples by source file path
        file_groups = defaultdict(list)
        base_path = None

        # Try to get the base path from dataset metadata
        if dataset.metadata and "config" in dataset.metadata:
            config = dataset.metadata["config"]
            if "path" in config:
                base_path = Path(config["path"])

        for sample in dataset.datasamples:
            source_file_path = (
                sample.metadata.get("source_file_path") if sample.metadata else None
            )
            if source_file_path:
                file_groups[source_file_path].append(sample)
            else:
                # If no source file path, put in a default group
                file_groups["unknown"].append(sample)

        logger.info(
            f"Exporting {len(file_groups)} file groups with preserved structure"
        )
        logger.info(f"Base path for relative calculation: {base_path}")

        # Export each file group
        for source_path, samples in file_groups.items():
            try:
                # Create a mini dataset for this file group
                file_dataset = BaseDataSet(
                    name=f"{dataset.name}_file_group",
                    datasamples=samples,
                    metadata=dataset.metadata,
                )

                # Determine output path structure
                # Try to get source name from dataset metadata
                source_name = None
                if dataset.metadata and "config" in dataset.metadata:
                    config = dataset.metadata["config"]
                    source_name = config.get("source")

                if source_name:
                    # Use source name from config as base filename
                    if split_name == "full":
                        file_stem = source_name
                    else:
                        file_stem = f"{source_name}_{split_name}"
                    relative_path = Path(file_stem)
                    logger.info(f"Using source-based filename: {file_stem}")
                elif source_path == "unknown":
                    # Handle samples without source file path
                    if split_name == "full":
                        relative_path = Path(f"{filename_prefix}_unknown")
                    else:
                        relative_path = Path(f"{filename_prefix}_{split_name}_unknown")
                else:
                    # Fallback to original file path based naming
                    source_path_obj = Path(source_path)

                    # Calculate relative path from base path
                    if base_path and base_path.is_dir():
                        try:
                            # Get relative path from the base directory
                            relative_to_base = source_path_obj.relative_to(base_path)
                            # Remove the file extension and add split suffix if needed
                            file_stem = relative_to_base.stem
                            if split_name != "full":
                                file_stem = f"{file_stem}_{split_name}"
                            # Preserve the directory structure relative to base
                            relative_path = relative_to_base.parent / file_stem
                        except ValueError:
                            # If source_path is not relative to base_path, fall back to simple filename
                            logger.warning(
                                f"Source path {source_path_obj} is not relative to base path {base_path}, using filename only"
                            )
                            file_stem = source_path_obj.stem
                            if split_name != "full":
                                file_stem = f"{file_stem}_{split_name}"
                            relative_path = Path(file_stem)
                    else:
                        # If no base path or it's not a directory, use just the filename
                        file_stem = source_path_obj.stem
                        if split_name != "full":
                            file_stem = f"{file_stem}_{split_name}"
                        relative_path = Path(file_stem)

                # Export in each requested format
                for format_type in formats:
                    self._export_structured_format(
                        file_dataset, output_dir, format_type, relative_path
                    )

            except Exception as e:
                logger.error(f"Failed to export file group {source_path}: {str(e)}")
                continue

    def _export_structured_format(
        self,
        dataset: BaseDataSet,
        output_dir: Path,
        format_type: str,
        relative_path: Path,
    ):
        """
        Export dataset in specified format with structured directory path.

        Args:
            dataset: Dataset to export
            output_dir: Base output directory
            format_type: Target export format (json/jsonl/parquet)
            relative_path: Relative path structure to preserve
        """
        # Create the full file path with extension
        filename = f"{relative_path.name}.{format_type}"
        full_output_dir = output_dir / relative_path.parent
        filepath = full_output_dir / filename

        # Create directory structure
        full_output_dir.mkdir(parents=True, exist_ok=True)

        # Use the common format export method
        self._export_by_format(dataset, filepath, format_type)

    def _export_by_format(self, dataset: BaseDataSet, filepath: Path, format_type: str):
        """
        Common method to export dataset in specified format.

        Args:
            dataset: Dataset to export
            filepath: Full file path for output
            format_type: Target export format
        """
        # Format handler mapping
        format_handlers = {
            "json": self._export_json,
            "jsonl": self._export_jsonl,
            "parquet": self._export_parquet,
        }

        format_key = format_type.lower()
        if format_key in format_handlers:
            format_handlers[format_key](dataset, filepath)
        else:
            logger.warning(f"Unsupported format: {format_type}")

    def _split_dataset(
        self, data_samples: List[DataSample], split_ratio: Dict[str, float]
    ) -> Tuple[List[DataSample], List[DataSample]]:
        """
        Split dataset into training and testing sets with specified ratios.

        Args:
            data_samples: List of data samples to split
            split_ratio: Dictionary with train/test ratios (must include 'train' key)

        Returns:
            Tuple of (training_samples, testing_samples)

        Raises:
            ValueError: If split ratio is invalid or missing required keys
        """
        if not split_ratio or "train" not in split_ratio:
            raise ValueError("Split ratio must contain 'train' key")

        train_ratio = split_ratio["train"]
        if not 0 < train_ratio < 1:
            raise ValueError("Train ratio must be between 0 and 1")

        # Check if we have group IDs in the data
        has_groups = any(
            sample.metadata and sample.metadata.get("data_group_id")
            for sample in data_samples
        )

        if has_groups:
            # Group-based splitting to prevent data leakage
            groups = defaultdict(list)
            ungrouped_samples = []

            for sample in data_samples:
                if sample.metadata and sample.metadata.get("data_group_id"):
                    group_id = sample.metadata["data_group_id"]
                    groups[group_id].append(sample)
                else:
                    ungrouped_samples.append(sample)

            # Shuffle groups for random split
            group_list = list(groups.keys())
            random.seed(42)  # For reproducible results
            random.shuffle(group_list)

            # Calculate split point based on number of groups
            train_group_count = int(len(group_list) * train_ratio)

            train_data = []
            test_data = []

            # Assign groups to train/test
            for i, group_id in enumerate(group_list):
                if i < train_group_count:
                    train_data.extend(groups[group_id])
                else:
                    test_data.extend(groups[group_id])

            # Handle ungrouped samples (split them individually)
            if ungrouped_samples:
                random.shuffle(ungrouped_samples)
                ungrouped_train_size = int(len(ungrouped_samples) * train_ratio)
                train_data.extend(ungrouped_samples[:ungrouped_train_size])
                test_data.extend(ungrouped_samples[ungrouped_train_size:])

            logger.info(
                f"Group-based split: {len(groups)} groups, {len(train_data)} training samples, {len(test_data)} test samples"
            )

        else:
            # Original individual sample splitting
            shuffled_data = data_samples.copy()
            random.seed(42)  # For reproducible results
            random.shuffle(shuffled_data)

            # Calculate split point
            train_size = int(len(shuffled_data) * train_ratio)

            train_data = shuffled_data[:train_size]
            test_data = shuffled_data[train_size:]

            logger.info(
                f"Individual split: {len(train_data)} training samples, {len(test_data)} test samples"
            )

        return train_data, test_data

    def _export_format(
        self,
        dataset: BaseDataSet,
        output_dir: Path,
        format_type: str,
        filename_prefix: str,
        split_name: str,
    ):
        """
        Export dataset in specified format with standard file naming.

        Args:
            dataset: Dataset to export
            output_dir: Target output directory
            format_type: Export format (json/jsonl/parquet)
            filename_prefix: Prefix for the output filename
            split_name: Split identifier for filename generation
        """
        if split_name == "full":
            filename = f"{filename_prefix}.{format_type}"
        else:
            filename = f"{filename_prefix}_{split_name}.{format_type}"

        filepath = output_dir / filename

        # Use the common format export method
        self._export_by_format(dataset, filepath, format_type)

    def _export_json(self, dataset: BaseDataSet, filepath: Path):
        """
        Export dataset to JSON format with pretty printing.

        Args:
            dataset: Dataset to export
            filepath: Target file path for JSON output

        Raises:
            Exception: If JSON export fails
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    dataset.to_dict(), f, ensure_ascii=False, indent=2, default=str
                )
            logger.info(f"Exported to JSON: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export JSON to {filepath}: {str(e)}")
            raise

    def _export_jsonl(self, dataset: BaseDataSet, filepath: Path):
        """
        Export dataset to JSONL format (one JSON object per line).

        Args:
            dataset: Dataset to export
            filepath: Target file path for JSONL output

        Raises:
            Exception: If JSONL export fails
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for sample in dataset.datasamples:
                    json.dump(sample.model_dump(), f, ensure_ascii=False, default=str)
                    f.write("\n")
            logger.info(f"Exported to JSONL: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export JSONL to {filepath}: {str(e)}")
            raise

    def _export_parquet(self, dataset: BaseDataSet, filepath: Path):
        """
        Export dataset to Parquet format for efficient storage and analytics.

        Flattens complex data structures to tabular format suitable for
        data analysis and machine learning pipelines.

        Args:
            dataset: Dataset to export
            filepath: Target file path for Parquet output

        Raises:
            Exception: If Parquet export fails
        """
        try:
            # Convert data samples to flat dictionary format
            records = []
            for sample in dataset.datasamples:
                record = {
                    "unique_id": sample.unique_id,
                    "input": json.dumps(
                        [msg.model_dump() for msg in sample.input], default=str
                    ),
                    "output": json.dumps(
                        [out.model_dump() for out in sample.output], default=str
                    ),
                    "task_category": sample.task_category,
                    "source": sample.source,
                    "created_at": sample.created_at,
                    "metadata": json.dumps(sample.metadata, default=str)
                    if sample.metadata
                    else None,
                }
                records.append(record)

            # Create DataFrame and save as Parquet
            df = pd.DataFrame(records)
            df.to_parquet(filepath, index=False, engine="pyarrow")
            logger.info(f"Exported to Parquet: {filepath}")
        except Exception as e:
            logger.error(f"Failed to export Parquet to {filepath}: {str(e)}")
            raise


def create_exporter(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataExporter:
    """
    Factory function to create a data export module with specified configuration.

    Args:
        name: Unique identifier for the export module
        config: Export configuration including formats, output settings, and split ratios
        metadata: Additional metadata for tracking and debugging

    Returns:
        Configured DataExporter instance ready for pipeline integration
    """
    return DataExporter(name=name, config=config, metadata=metadata)
