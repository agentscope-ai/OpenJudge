"""
Data Load Module - comprehensive data loading framework with multiple strategies and converters.
Supports loading from local files and remote sources with automatic format detection and conversion.
"""
import json
import random
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import BaseDataSet, DataSample
from rm_gallery.core.utils.file import read_dataset


class DataConverter:
    """
    Base class for data format converters that transform raw data into DataSample format.

    Separates data format conversion logic from data loading logic for modular design.
    All converters must implement the convert_to_data_sample method for their specific format.

    Attributes:
        config: Configuration parameters specific to the converter
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize converter with optional configuration.

        Args:
            config: Converter-specific configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> Union[DataSample, List[DataSample]]:
        """
        Convert raw data dictionary to DataSample format.

        Args:
            data_dict: Raw data dictionary from the source
            source_info: Metadata about data source (file_path, dataset_name, etc.)

        Returns:
            Single DataSample or list of DataSamples

        Raises:
            NotImplementedError: If not implemented by concrete converter
        """
        pass


class DataConverterRegistry:
    """
    Registry for managing data format converters with automatic discovery.

    Provides decorator-based registration and factory methods for converter instantiation.
    Enables extensible converter ecosystem for different data formats.
    """

    _converters: Dict[str, Type[DataConverter]] = {}

    @classmethod
    def register(cls, data_source: str):
        """
        Decorator for registering data converters with specific source identifiers.

        Args:
            data_source: String identifier for the data source format

        Returns:
            Decorator function that registers the converter class
        """

        def decorator(converter_class: Type[DataConverter]):
            cls._converters[data_source] = converter_class
            return converter_class

        return decorator

    @classmethod
    def get_converter(
        cls, data_source: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[DataConverter]:
        """
        Get converter instance for specified data source with fallback to generic.

        Args:
            data_source: Data source identifier to find converter for
            config: Configuration parameters for the converter

        Returns:
            Configured converter instance or None if not found
        """
        converter_class = cls._converters.get(data_source)
        if converter_class:
            return converter_class(config)
        return None

    @classmethod
    def list_sources(cls) -> List[str]:
        """
        List all registered data source identifiers.

        Returns:
            List of registered source identifiers
        """
        return list(cls._converters.keys())


class DataLoad(BaseDataModule):
    """
    Unified data loading module supporting multiple strategies and sources.

    Serves as both the main loading interface and base class for loading strategies.
    Supports local file loading and remote dataset loading with automatic format detection.

    Attributes:
        load_strategy_type: Strategy identifier (local/huggingface)
        data_source: Source identifier for converter selection

    Input Sources:
        - Local: JSON, JSONL, Parquet files or directories
        - Remote: HuggingFace datasets with various configurations

    Output: BaseDataSet containing converted DataSample objects
    """

    load_strategy_type: str = Field(
        default="local", description="data load strategy type (local or remote)"
    )
    data_source: str = Field(default="*", description="data source")

    def __init__(
        self,
        name: str,
        load_strategy_type: str = "local",
        data_source: str = "*",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize data load module with strategy and source configuration.

        Args:
            name: Unique identifier for the loading module
            load_strategy_type: Loading strategy type (local/huggingface)
            data_source: Data source identifier for converter selection
            config: Strategy-specific configuration parameters
            metadata: Additional metadata for tracking and debugging
            **kwargs: Additional initialization parameters
        """
        super().__init__(
            module_type=DataModuleType.LOAD,
            name=name,
            load_strategy_type=load_strategy_type,
            data_source=data_source,
            config=config or {},
            metadata=metadata,
            **kwargs,
        )
        self.validate_config(config or {})

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary for strategy-specific requirements.

        Override this method in subclasses to add specific validation rules
        for different loading strategies.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def load_data(self, **kwargs) -> List[DataSample]:
        """
        Load data from the configured source using the appropriate strategy.

        Automatically selects and instantiates the correct loading strategy
        based on load_strategy_type configuration.

        Args:
            **kwargs: Additional parameters passed to the loading strategy

        Returns:
            List of DataSample objects loaded from the source

        Raises:
            ValueError: If unsupported strategy type is specified
            RuntimeError: If loading fails at any stage
        """
        # If this is a strategy instance (subclass), call the abstract method
        if self.__class__ != DataLoad:
            return self._load_data_impl(**kwargs)

        # Choose strategy based on load_strategy_type
        if self.load_strategy_type == "local":
            strategy = FileDataLoadStrategy(
                name=self.name,
                load_strategy_type=self.load_strategy_type,
                data_source=self.data_source,
                config=self.config.copy(),
                metadata=self.metadata,
            )
        elif self.load_strategy_type == "huggingface":
            strategy = HuggingFaceDataLoadStrategy(
                name=self.name,
                load_strategy_type=self.load_strategy_type,
                data_source=self.data_source,
                config=self.config.copy(),
                metadata=self.metadata,
            )
        else:
            error_msg = f"Unsupported load strategy type: {self.load_strategy_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return strategy.load_data(**kwargs)

    def _load_data_impl(self, **kwargs) -> List[DataSample]:
        """
        Abstract method for strategy implementations to override.

        Each loading strategy must implement this method to handle
        the actual data loading and conversion process.

        Args:
            **kwargs: Strategy-specific loading parameters

        Returns:
            List of DataSample objects from the source

        Raises:
            NotImplementedError: If not implemented by strategy subclass
        """
        raise NotImplementedError("Subclasses must implement _load_data_impl method")

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs
    ) -> BaseDataSet:
        """
        Execute the data loading pipeline and return structured dataset.

        Loads data using the configured strategy, applies optional limits,
        and packages the result as a BaseDataSet for pipeline integration.

        Args:
            input_data: Unused for loading modules (loads from external sources)
            **kwargs: Additional parameters passed to loading strategy

        Returns:
            BaseDataSet containing loaded and converted data samples

        Raises:
            RuntimeError: If data loading process fails
        """
        try:
            # Load data using strategy
            loaded_items = self.load_data(**kwargs)

            # Convert loaded items to DataSample objects if needed
            data_samples = []
            for item in loaded_items:
                data_samples.append(item)

            # Apply limit (if specified)
            if (
                "limit" in self.config
                and self.config["limit"] is not None
                and self.config["limit"] > 0
            ):
                limit = min(int(self.config["limit"]), len(data_samples))
                data_samples = random.sample(data_samples, limit)
                logger.info(
                    f"Applied limit of {limit}, final count: {len(data_samples)}"
                )

            # Create output dataset
            output_dataset = BaseDataSet(
                name=self.name,
                metadata={
                    "source": self.data_source,
                    "strategy_type": self.load_strategy_type,
                    "config": self.config,
                },
                datas=data_samples,
            )
            logger.info(
                f"Successfully loaded {len(data_samples)} items from {self.data_source}"
            )

            return output_dataset
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


class FileDataLoadStrategy(DataLoad):
    """
    File-based data loading strategy for local JSON, JSONL, and Parquet files.

    Supports loading from single files or entire directories with recursive file discovery.
    Automatically detects file formats and applies appropriate parsers with error handling.

    Configuration Requirements:
        - path: File or directory path to load from

    Supported Formats:
        - JSON: Single object or array of objects
        - JSONL: One JSON object per line
        - Parquet: Columnar data format with pandas integration
    """

    def __init__(self, **kwargs):
        """
        Initialize file loading strategy with converter registration.

        Args:
            **kwargs: Initialization parameters passed to parent class
        """
        super().__init__(**kwargs)
        # Initialize data_converter after parent initialization as a normal attribute
        converter = DataConverterRegistry.get_converter(self.data_source, self.config)
        # Set as a normal Python attribute, not a Pydantic field
        object.__setattr__(self, "data_converter", converter)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate file loading configuration requirements.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If required configuration is missing or invalid
            FileNotFoundError: If specified path does not exist
        """
        if "path" not in config:
            raise ValueError("File data strategy requires 'path' in config")
        if not isinstance(config["path"], str):
            raise ValueError("'path' must be a string")

        path = Path(config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Could not find path '{path}'")

        # If it's a file, validate the file format
        if path.is_file():
            ext = path.suffix.lower()
            if ext not in [".json", ".jsonl", ".parquet"]:
                raise ValueError(
                    f"Unsupported file format: {ext}. Supported formats: .json, .jsonl, .parquet"
                )
        # If it's a directory, check if it contains any supported files
        elif path.is_dir():
            supported_files = self._find_supported_files(path)
            if not supported_files:
                raise ValueError(
                    f"Directory '{path}' contains no supported files. Supported formats: .json, .jsonl, .parquet"
                )
        else:
            raise ValueError(f"Path '{path}' is neither a file nor a directory")

    def _find_supported_files(self, directory: Path) -> List[Path]:
        """
        Recursively find all supported files in directory and subdirectories.

        Args:
            directory: Directory path to search

        Returns:
            Sorted list of supported file paths
        """
        supported_extensions = {".json", ".jsonl", ".parquet"}
        supported_files = []

        # Walk through directory and all subdirectories
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                supported_files.append(file_path)

        # Sort files for consistent ordering
        return sorted(supported_files)

    def _load_data_impl(self, **kwargs) -> List[DataSample]:
        path = Path(self.config["path"])

        try:
            all_data_samples = []

            # If it's a single file, load it directly
            if path.is_file():
                ext = path.suffix.lower()
                if ext == ".json":
                    file_data = self._load_json(path, source_file_path=path)
                elif ext == ".jsonl":
                    file_data = self._load_jsonl(path, source_file_path=path)
                elif ext == ".parquet":
                    file_data = self._load_parquet(path, source_file_path=path)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
                all_data_samples.extend(file_data)
                logger.info(f"Loaded {len(file_data)} samples from file: {path}")

            # If it's a directory, load all supported files
            elif path.is_dir():
                supported_files = self._find_supported_files(path)
                logger.info(
                    f"Found {len(supported_files)} supported files in directory: {path}"
                )

                for file_path in supported_files:
                    try:
                        ext = file_path.suffix.lower()
                        if ext == ".json":
                            file_data = self._load_json(
                                file_path, source_file_path=file_path
                            )
                        elif ext == ".jsonl":
                            file_data = self._load_jsonl(
                                file_path, source_file_path=file_path
                            )
                        elif ext == ".parquet":
                            file_data = self._load_parquet(
                                file_path, source_file_path=file_path
                            )
                        else:
                            logger.warning(
                                f"Skipping unsupported file format: {file_path}"
                            )
                            continue

                        all_data_samples.extend(file_data)
                        logger.info(
                            f"Loaded {len(file_data)} samples from file: {file_path}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to load data from {file_path}: {str(e)}")
                        # Continue with other files instead of failing completely
                        continue

                logger.info(
                    f"Total loaded {len(all_data_samples)} samples from {len(supported_files)} files"
                )

            else:
                raise ValueError(f"Path '{path}' is neither a file nor a directory")

            return all_data_samples

        except Exception as e:
            raise RuntimeError(f"Failed to load data from {path}: {str(e)}")

    def _load_json(self, path: Path, source_file_path: Path) -> List[DataSample]:
        """Load data from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_samples = []
        if isinstance(data, list):
            for item in data:
                samples = self._convert_to_data_sample(item, source_file_path)
                if isinstance(samples, list):
                    all_samples.extend(samples)
                else:
                    all_samples.append(samples)
        elif isinstance(data, dict):
            samples = self._convert_to_data_sample(data, source_file_path)
            if isinstance(samples, list):
                all_samples.extend(samples)
            else:
                all_samples.append(samples)
        else:
            raise ValueError("Invalid JSON format: expected list or dict")

        return all_samples

    def _load_jsonl(self, path: Path, source_file_path: Path) -> List[DataSample]:
        """Load data from JSONL file"""
        data_list = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    samples = self._convert_to_data_sample(data, source_file_path)
                    if isinstance(samples, list):
                        data_list.extend(samples)
                    else:
                        data_list.append(samples)
        return data_list

    def _load_parquet(self, path: Path, source_file_path: Path) -> List[DataSample]:
        """Load data from Parquet file"""
        try:
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError("Please install pandas package: pip install pandas")

        data_list = []
        for _, row in df.iterrows():
            try:
                # Convert row to dict and handle any non-serializable types
                data_dict = {}
                for k, v in row.items():
                    if hasattr(v, "item"):
                        try:
                            data_dict[k] = v.item()
                        except (ValueError, AttributeError):
                            # if array type, convert to list and handle nested structures
                            if hasattr(v, "tolist"):
                                data_dict[k] = v.tolist()
                            else:
                                data_dict[k] = v
                    elif hasattr(v, "tolist"):
                        # Handle numpy arrays
                        data_dict[k] = v.tolist()
                    else:
                        data_dict[k] = v

                # ensure data dict contains necessary fields
                if "prompt" not in data_dict:
                    logger.warning(f"Row missing 'prompt' field, skipping: {data_dict}")
                    continue

                # convert data to DataSample object
                samples = self._convert_to_data_sample(data_dict, source_file_path)
                if samples is not None:
                    if isinstance(samples, list):
                        data_list.extend(samples)
                    else:
                        data_list.append(samples)
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue

        return data_list

    def _convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_file_path: Path
    ) -> Union[DataSample, List[DataSample]]:
        """Convert raw data dictionary to DataSample format"""
        if hasattr(self, "data_converter") and self.data_converter:
            source_info = {
                "source_file_path": str(source_file_path),
                "load_type": "local",
            }
            return self.data_converter.convert_to_data_sample(data_dict, source_info)
        else:
            # Fallback to abstract method for backward compatibility
            return self._convert_to_data_sample_impl(data_dict, source_file_path)

    def _convert_to_data_sample_impl(
        self, data_dict: Dict[str, Any], source_file_path: Path
    ) -> DataSample:
        """Abstract method for backward compatibility - override in subclasses if not using converters"""
        raise NotImplementedError(
            "Either use a data converter or implement _convert_to_data_sample_impl method"
        )


class HuggingFaceDataLoadStrategy(DataLoad):
    """
    HuggingFace dataset loading strategy for remote datasets from Hugging Face Hub.

    Supports streaming and non-streaming modes with configurable splits and trust settings.
    Automatically handles dataset download, caching, and conversion to internal format.

    Configuration Options:
        - dataset_config: Optional dataset configuration name
        - huggingface_split: Dataset split to load (train/test/validation)
        - streaming: Enable streaming mode for large datasets
        - trust_remote_code: Allow execution of remote code in datasets
        - limit: Maximum number of samples to load (streaming mode)
    """

    def __init__(self, **kwargs):
        """
        Initialize HuggingFace loading strategy with converter registration.

        Args:
            **kwargs: Initialization parameters passed to parent class
        """
        super().__init__(**kwargs)
        # Initialize data_converter after parent initialization as a normal attribute
        converter = DataConverterRegistry.get_converter(self.data_source, self.config)
        # Set as a normal Python attribute, not a Pydantic field
        object.__setattr__(self, "data_converter", converter)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate HuggingFace dataset configuration parameters.

        Args:
            config: Configuration dictionary to validate

        Note:
            Currently minimal validation - can be extended for specific requirements
        """
        pass

    def _load_data_impl(self, **kwargs) -> List[DataSample]:
        """
        Load data from HuggingFace dataset with automatic conversion.

        Downloads dataset from HuggingFace Hub, applies configured settings,
        and converts each item to DataSample format using registered converters.

        Args:
            **kwargs: Additional loading parameters

        Returns:
            List of converted DataSample objects

        Raises:
            RuntimeError: If dataset loading or conversion fails
        """
        dataset_name = self.name
        dataset_config = self.config.get("dataset_config", None)
        split = self.config.get("huggingface_split", "train")
        streaming = self.config.get("streaming", False)
        trust_remote_code = self.config.get("trust_remote_code", False)

        try:
            logger.info(
                f"Loading dataset: {dataset_name}, config: {dataset_config}, split: {split}"
            )

            # Load dataset from HuggingFace
            dataset = read_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=streaming,
                trust_remote_code=trust_remote_code,
            )

            # Convert to list if streaming
            if streaming:
                # For streaming datasets, take a limited number of samples
                limit = self.config.get("limit", 1000)
                dataset_items = []
                for i, item in enumerate(dataset):
                    if i >= limit:
                        break
                    dataset_items.append(item)
            else:
                dataset_items = dataset

            # Convert to DataSample objects
            data_samples = []
            for item in dataset_items:
                try:
                    samples = self._convert_to_data_sample(item)
                    if samples is not None:
                        if isinstance(samples, list):
                            data_samples.extend(samples)
                        else:
                            data_samples.append(samples)
                except Exception as e:
                    logger.error(f"Error converting item to DataSample: {str(e)}")
                    continue

            logger.info(
                f"Successfully loaded {len(data_samples)} samples from HuggingFace dataset: {dataset_name}"
            )
            return data_samples

        except Exception as e:
            raise RuntimeError(
                f"Failed to load data from HuggingFace dataset {dataset_name}: {str(e)}"
            )

    def _convert_to_data_sample(
        self, data_dict: Dict[str, Any]
    ) -> Union[DataSample, List[DataSample]]:
        """
        Convert raw HuggingFace data dictionary to DataSample format using registered converter.

        Args:
            data_dict: Raw data item from HuggingFace dataset

        Returns:
            Converted DataSample object(s) or None if conversion fails
        """
        if hasattr(self, "data_converter") and self.data_converter:
            source_info = {
                "dataset_name": self.config.get("name"),
                "load_type": "huggingface",
                "dataset_config": self.config.get("dataset_config"),
                "split": self.config.get("huggingface_split", "train"),
            }
            return self.data_converter.convert_to_data_sample(data_dict, source_info)
        else:
            # Fallback to abstract method for backward compatibility
            return self._convert_to_data_sample_impl(data_dict)

    def _convert_to_data_sample_impl(self, data_dict: Dict[str, Any]) -> DataSample:
        """
        Abstract fallback method for backward compatibility.

        Args:
            data_dict: Raw data dictionary to convert

        Returns:
            DataSample object

        Raises:
            NotImplementedError: If no converter is available and method not implemented
        """
        raise NotImplementedError(
            "Either use a data converter or implement _convert_to_data_sample_impl method"
        )


def create_load_module(
    name: str,
    load_strategy_type: str = "local",
    data_source: str = "*",
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataLoad:
    """
    Factory function to create data loading module with specified strategy.

    Args:
        name: Unique identifier for the loading module
        load_strategy_type: Loading strategy type (local/huggingface)
        data_source: Data source identifier for converter selection
        config: Strategy-specific configuration parameters
        metadata: Additional metadata for tracking and debugging

    Returns:
        Configured DataLoad instance ready for pipeline integration
    """
    return DataLoad(
        name=name,
        load_strategy_type=load_strategy_type,
        data_source=data_source,
        config=config,
        metadata=metadata,
    )
