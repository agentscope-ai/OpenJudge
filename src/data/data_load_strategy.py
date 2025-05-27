import fnmatch
import json
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Type, List, Any
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.base_module import BaseModule
from .data_schema import DataSample, ContentDict, Reward, DataInfo, ContextDict
from .base import BaseData


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass(frozen=True)
class StrategyKey(BaseModule):
    """
    Immutable key for strategy registration with wildcard support
    """
    data_type: str
    data_source: str
    dimension: str = '*'

    def matches(self, other: 'StrategyKey') -> bool:
        """
        Check if this key matches another key with wildcard support
        """
        return (fnmatch.fnmatch(other.data_type, self.data_type)
                and fnmatch.fnmatch(other.data_source, self.data_source)
                and fnmatch.fnmatch(other.dimension, self.dimension))


class DataLoadStrategy(BaseModule):
    """
    Abstract class for data load strategy
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config(config)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary
        Override this method in subclasses to add specific validation rules
        """
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> List[BaseData]:
        """
        Load data from the source and return a list of BaseData objects
        """
        pass

class DataLoadStrategyRegistry(BaseModule):
    """
    Registry for data load strategies with wildcard matching
    """
    _strategies: Dict[StrategyKey, Type[DataLoadStrategy]] = {}

    @classmethod
    def get_strategy_class(
            cls, data_type: str, data_source: str, dimension: str) -> Optional[Type[DataLoadStrategy]]:
        """
        Retrieve the most specific matching strategy
        """
        logger.info(f'Getting strategy class for data_type: {data_type}, data_source: {data_source}, dimension: {dimension}')

        # Default to wildcard if not provided
        data_type = data_type or '*'
        data_source = data_source or '*'

        # Create the lookup key
        lookup_key = StrategyKey(data_type, data_source, dimension)

        # First, check for exact match
        exact_match = cls._strategies.get(lookup_key)
        if exact_match:
            return exact_match

        # Find all matching wildcard strategies
        matching_strategies = []
        for registered_key, strategy in cls._strategies.items():
            if registered_key.matches(lookup_key):
                matching_strategies.append((registered_key, strategy))

        # Sort matching strategies by specificity
        if matching_strategies:
            def specificity_score(key: StrategyKey) -> int:
                return sum(1 for part in [key.data_type, key.data_source] if part == '*')

            matching_strategies.sort(key=lambda x: specificity_score(x[0]))
            found = matching_strategies[0][1]
            logger.info(f'Found matching strategy: {found}')
            return found

        logger.warning(f'No matching strategy found for data_type: {data_type}, data_source: {data_source}, dimension: {dimension}')
        return None

    @classmethod
    def register(cls, data_type: str, data_source: str, dimension: str):
        """
        Decorator for registering data load strategies
        """
        def decorator(strategy_class: Type[DataLoadStrategy]):
            key = StrategyKey(data_type, data_source, dimension)
            cls._strategies[key] = strategy_class
            return strategy_class
        return decorator

class FileDataLoadStrategy(DataLoadStrategy):
    """
    Base strategy for loading data from files (JSON, JSONL, Parquet)
    """
    def validate_config(self, config: Dict[str, Any]) -> None:
        if 'path' not in config:
            raise ValueError("File data strategy requires 'path' in config")
        if not isinstance(config['path'], str):
            raise ValueError("'path' must be a string")
        
        path = Path(config['path'])
        if not path.exists():
            raise FileNotFoundError(f"Could not find file '{path}'")
            
        ext = path.suffix.lower()
        if ext not in ['.json', '.jsonl', '.parquet']:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: .json, .jsonl, .parquet")

    def load_data(self, **kwargs) -> List[BaseData]:
        path = Path(self.config['path'])
        ext = path.suffix.lower()
        
        try:
            if ext == '.json':
                return self._load_json(path)
            elif ext == '.jsonl':
                return self._load_jsonl(path)
            elif ext == '.parquet':
                return self._load_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            raise RuntimeError(f'Failed to load data from {path}: {str(e)}')

    def _load_json(self, path: Path) -> List[BaseData]:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return [self._convert_to_base_data(item) for item in data]
        elif isinstance(data, dict):
            return [self._convert_to_base_data(data)]
        else:
            raise ValueError("Invalid JSON format: expected list or dict")

    def _load_jsonl(self, path: Path) -> List[BaseData]:
        """Load data from JSONL file"""
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    data_list.append(self._convert_to_base_data(data))
        return data_list

    def _load_parquet(self, path: Path) -> List[BaseData]:
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
                    if hasattr(v, 'item'):
                        try:
                            data_dict[k] = v.item()
                        except ValueError:
                            # if array type, convert to list
                            data_dict[k] = v.tolist()
                    else:
                        data_dict[k] = v
                
                # ensure data dict contains necessary fields
                if 'prompt' not in data_dict:
                    logger.warning(f"Row missing 'prompt' field, skipping: {data_dict}")
                    continue
                    
                # convert data to BaseData object
                base_data = self._convert_to_base_data(data_dict)
                if base_data is not None and isinstance(base_data, BaseData):
                    data_list.append(base_data)
                else:
                    logger.warning(f"Failed to convert row to BaseData: {data_dict}")
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue
            
        return data_list

    @abstractmethod
    def _convert_to_base_data(self, data_dict: Dict[str, Any]) -> BaseData:
        """Convert raw data dictionary to BaseData format"""
        pass


@DataLoadStrategyRegistry.register('local', '*', '*')
class LocalDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading local files (JSON, JSONL, Parquet)
    """
    def _convert_to_base_data(self, data_dict: Dict[str, Any]) -> BaseData:
        """Convert raw data dictionary to BaseData format"""
        pass


@DataLoadStrategyRegistry.register('remote', 'huggingface', '*')
class HuggingfaceDataLoadStrategy(DataLoadStrategy):
    """
    Strategy for loading data from Huggingface datasets
    """
    def convert_to_base_data(self, data_dict: Dict[str, Any]) -> BaseData:
        """Convert raw data dictionary to BaseData format"""
        pass


@DataLoadStrategyRegistry.register('local', 'rewardbench', '*')
class ConversationDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading conversation data with prompt, chosen and rejected responses
    """
    def _convert_to_base_data(self, data_dict: Dict[str, Any]) -> BaseData:
        """Convert conversation data to BaseData format"""
        # generate unique id
        import hashlib
        content = str(data_dict.get('prompt', []))
        unique_id = hashlib.md5(content.encode()).hexdigest()
        
        # process prompt as conversation history
        inputs = []
        prompt = data_dict.get('prompt')
        
        # Convert single-turn conversation to list format
        if isinstance(prompt, dict):
            prompt = [prompt]
            
        if isinstance(prompt, list):
            for turn in prompt:
                if isinstance(turn, dict):
                    role = turn.get('role', 'user')
                    content = turn.get('content', '')
                    if content:  # Only add non-empty content
                        inputs.append(ContentDict(
                            role=role,
                            content=content
                        ))
        else:
            logger.warning(f"Unexpected prompt format: {type(prompt)}")
            return None

        # 创建outputs列表
        outputs = []
        
        # 添加chosen response
        if 'chosen' in data_dict:
            reward = Reward(total_score=1.0)
            # Add multiple reward details directly
            reward.set_reward(self.config['dimension'], 1.0, "Chosen response")
            chosen_output = ContentDict(
                role="assistant",
                content=data_dict['chosen'],
                content_label="chosen",
                rewards=reward
            )
            outputs.append(chosen_output)

        # 添加rejected response
        if 'rejected' in data_dict:
            reward = Reward(total_score=0.0)
            reward.set_reward(self.config['dimension'], 0.0, "Rejected response")
            rejected_output = ContentDict(
                role="assistant",
                content=data_dict['rejected'],
                content_label="rejected",
                rewards=reward
            )
            outputs.append(rejected_output)

        # 如果没有有效的输出，返回None
        if not outputs:
            logger.warning(f"No valid outputs found in data: {data_dict}")
            return None

        # 创建单个sample
        sample = DataSample(
            input=inputs,
            outputs=outputs,
            contexts=[
                ContextDict(
                    context_type='supply',
                    context='xxxx'
                ),
                ContextDict(
                    context_type='demand',
                    context='xxxx'
                )
            ],
            data_info=DataInfo(
                domain=self.config['dimension'],
                source=self.config['source']
            )
        )

        # 创建BaseData对象
        try:
            base_data = BaseData(
                unique_id=unique_id,
                evaluation_sample=sample,
                extra_metadata={
                    'source': self.config['source'],
                    'has_chosen': 'chosen' in data_dict,
                    'has_rejected': 'rejected' in data_dict
                }
            )
            return base_data
        except Exception as e:
            logger.error(f"Error creating BaseData object: {str(e)}")
            return None


@DataLoadStrategyRegistry.register('local', 'chatmessage', '*')
class ChatMessageDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading chat message data from local files (JSON, JSONL, Parquet)
    Supports converting chat messages to BaseData format
    """
    def _convert_to_base_data(self, data_dict: Dict[str, Any]) -> BaseData:
        """Convert chat message data to BaseData format
        
        Expected input format:
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
        }
        """
        # Generate unique id from messages
        import hashlib
        messages_str = str(data_dict.get('messages', []))
        unique_id = hashlib.md5(messages_str.encode()).hexdigest()
        
        # Process messages
        inputs = []
        messages = data_dict.get('messages', [])
        
        if not isinstance(messages, list):
            logger.warning(f"Messages must be a list, got {type(messages)}")
            return None
            
        for msg in messages:
            if not isinstance(msg, dict):
                logger.warning(f"Message must be a dict, got {type(msg)}")
                continue
                
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if not content:  # Skip empty messages
                continue
                
            inputs.append(ContentDict(
                role=role,
                content=content
            ))
            
        if not inputs:
            logger.warning("No valid messages found in data")
            return None
            
        # Handle the last message
        outputs = []
        if inputs and inputs[-1].role == "assistant":
            # Move the last assistant message to outputs
            last_message = inputs.pop()
            reward = Reward(total_score=1.0)
            reward.set_reward(self.config['dimension'], 1.0, "Original response")
            outputs.append(ContentDict(
                role="assistant",
                content=last_message.content,
                content_label="original",
                rewards=reward
            ))
        elif inputs and inputs[-1].role == "user":
            # Remove the last user message
            inputs.pop()
            
        # Create evaluation sample
        sample = DataSample(
            input=inputs,
            outputs=outputs,
            data_info=DataInfo(
                domain=self.config['dimension'],
                source=self.config['source']
            )
        )
        
        # Create BaseData object
        try:
            base_data = BaseData(
                unique_id=unique_id,
                evaluation_sample=sample,
                extra_metadata={
                    'source': self.config['source'],
                    'message_count': len(inputs)
                }
            )
            return base_data
        except Exception as e:
            logger.error(f"Error creating BaseData object: {str(e)}")
            return None




