"""
Data Load Module - load data from various data sources
"""
import fnmatch
import json
import pandas as pd
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional, Type, ClassVar
from pathlib import Path
from loguru import logger
from pydantic import Field
from datasets import load_dataset

from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import BaseDataSet, DataSample, ChatMessage, DataOutput, Step


@dataclass(frozen=True)
class StrategyKey:
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


class DataLoadStrategy(BaseDataModule):
    """
    Abstract class for data load strategy
    """
    config: Dict[str, Any]

    def __init__(self, config: Dict[str, Any], **kwargs):
        # Provide required fields for BaseDataModule
        super().__init__(
            module_type=DataModuleType.LOAD,
            name=kwargs.get('name', f"{self.__class__.__name__}"),
            config=config,
            **kwargs
        )
        self.validate_config(config)

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary
        Override this method in subclasses to add specific validation rules
        """
        pass

    @abstractmethod
    def load_data(self, **kwargs) -> List[DataSample]:
        """
        Load data from the source and return a list of DataSample objects
        """
        pass

    def run(self, **kwargs) -> List[DataSample]:
        """
        Run method implementation for DataLoadStrategy
        """
        return self.load_data(**kwargs)


class DataLoadStrategyRegistry:
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
        lookup_key = StrategyKey(data_type=data_type, data_source=data_source, dimension=dimension)

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
            key = StrategyKey(data_type=data_type, data_source=data_source, dimension=dimension)
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

    def load_data(self, **kwargs) -> List[DataSample]:
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

    def _load_json(self, path: Path) -> List[DataSample]:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            return [self._convert_to_data_sample(item) for item in data]
        elif isinstance(data, dict):
            return [self._convert_to_data_sample(data)]
        else:
            raise ValueError("Invalid JSON format: expected list or dict")

    def _load_jsonl(self, path: Path) -> List[DataSample]:
        """Load data from JSONL file"""
        data_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    data_list.append(self._convert_to_data_sample(data))
        return data_list

    def _load_parquet(self, path: Path) -> List[DataSample]:
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
                        except (ValueError, AttributeError):
                            # if array type, convert to list and handle nested structures
                            if hasattr(v, 'tolist'):
                                data_dict[k] = v.tolist()
                            else:
                                data_dict[k] = v
                    elif hasattr(v, 'tolist'):
                        # Handle numpy arrays
                        data_dict[k] = v.tolist()
                    else:
                        data_dict[k] = v
                
                # ensure data dict contains necessary fields
                if 'prompt' not in data_dict:
                    logger.warning(f"Row missing 'prompt' field, skipping: {data_dict}")
                    continue
                    
                # convert data to DataSample object
                data_sample = self._convert_to_data_sample(data_dict)
                if data_sample is not None:
                    data_list.append(data_sample)
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}")
                continue
            
        return data_list

    @abstractmethod
    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert raw data dictionary to DataSample format"""
        pass


@DataLoadStrategyRegistry.register('remote', 'huggingface', '*')
class HuggingfaceDataLoadStrategy(DataLoadStrategy):
   def load_data(self, **kwargs) -> List[DataSample]:
       pass


@DataLoadStrategyRegistry.register('local', 'rewardbench', '*')
class ConversationDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading conversation data with prompt, chosen and rejected responses
    """
    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert conversation data to DataSample format"""
        # generate unique id
        content = str(data_dict.get('prompt', []))
        unique_id = hashlib.md5(content.encode()).hexdigest()
        
        # Create input from prompt
        data_input = self._create_conversation_input(data_dict)
        
        # Create outputs from chosen/rejected responses
        data_output = self._create_conversation_output(data_dict)
        
        try:
            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source='rewardbench',
                domain=self.config.get('dimension', 'conversation'),
                metadata={
                    'raw_data': data_dict,
                    'load_strategy': 'ConversationDataLoadStrategy'
                }
            )
            
            return data_sample
            
        except Exception as e:
            logger.error(f"Error creating conversation DataSample: {str(e)}")
            return None
    
    def _create_conversation_input(self, data_dict: Dict[str, Any]) -> List[ChatMessage]:
        """Create DataInput from conversation prompt"""
        history = []
        prompt = data_dict.get('prompt')
        
        # Convert single-turn conversation to list format
        if isinstance(prompt, dict):
            prompt = [prompt]
            
        if isinstance(prompt, list):
            for turn in prompt:
                if isinstance(turn, dict):
                    role = turn.get('role', 'user')
                    content = turn.get('content', str(turn))
                    history.append(ChatMessage(role=role, content=content))
                else:
                    history.append(ChatMessage(role='user', content=str(turn)))
        elif isinstance(prompt, str):
            history.append(ChatMessage(role='user', content=prompt))
        
        return history
    
    def _create_conversation_output(self, data_dict: Dict[str, Any]) -> List[DataOutput]:
        """Create DataOutput list from conversation responses"""
        outputs = []
        
        # Handle chosen response
        if 'chosen' in data_dict:
            chosen_content = data_dict['chosen']
            if isinstance(chosen_content, list):
                # Multi-turn chosen response
                for turn in chosen_content:
                    if isinstance(turn, dict):
                        content = turn.get('content', str(turn))
                    else:
                        content = str(turn)
                    outputs.append(DataOutput(
                        answer=Step(role='assistant', content=content,label={"preference": "chosen"}),
                    ))
            else:
                outputs.append(DataOutput(
                    answer=Step(role='assistant', content=str(chosen_content),label={"preference": "chosen"}),
                ))
        
        # Handle rejected response
        if 'rejected' in data_dict:
            rejected_content = data_dict['rejected']
            if isinstance(rejected_content, list):
                # Multi-turn rejected response
                for turn in rejected_content:
                    if isinstance(turn, dict):
                        content = turn.get('content', str(turn))
                    else:
                        content = str(turn)
                    outputs.append(DataOutput(
                        answer=Step(role='assistant', content=content,label={"preference": "rejected"}),
                    ))
            else:
                outputs.append(DataOutput(
                    answer=Step(role='assistant', content=str(rejected_content),label={"preference": "rejected"}),
                ))
        
        return outputs


@DataLoadStrategyRegistry.register('local', 'chatmessage', '*')
class ChatMessageDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading chat message data
    """
    
    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert chat message data to DataSample format"""
        # generate unique id
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()
        
        try:
            # Create input from messages
            data_input = self._create_chat_input(data_dict)
            
            # Create output from response
            data_output = self._create_chat_output(data_dict)
            
            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source='chatmessage',
                domain=data_dict.get('domain', 'chat'),
                metadata={
                    'raw_data': data_dict,
                    'load_strategy': 'ChatMessageDataLoadStrategy'
                }
            )
            
            return data_sample
            
        except Exception as e:
            logger.error(f"Error creating chat DataSample: {str(e)}")
            return None
    
    def _create_chat_input(self, data_dict: Dict[str, Any]) -> List[ChatMessage]:
        """Create DataInput from chat messages"""
        history = []
        
        # Handle messages field
        if 'messages' in data_dict:
            messages = data_dict['messages']
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('content', str(msg))
                        history.append(ChatMessage(role=role, content=content))
                    else:
                        history.append(ChatMessage(role='user', content=str(msg)))
        
        # Handle conversation field
        elif 'conversation' in data_dict:
            conversation = data_dict['conversation']
            if isinstance(conversation, list):
                for turn in conversation:
                    if isinstance(turn, dict):
                        role = turn.get('role', 'user')
                        content = turn.get('content', str(turn))
                        history.append(ChatMessage(role=role, content=content))
        
        # Handle simple text field
        elif 'text' in data_dict:
            history.append(ChatMessage(role='user', content=str(data_dict['text'])))
        
        return history
    
    def _create_chat_output(self, data_dict: Dict[str, Any]) -> List[DataOutput]:
        """Create DataOutput from chat response"""
        outputs = []
        
        # Handle response field
        if 'response' in data_dict:
            outputs.append(DataOutput(
                answer=Step(role='assistant', content=str(data_dict['response']))
            ))
        
        # Handle assistant message in messages
        elif 'messages' in data_dict:
            messages = data_dict['messages']
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'assistant':
                        outputs.append(DataOutput(
                            answer=Step(role='assistant', content=str(msg.get('content', '')))
                        ))
        
        return outputs


@DataLoadStrategyRegistry.register('local', 'prmbench', '*')
class PRMDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading Process Reward Model (PRM) data
    Handles mathematical reasoning data with step-wise processes
    """
    
    # define as class attribute instead of instance attribute
    DIMENSION_CLASSIFICATION_MAPPING: ClassVar[Dict[str, str]] = {
        'confidence': 'confidence',
        '*': None  # wildcard, means no filtering
    }
    
    def load_data(self, **kwargs) -> List[DataSample]:
        """Override load_data method, add dimension filtering"""
        # first call parent class method to load all data
        all_data = super().load_data(**kwargs)
        
        # get current dimension config
        current_dimension = self.config.get('dimension', '*')
        
        # if wildcard or no mapping, return all data
        if current_dimension == '*' or current_dimension not in self.DIMENSION_CLASSIFICATION_MAPPING:
            logger.info(f"No filtering applied for dimension: {current_dimension}")
            return all_data
        
        # get corresponding classification
        target_classification = self.DIMENSION_CLASSIFICATION_MAPPING[current_dimension]
        
        # filter data
        filtered_data = []
        for data_sample in all_data:
            if data_sample and data_sample.metadata:
                data_classification = data_sample.metadata.get('classification')
                if data_classification == target_classification:
                    filtered_data.append(data_sample)
        
        logger.info(f"Filtered data by dimension '{current_dimension}' -> classification '{target_classification}': "
                   f"{len(all_data)} -> {len(filtered_data)} items")
        
        return filtered_data
    
    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert PRM data to DataSample format
        
        Expected input format:
        {
            "original_question": "...",
            "modified_question": "...", 
            "original_process": ["step1", "step2", ...],
            "modified_process": ["step1", "step2", ...],
            "modified_steps": [5, 6],
            "error_steps": [5, 6],
            "reason": "...",
            "idx": "...",
            "question": "...",
            "classification": "confidence"
        }
        """
        
        # Generate unique id from idx or question
        unique_id = data_dict.get('idx', hashlib.md5(str(data_dict.get('question', '')).encode()).hexdigest())
        
        try:
            # Create input from question
            data_input = self._create_prm_input(data_dict)
            
            # Create outputs from processes
            data_output = self._create_prm_output(data_dict)
            
            # Create DataSample object
            data_sample = DataSample(
                unique_id=str(unique_id),
                input=data_input,
                output=data_output,
                source='prmbench',
                domain=data_dict.get('classification', 'reasoning'),
                metadata={
                    'classification': data_dict.get('classification'),
                    'modified_steps': data_dict.get('modified_steps', []),
                    'error_steps': data_dict.get('error_steps', []),
                    'reason': data_dict.get('reason'),
                    'idx': data_dict.get('idx'),
                    'original_process_length': len(data_dict.get('original_process', [])),
                    'modified_process_length': len(data_dict.get('modified_process', [])),
                    'load_strategy': 'PRMDataLoadStrategy'
                }
            )
            
            return data_sample
            
        except Exception as e:
            logger.error(f"Error creating DataSample from PRM data: {str(e)}")
            return None
    
    def _create_prm_input(self, data_dict: Dict[str, Any]) -> List[ChatMessage]:
        """Create DataInput from PRM question"""
        question = data_dict.get('question') or data_dict.get('original_question', '')
        
        return [ChatMessage(role='user', content=question)]
    
    def _create_prm_output(self, data_dict: Dict[str, Any]) -> List[DataOutput]:
        """Create DataOutput list from PRM processes"""
        outputs = []
        
        # Original process output
        if 'original_process' in data_dict:
            original_steps = []
            for i, step_content in enumerate(data_dict['original_process']):
                step = Step(
                    role='assistant',
                    content=step_content,
                    label={"correctness": "correct", "step_idx": i + 1}
                )
                original_steps.append(step)
            
            outputs.append(DataOutput(
                answer=Step(
                    role='assistant',
                    content='\n'.join(data_dict['original_process']),
                    label={"process_type": "original_correct"}
                ),
                steps=original_steps
            ))
        
        # Modified process output (with errors)
        if 'modified_process' in data_dict:
            modified_steps = []
            error_steps = set(data_dict.get('error_steps', []))
            
            for i, step_content in enumerate(data_dict['modified_process']):
                step_idx = i + 1
                is_correct = step_idx not in error_steps
                
                step = Step(
                    role='assistant',
                    content=step_content,
                    label={
                        "correctness": "correct" if is_correct else "error",
                        "step_idx": step_idx
                    }
                )
                modified_steps.append(step)
            
            # Calculate correctness score based on error ratio
            total_steps = len(data_dict['modified_process'])
            error_count = len(error_steps)
            
            outputs.append(DataOutput(
                answer=Step(
                    role='assistant',
                    content='\n'.join(data_dict['modified_process']),
                    label={"process_type": f"Modified process with {error_count}/{total_steps} error steps"},
                ),
                steps=modified_steps
                )
            )
        
        return outputs


class DataLoadModule(BaseDataModule):
    """Data Load Module - load data from various data sources"""
    
    load_strategy_type: str = Field(default="local", description="data load strategy type (local or remote)")
    data_source: str = Field(default="*", description="data source")
    dimension: str = Field(default="*", description="data dimension")
    load_config: Dict[str, Any] = Field(default_factory=dict, description="load config")
    
    def __init__(self, 
                 name: str,
                 config: Optional[Dict[str, Any]] = None,
                 load_strategy_type: str = "local",
                 data_source: str = "*",
                 dimension: str = "*",
                 load_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        initialize data load module
        
        Args:
            name: module name
            config: module config
            load_strategy_type: load strategy type
            data_source: data source
            dimension: data dimension
            load_config: load config
        """
        super().__init__(
            module_type=DataModuleType.LOAD,
            name=name,
            config=config,
            load_strategy_type=load_strategy_type,
            data_source=data_source,
            dimension=dimension,
            load_config=load_config or {},
            **kwargs
        )
    
    def run(self, input_data: Union[BaseDataSet, List[DataSample], None] = None, **kwargs) -> BaseDataSet:
        """load data"""
        try:
            
            # Get appropriate data loading strategy
            strategy_class = DataLoadStrategyRegistry.get_strategy_class(
                data_type=self.load_strategy_type,
                data_source=self.data_source,
                dimension=self.dimension
            )
            
            if not strategy_class:
                error_msg = f"No suitable data load strategy found for type: {self.load_strategy_type}, source: {self.data_source}, dimension: {self.dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Initialize and use strategy to load data
            # Add dimension to the config passed to strategy
            strategy_config = self.load_config.copy()
            strategy_config['dimension'] = self.dimension
            strategy = strategy_class(strategy_config)
            try:
                loaded_items = strategy.load_data()
            except Exception as load_error:
                logger.error(f"Error in strategy.load_data(): {str(load_error)}")
                raise load_error
            
            # Convert loaded items to DataSample objects if needed
            data_samples = []
            
            for item in loaded_items:
                data_samples.append(item)

            # Apply limit (if specified)
            if 'limit' in self.load_config and self.load_config['limit'] is not None and self.load_config['limit'] > 0:
                limit = min(int(self.load_config['limit']), len(data_samples))
                data_samples = random.sample(data_samples, limit)
                logger.info(f"Applied limit of {limit}, final count: {len(data_samples)}")
            
            # Create output dataset
            output_dataset = BaseDataSet(
                name=f"loaded_dataset_{self.data_source}",
                metadata={
                    "source": self.data_source,
                    "strategy_type": self.load_strategy_type,
                    "dimension": self.dimension,
                    "load_config": self.load_config
                },
                datas=data_samples
            )
            logger.info(f"Successfully loaded {len(data_samples)} items from {self.data_source}")
            
            return output_dataset
        except Exception as e:
            error_msg = f"Data loading failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


def create_load_module(name: str, 
                      config: Optional[Dict[str, Any]] = None,
                      load_strategy_type: str = "local",
                      data_source: str = "*",
                      dimension: str = "*",
                      load_config: Optional[Dict[str, Any]] = None
                      ) -> DataLoadModule:
    """create data load module factory function"""
    return DataLoadModule(
        name=name,
        config=config,
        load_strategy_type=load_strategy_type,
        data_source=data_source,
        dimension=dimension,
        load_config=load_config
    ) 