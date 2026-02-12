import copy
import logging
import os
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Union

import datasets
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration for different datasets and templates."""

    dataset_name: str = "default"
    prompt_template: str = ""
    input_field: str = "input"
    output_field: str = "output"
    response_field: str = "answer"
    label_field: str = "label"
    score_field: str = "score"
    max_samples: int = -1
    custom_principles: List[str] = field(default_factory=list)
    custom_task_description: str = ""
    template_config: Dict[str, Any] = field(default_factory=dict)
    file_format: str = ""
    file_formats_supported: List[str] = field(default_factory=lambda: ["parquet", "json", "jsonl"])


class BaseChatRLDataset(Dataset):
    """Base class for chat reinforcement learning dataset."""

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: Union[DictConfig, Dict[str, Any]],
        processor=None,  # Keep for backward compatibility, but not used
        max_samples: int = -1,  # Add max_samples parameter
    ):
        # Initialize basic attributes
        self.data_files = self._normalize_data_files(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples

        # Parse configuration - support both DictConfig and regular dict
        self._parse_config(config)

        # Validate file formats and load data
        self._validate_file_formats()
        self._load_dataset()

    def _normalize_data_files(self, data_files):
        """Convert data files to list format."""
        if isinstance(data_files, str):
            data_files = [data_files]
        elif hasattr(data_files, "_iter_"):  # Handle ListConfig or similar
            data_files = list(data_files)
        return copy.deepcopy(data_files)

    def _parse_config(self, config: Union[DictConfig, Dict[str, Any]]):
        """Parse configuration parameters from either DictConfig or dict."""
        if hasattr(config, "__dict__") or isinstance(config, dict):
            config_dict = dict(config) if hasattr(config, "__dict__") else config
        else:
            config_dict = {}

        # Load configuration settings with defaults
        self.cache_dir = os.path.expanduser(config_dict.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config_dict.get("prompt_key", "prompt")
        self.max_prompt_length = config_dict.get("max_prompt_length", 1024)
        self.return_raw_chat = config_dict.get("return_raw_chat", False)
        self.truncation = config_dict.get("truncation", "error")
        self.filter_overlong_prompts = config_dict.get("filter_overlong_prompts", True)
        self.num_workers = min(
            config_dict.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)), os.cpu_count()
        )
        self.serialize_dataset = False

        # Add dataset-specific configuration
        self.dataset_config_dict = config_dict.get("dataset_config", {})
        self.input_field = self.dataset_config_dict.get("input_field", "input")
        self.output_field = self.dataset_config_dict.get("output_field", "output")
        self.file_format = self.dataset_config_dict.get("file_format", "")

    def _validate_file_formats(self):
        """Validate file formats and determine actual formats."""
        validated_files = []
        for file_path in self.data_files:
            actual_format = self._detect_file_format(file_path)
            if actual_format not in ["parquet", "json", "jsonl"]:
                raise ValueError(
                    f"Unsupported file format for {file_path}: {actual_format}. "
                    f"Supported formats: parquet, json, jsonl"
                )
            validated_files.append((file_path, actual_format))

        self.validated_files = validated_files
        pprint(f"Validated {len(validated_files)} files with formats: {[fmt for _, fmt in validated_files]}")

    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format based on extension or auto-detection."""
        if self.file_format:
            return self.file_format

        _, ext = os.path.splitext(file_path.lower())
        if ext in [".parquet", ".json", ".jsonl"]:
            return ext[1:]
        else:
            # Try to detect format by attempting to load
            try:
                # Test if it's parquet
                test_ds = datasets.load_dataset("parquet", data_files=file_path)
                return "parquet"
            except:
                try:
                    # Test if it's json
                    test_ds = datasets.load_dataset("json", data_files=file_path)
                    return "json"
                except:
                    raise ValueError(f"Cannot determine format for file: {file_path}")

    def _load_dataset(self):
        """Load and process dataset from multiple files of different formats."""
        self._download_files()

        # Load dataframes from different formats
        dataframes = []
        for file_path, file_format in self.validated_files:
            df = self._load_single_file(file_path, file_format)
            dataframes.append(df)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        pprint(f"Combined dataset length: {total}")

        # Handle max_samples parameter
        if self.max_samples > 0 and self.max_samples < total:
            import numpy as np

            indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            pprint(f"Selected {self.max_samples} samples (total: {total})")

        # Filter overlong prompts
        if self.filter_overlong_prompts:
            self._filter_long_prompts()

    def _download_files(self):
        """Download files to local cache."""
        from verl.utils.fs import copy_to_local

        downloaded_files = []
        for file_path, file_format in self.validated_files:
            downloaded_path = copy_to_local(src=file_path, cache_dir=self.cache_dir)
            downloaded_files.append((downloaded_path, file_format))

        self.validated_files = downloaded_files

    def _load_single_file(self, file_path: str, file_format: str) -> Dataset:
        """Load a single file based on its format."""
        pprint(f"Loading {file_format} file: {file_path}")

        if file_format == "parquet":
            dataset = datasets.load_dataset("parquet", data_files=file_path)["train"]
        elif file_format in ["json", "jsonl"]:
            dataset = datasets.load_dataset("json", data_files=file_path)["train"]
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        pprint(f"Loaded {len(dataset)} samples from {file_path}")
        return dataset

    def _filter_long_prompts(self):
        """Filter out overlong prompts."""
        # Extract tokenizer and params to local variables to avoid pickle serialization issues
        tokenizer = self.tokenizer
        max_length = self.max_prompt_length
        prompt_key = self.prompt_key
        input_field = self.input_field

        def is_prompt_valid(doc):
            try:
                # Inline prompt extraction logic - handles both nested and flat structures
                prompt = None

                # Try nested structure first: doc[input_field][prompt_key]
                if input_field and input_field in doc:
                    inner = doc[input_field]
                    if isinstance(inner, dict) and prompt_key in inner:
                        prompt = inner[prompt_key]
                    elif isinstance(inner, str):
                        prompt = inner  # Fallback: input_field might directly contain the prompt string

                # Fallback to top-level prompt_key if nested extraction failed
                if prompt is None and prompt_key in doc:
                    prompt = doc[prompt_key]

                # Keep samples where prompt can't be extracted (safer than dropping data)
                if not prompt or not isinstance(prompt, str):
                    return True

                # Check token length
                return len(tokenizer.encode(prompt)) <= max_length

            except Exception as e:
                logger.error(f"Error during filtering: {e}")
                return True  # Keep sample on error to avoid data loss

        original_len = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            is_prompt_valid,
            num_proc=1,  # Use single process to avoid serialization issues
            desc=f"Filtering prompts exceeding {max_length} tokens",
        )
        pprint(f"Dataset length after filtering: {len(self.dataframe)} (original: {original_len})")

    def _extract_prompt_from_doc(self, doc: dict, input_field: str, prompt_key: str) -> str:
        """Extract prompt from document supporting multiple formats."""
        # Handle the new JSON structure with input.query
        if input_field in doc:
            input_data = doc[input_field]
            if isinstance(input_data, dict) and "query" in input_data:
                return input_data["query"]
            elif isinstance(input_data, list):
                for msg in input_data:
                    if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                        return msg["content"]
            elif isinstance(input_data, str):
                return input_data

        # Fallback to old data structure
        prompt = doc.get(prompt_key)
        if prompt is None:
            prompt = doc.get("x", [])
            if prompt:
                return prompt[-1].get("content", "")

        if isinstance(prompt, str):
            return prompt[: self.max_prompt_length]
        elif isinstance(prompt, list) and prompt:
            return prompt[0].get("content", "") if isinstance(prompt[0], dict) else str(prompt[0])

        return ""

    def _extract_prompt(self, example):
        """Extract prompt from example - supports configurable field names."""
        return self._extract_prompt_from_doc(example, self.input_field, self.prompt_key)

    def _build_messages(self, example: dict) -> List[dict]:
        """Build chat messages from example - subclasses must override."""
        raise NotImplementedError("Subclasses must implement _build_messages")

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """Format template - subclasses must override."""
        raise NotImplementedError("Subclasses must implement _format_template")

    def _extract_ground_truth(self, row_dict):
        """Extract ground truth label - subclasses must override."""
        raise NotImplementedError("Subclasses must implement _extract_ground_truth")

    def __getitem__(self, item):
        """Get an item from the dataset."""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)

        # Format prompt
        raw_prompt_messages = self._format_template(messages, row_dict)

        # Try using enable_thinking parameter, fallback if not supported
        try:
            raw_prompt = self.tokenizer.apply_chat_template(
                raw_prompt_messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
            )
        except TypeError:
            # If tokenizer doesn't support enable_thinking parameter, skip it
            raw_prompt = self.tokenizer.apply_chat_template(
                raw_prompt_messages, add_generation_prompt=True, tokenize=False
            )

        # Tokenize
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Post-process
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)

        # Prepare raw prompt IDs
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}")

        # Build result
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": row_dict.get("source", "default"),
        }

        if self.return_raw_chat:
            result["raw_prompt"] = messages

        return result

    def __len__(self):
        return len(self.dataframe)

    def resume_dataset_state(self):
        """Resume dataset state for checkpointing."""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self.data_files = copy.deepcopy(self.original_data_files)
            self._load_dataset()
        else:
            pprint("Using old dataloader checkpoint file, recommend training from scratch")

    def __getstate__(self):
        """Get state for serialization."""
        state = self.__dict__.copy()
        state.pop("dataframe", None)
        return state

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        return {
            "total_samples": len(self),
            "data_files": self.data_files,
            "validated_files": [(path, fmt) for path, fmt in self.validated_files],
            "input_field": getattr(self, "input_field", "input"),
            "output_field": getattr(self, "output_field", "output"),
            "max_prompt_length": self.max_prompt_length,
            "filter_overlong_prompts": self.filter_overlong_prompts,
        }


class PointwiseChatRLDataset(BaseChatRLDataset):
    """Pointwise chat reinforcement learning dataset - for single response quality scoring."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        # Parse dataset config before calling parent constructor
        self.dataset_config = self._parse_dataset_config(config)
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        pprint(f"Using Pointwise mode with dataset: {self.dataset_config.dataset_name}")
        pprint(f"Prompt template: {self.dataset_config.prompt_template}")

    def _parse_dataset_config(self, config) -> DatasetConfig:
        """Parse dataset configuration from config."""
        if hasattr(config, "dataset_config"):
            dataset_config_dict = config.dataset_config
            if isinstance(dataset_config_dict, DatasetConfig):
                return dataset_config_dict
            elif isinstance(dataset_config_dict, dict):
                config_dict = {**dataset_config_dict}
                return DatasetConfig(**config_dict)
        elif isinstance(config, dict) and "dataset_config" in config:
            dataset_config_dict = config["dataset_config"]
            if isinstance(dataset_config_dict, DatasetConfig):
                return dataset_config_dict
            elif isinstance(dataset_config_dict, dict):
                config_dict = {**dataset_config_dict}
                return DatasetConfig(**config_dict)

        return DatasetConfig()

    def _parse_config(self, config: Union[DictConfig, Dict[str, Any]]):
        """Override parent method to incorporate dataset config."""
        super()._parse_config(config)

        # Use the parsed config dict from parent
        self.input_field = self.dataset_config_dict.get("input_field", "input")
        self.output_field = self.dataset_config_dict.get("output_field", "output")
        self.file_format = self.dataset_config_dict.get("file_format", "")

    def _build_messages(self, example: dict) -> List[dict]:
        """Build chat messages from example - Pointwise mode with text format only."""
        messages = []

        # Check if it's the new JSON structure (has 'input' key with nested structure)
        if "input" in example and isinstance(example["input"], dict) and "query" in example["input"]:
            # New JSON format
            query = example["input"].get("query", "")
            if query:
                messages.append({"role": "user", "content": query})

            # Get chosen response (positive example)
            if "chosen" in example and isinstance(example["chosen"], dict):
                response_data = example["chosen"].get("response", {})
                if isinstance(response_data, dict):
                    response_content = response_data.get("content", "")
                    if response_content:
                        messages.append({"role": "assistant", "content": response_content})
        else:
            # Old format - handle standard structure
            messages = self._build_old_format_messages(example)

        return messages

    def _build_old_format_messages(self, example: dict) -> List[dict]:
        """Build messages in old format for backward compatibility."""
        messages = []

        # Extract user message from input field
        input_key = self.dataset_config.input_field
        if input_key in example and example[input_key]:
            input_data = example[input_key]
            if isinstance(input_data, list):
                for msg in input_data:
                    if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                        messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(input_data, str):
                messages.append({"role": "user", "content": input_data})

        # Pointwise mode: get first response
        output_key = self.dataset_config.output_field
        if output_key in example and example[output_key]:
            output_data = example[output_key]
            output_item = output_data[0] if isinstance(output_data, list) else output_data
            response_key = self.dataset_config.response_field

            if isinstance(output_item, dict):
                answer = output_item.get(response_key, {})
                if isinstance(answer, dict) and answer.get("role") == "assistant":
                    content = answer.get("content", "")
                    if content:
                        messages.append({"role": "assistant", "content": content})
                elif isinstance(answer, str):
                    messages.append({"role": "assistant", "content": answer})

        # Fallback to original structure
        if len(messages) <= 1:
            prompt = self._extract_prompt(example)
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """Format template based on configured prompt template type."""
        # Import the specific prompts from openjudge
        try:
            from openjudge.graders.common.correctness import CORRECTNESS_PROMPT_EN
        except ImportError:
            # Define default if import fails
            CORRECTNESS_PROMPT_EN = (
                "Evaluate the factual correctness of the response. Query: {query}, Response: {response}"
            )

        try:
            from openjudge.graders.common.hallucination import HALLUCINATION_PROMPT_EN
        except ImportError:
            # Define default if import fails
            HALLUCINATION_PROMPT_EN = "Check for hallucinations in the response. Query: {query}, Response: {response}"

        try:
            from openjudge.graders.common.relevance import RELEVANCE_PROMPT_EN
        except ImportError:
            # Define default if import fails
            RELEVANCE_PROMPT_EN = (
                "Evaluate the relevance of the response to the query. Query: {query}, Response: {response}"
            )

        try:
            from openjudge.graders.common.harmfulness import HARMFULNESS_PROMPT_EN
        except ImportError:
            # Define default if import fails
            HARMFULNESS_PROMPT_EN = (
                "Evaluate the harmfulness of the response to the query. Query: {query}, Response: {response}"
            )

        try:
            from openjudge.graders.common.instruction_following import (
                INSTRUCTION_FOLLOWING_PROMPT_EN,
            )
        except ImportError:
            # Define default if import fails
            INSTRUCTION_FOLLOWING_PROMPT_EN = (
                "Evaluate the instruction_following of the response to the query. Query: {query}, Response: {response}"
            )

        task_type = example.get("task_type", "unknown")

        if task_type == "correctness":
            grader_template = CORRECTNESS_PROMPT_EN
        elif task_type == "hallucination":
            grader_template = HALLUCINATION_PROMPT_EN
        elif task_type == "relevance":
            grader_template = RELEVANCE_PROMPT_EN
        elif task_type == "harmlessness":
            grader_template = HARMFULNESS_PROMPT_EN
        elif task_type == "instruction_following":
            grader_template = INSTRUCTION_FOLLOWING_PROMPT_EN
        else:
            # Default to correctness if unknown template
            pprint(f"task type: {task_type}")
            grader_template = CORRECTNESS_PROMPT_EN
        return self._format_grader_template(messages, example, grader_template)

    def _format_grader_template(self, messages: List[dict], example: dict, grader_prompt: str) -> str:
        """Format correctness evaluation template using openjudge prompt."""
        if "input" in example and isinstance(example["input"], dict) and "query" in example["input"]:
            # New JSON format
            query = example["input"].get("query", "")
            context = example["input"].get("context") or ""  # Handle null value
            reference_response = example["input"].get("reference", "")

            response = ""
            if "answer" in example and isinstance(example["answer"], dict):
                answer_response = example["answer"].get("response", {})
                if isinstance(answer_response, dict):
                    response = answer_response.get("content", "")
            # Also try 'response' field as fallback
            elif "response" in example and isinstance(example["response"], dict):
                response = example["response"].get("content", "")
        else:
            # Old format - extract from messages
            query = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
            response = self._get_response_content(example)

        # Replace placeholders in the grader prompt
        formatted_prompt = (
            grader_prompt.replace("{query}", query if query is not None else "")
            .replace("{response}", response if response is not None else "")
            .replace("{reference_response}", reference_response if reference_response is not None else "")
            .replace("{context}", str(context) if context is not None else "")
        )

        return [{"role": "user", "content": formatted_prompt}]

    def _get_response_content(self, example: dict) -> str:
        """Helper method to extract response content consistently."""
        # Check if it's the new JSON structure
        if "input" in example and isinstance(example["input"], dict) and "query" in example["input"]:
            # New JSON format - get from chosen response
            if "chosen" in example and isinstance(example["chosen"], dict):
                response_data = example["chosen"].get("response", {})
                if isinstance(response_data, dict):
                    return response_data.get("content", "")
        else:
            # Old format - use original logic
            output_key = self.dataset_config.output_field
            response_key = self.dataset_config.response_field

            if output_key in example and example[output_key]:
                output_data = example[output_key]
                output_item = output_data[0] if isinstance(output_data, list) else output_data
                if isinstance(output_item, dict):
                    answer = output_item.get(response_key, {})
                    if isinstance(answer, dict):
                        return answer.get("content", "")
                    elif isinstance(answer, str):
                        return answer
        return ""

    def _extract_ground_truth(self, row_dict):
        """Extract pointwise ground truth label with configurable fields."""
        try:
            score_value = 0
            # Check if it's the new JSON structure
            if "input" in row_dict and isinstance(row_dict["input"], dict) and "query" in row_dict["input"]:
                # New JSON format - extract score value
                if "score" in row_dict:
                    score_value = row_dict["score"]
            else:
                # Old format - use original logic
                output_key = self.dataset_config.output_field
                label_key = self.dataset_config.label_field
                score_key = self.dataset_config.score_field

                output_data = row_dict.get(output_key, [])
                if output_data:
                    output_item = output_data[0] if isinstance(output_data, list) else output_data
                    if isinstance(output_item, dict):
                        answer = output_item.get(self.dataset_config.response_field, {})
                        if isinstance(answer, dict):
                            label_data = answer.get(label_key, {})
                            if isinstance(label_data, dict):
                                score_value = label_data.get(score_key, 0)
                            elif isinstance(label_data, (int, float)):
                                score_value = label_data
                            elif isinstance(label_data, str):
                                try:
                                    score_value = float(label_data)
                                except ValueError:
                                    score_value = 0

            return {
                self.dataset_config.score_field: score_value,
                "task_type": "pointwise",
                "prompt_template": self.dataset_config.prompt_template,
            }
        except Exception as e:
            pprint(f"Failed to extract label from {row_dict}: {e}")
            return {
                self.dataset_config.score_field: 0,
                "task_type": "pointwise",
                "prompt_template": self.dataset_config.prompt_template,
            }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset configuration."""
        base_info = super().get_dataset_info()
        base_info.update(
            {
                "dataset_name": self.dataset_config.dataset_name,
                "prompt_template": self.dataset_config.prompt_template,
                "input_field": self.dataset_config.input_field,
                "output_field": self.dataset_config.output_field,
                "response_field": self.dataset_config.response_field,
                "label_field": self.dataset_config.label_field,
                "score_field": self.dataset_config.score_field,
                "file_format": self.dataset_config.file_format,
                "config": self.dataset_config.__dict__,
            }
        )
        return base_info
