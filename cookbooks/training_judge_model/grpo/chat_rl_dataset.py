# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
from typing import List, Union

import datasets
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

# Note: Removed pydantic template classes to avoid Ray pickle serialization issues


logger = logging.getLogger(__name__)


class BaseChatRLDataset(Dataset):
    """Base class for chat reinforcement learning dataset."""

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,  # Keep for backward compatibility, but not used
        max_samples: int = -1,  # Add max_samples parameter
    ):
        # Initialize basic attributes
        self.data_files = self._normalize_data_files(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples

        # Load configuration settings
        self._load_config()

        # Load and process data
        self._load_dataset()

    def _normalize_data_files(self, data_files):
        """Convert data files to list format."""
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        return copy.deepcopy(data_files)

    def _load_config(self):
        """Load configuration parameters."""
        self.cache_dir = os.path.expanduser(self.config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.truncation = self.config.get("truncation", "error")
        self.filter_overlong_prompts = self.config.get("filter_overlong_prompts", True)
        self.num_workers = min(
            self.config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)), os.cpu_count()
        )
        self.serialize_dataset = False

    def _download_files(self):
        """Download files to local cache."""
        from verl.utils.fs import copy_to_local

        for i, file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=file, cache_dir=self.cache_dir)

    def _load_dataset(self):
        """Load and process dataset."""
        self._download_files()

        # Load parquet files
        dataframes = []
        for file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=file)["train"]
            dataframes.append(df)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        total = len(self.dataframe)
        logger.info(f"Dataset length: {total}")

        # Handle max_samples parameter
        if self.max_samples > 0 and self.max_samples < total:
            import numpy as np

            indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            logger.info(f"Selected {self.max_samples} samples (total: {total})")

        # Filter overlong prompts
        if self.filter_overlong_prompts:
            self._filter_long_prompts()

    def _filter_long_prompts(self):
        """Filter out overlong prompts."""
        # Extract tokenizer and params to local variables to avoid pickle serialization issues
        tokenizer = self.tokenizer
        max_length = self.max_prompt_length
        prompt_key = self.prompt_key

        def is_prompt_valid(doc):
            try:
                # Inline prompt extraction logic to avoid calling self methods
                prompt = ""
                if "input" in doc and doc["input"]:
                    for msg in doc["input"]:
                        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                            prompt = msg["content"]
                            break

                if not prompt:
                    # Fallback to other fields
                    prompt = doc.get(prompt_key, "")
                    if isinstance(prompt, list) and prompt:
                        prompt = prompt[0].get("content", "") if isinstance(prompt[0], dict) else str(prompt[0])

                if not prompt:
                    return True  # Keep sample if prompt cannot be extracted

                return len(tokenizer.encode(prompt)) <= max_length
            except Exception as e:
                logger.error(f"Error during filtering: {e}")
                return True  # Keep sample on error

        original_len = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            is_prompt_valid,
            num_proc=1,  # Use single process to avoid serialization issues
            desc=f"Filtering prompts exceeding {max_length} tokens",
        )
        logger.info(f"Dataset length after filtering: {len(self.dataframe)} (original: {original_len})")

    def _extract_prompt(self, example):
        """Extract prompt from example."""
        # First try new data structure
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    return msg["content"]

        # Fallback to old data structure
        prompt = example.get(self.prompt_key)
        if prompt is None:
            prompt = example.get("x", [])
            if prompt:
                return prompt[-1].get("content", "")

        if isinstance(prompt, str):
            return prompt[: self.max_prompt_length]
        elif isinstance(prompt, list) and prompt:
            return prompt[0].get("content", "") if isinstance(prompt[0], dict) else str(prompt[0])

        return ""

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
            "data_source": row_dict.get("source", "helpsteer2"),
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
            logger.info("Using old dataloader checkpoint file, recommend training from scratch")

    def __getstate__(self):
        """Get state for serialization."""
        state = self.__dict__.copy()
        state.pop("dataframe", None)
        return state


class PairwiseChatRLDataset(BaseChatRLDataset):
    """Pairwise chat reinforcement learning dataset."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        # Pairwise related configuration
        self.pairwise_response_index = self.config.get("pairwise_response_index", 0)  # Which response to train on
        logger.info(f"Using Pairwise mode, selected response index: {self.pairwise_response_index}")

    def _build_messages(self, example: dict) -> List[dict]:
        """Build chat messages from example - Pairwise mode."""
        messages = []

        # Extract user message from input field
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if msg.get("role") == "user" and msg.get("content"):
                    messages.append({"role": "user", "content": msg["content"]})

        # Pairwise mode: select the specified response
        if "output" in example and example["output"]:
            if self.pairwise_response_index < len(example["output"]):
                output_item = example["output"][self.pairwise_response_index]
                answer = output_item.get("answer", {})
                if isinstance(answer, dict) and answer.get("role") == "assistant":
                    content = answer.get("content", "")
                    if content:
                        messages.append({"role": "assistant", "content": content})

        # Fallback to original structure
        if len(messages) <= 1:
            prompt = self._extract_prompt(example)
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """Format pairwise template."""
        task_desc = """You are a professional expert in response comparison.
You will be provided with a query and two different responses (A and B) to that query.
Your task is to determine which response is better by comparing their quality across multiple dimensions.
Please consider the following principles in your evaluation and then indicate your preference."""

        principles = [
            "Helpfulness: How well does the response address the user's needs",
            "Accuracy: Factual correctness and reliability of information",
            "Safety: Avoiding harmful or inappropriate content",
        ]

        # Extract query
        query = next((msg["content"] for msg in messages if msg["role"] == "user"), "")

        # Get two responses
        response_a = ""
        response_b = ""

        if "output" in example and len(example["output"]) >= 2:
            response_a = example["output"][0].get("answer", {}).get("content", "")
            response_b = example["output"][1].get("answer", {}).get("content", "")

        # Use string formatting directly to avoid PairwiseTrainTemplate class (prevent pickle serialization issues)
        principles_str = "".join(f"{i}. {principle}\n" for i, principle in enumerate(principles, start=1))

        prompt = f"""# Task Description
{task_desc}
# Principles
{principles_str}
# Examples

# Query
{query}
# Response A
{response_a}
# Response B
{response_b}
# Output Format
<think>Analysis process based on principles</think><better>A or B</better>
"""
        return [{"role": "user", "content": prompt}]

    def _extract_ground_truth(self, row_dict):
        """Extract pairwise ground truth label."""
        try:
            output_data = row_dict.get("output", [])
            if output_data and len(output_data) >= 2:
                # Get label from selected response
                selected_answer = output_data[self.pairwise_response_index].get("answer", {})
                if isinstance(selected_answer, dict):
                    label_data = selected_answer.get("label", {})
                    if isinstance(label_data, dict):
                        # For pairwise, return preference information
                        preference = label_data.get("preference", "")
                        strength = label_data.get("preference_strength", 0)
                        response_id = label_data.get("response_id", "")

                        return {
                            "preference": preference,
                            "preference_strength": strength,
                            "response_id": response_id,
                            "task_type": "pairwise",
                        }

            return ""
        except:
            return ""


class PointwiseChatRLDataset(BaseChatRLDataset):
    """Pointwise chat reinforcement learning dataset - for single response quality scoring."""

    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(data_files, tokenizer, config, processor, max_samples)
        logger.info("Using Pointwise mode")

    def _build_messages(self, example: dict) -> List[dict]:
        """Build chat messages from example - Pointwise mode."""
        messages = []

        # Extract user message from input field
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    messages.append({"role": "user", "content": msg["content"]})

        # Pointwise mode: get first response
        if "output" in example and example["output"]:
            output_item = example["output"][0] if isinstance(example["output"], list) else example["output"]
            answer = output_item.get("answer", {}) if isinstance(output_item, dict) else {}
            if isinstance(answer, dict) and answer.get("role") == "assistant":
                content = answer.get("content", "")
                if content:
                    messages.append({"role": "assistant", "content": content})

        # Fallback to original structure
        if len(messages) <= 1:
            prompt = self._extract_prompt(example)
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        return messages

    def _format_template(self, messages: List[dict], example: dict) -> str:
        """Format pointwise template."""
        task_desc = """You are a professional expert in response quality evaluation.
You will be provided with a query and a response to that query.
Your task is to evaluate the quality of the response and assign a helpfulness score from 0 to 4.
Please consider the following principles in your evaluation."""

        principles = [
            "Helpfulness: How well does the response address the user's needs (0=not helpful, 4=extremely helpful)",
            "Accuracy: Factual correctness and reliability of information",
            "Clarity: How clearly and understandably the response is written",
            "Completeness: Whether the response fully addresses all aspects of the question",
            "Relevance: How directly related the response is to the question",
            "Safety: Avoiding harmful or inappropriate content",
        ]

        # Extract query
        query = next((msg["content"] for msg in messages if msg["role"] == "user"), "")

        # Get response
        response = ""
        if "output" in example and example["output"]:
            output_item = example["output"][0] if isinstance(example["output"], list) else example["output"]
            if isinstance(output_item, dict):
                response = output_item.get("answer", {}).get("content", "")

        # Use string formatting directly
        principles_str = "".join(f"{i}. {principle}\n" for i, principle in enumerate(principles, start=1))

        prompt = f"""# Task Description
{task_desc}
# Principles
{principles_str}
# Query
{query}
# Response
{response}
# Output Format
<think>Analysis process based on principles</think><score>helpfulness score from 0 to 4</score>
"""
        return [{"role": "user", "content": prompt}]

    def _extract_ground_truth(self, row_dict):
        """Extract pointwise ground truth label."""
        try:
            helpfulness = 0
            output_data = row_dict.get("output", [])
            if output_data:
                output_item = output_data[0] if isinstance(output_data, list) else output_data
                if isinstance(output_item, dict):
                    answer = output_item.get("answer", {})
                    if isinstance(answer, dict):
                        label_data = answer.get("label", {})
                        if isinstance(label_data, dict):
                            # For pointwise, return scoring information
                            helpfulness = label_data.get("helpfulness", 0)
            return {"helpfulness": helpfulness, "task_type": "pointwise"}
        except Exception as e:
            logger.error(f"Failed to extract label from {row_dict}: {e}")
            return {"helpfulness": helpfulness, "task_type": "pointwise"}


# Backward compatible aliases
