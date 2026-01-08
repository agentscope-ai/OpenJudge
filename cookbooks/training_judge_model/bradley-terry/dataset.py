# -*- coding: utf-8 -*-
"""
Bradley-Terry Dataset for Judge Model Training
- Loads preference data from parquet files
- Each sample contains chosen and rejected responses
- Returns data in format suitable for Bradley-Terry loss
- Uses chat template for Instruct models
"""

import json
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local


class BTDataset(Dataset):
    """
    Bradley-Terry Dataset for preference learning

    Expected data format in parquet:
    - chosen: messages list [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "yyy"}]
    - rejected: messages list [{"role": "user", "content": "xxx"}, {"role": "assistant", "content": "yyy"}]

    Data is processed with tokenizer.apply_chat_template() for Instruct models.
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: Union[str, PreTrainedTokenizer],
        config: Dict[str, Any],
    ) -> None:
        self.max_length = config.get("max_length", 4096)
        self.truncation = config.get("truncation", "left")
        self.use_shm = config.get("use_shm", False)

        # Keys for data columns
        self.chosen_key = config.get("chosen_key", "chosen")
        self.rejected_key = config.get("rejected_key", "rejected")

        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self) -> None:
        """Download parquet files to local if needed"""
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_process(self) -> None:
        """Read and concatenate all parquet files"""
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)

        self.dataframe = pd.concat(dataframes, ignore_index=True)

        # Extract chosen and rejected fields (JSON string format, parse to messages list)
        self.chosen_messages = [json.loads(msg) for msg in self.dataframe[self.chosen_key].tolist()]
        self.rejected_messages = [json.loads(msg) for msg in self.dataframe[self.rejected_key].tolist()]

        print(
            f"Loaded {len(self.chosen_messages)} preference pairs from {len(self.parquet_files)} files",
        )

    def __len__(self) -> int:
        return len(self.chosen_messages)

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply chat template to convert messages to model-expected format.

        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}, ...]
        """
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Remove BOS token if present
        if self.tokenizer.bos_token and formatted.startswith(self.tokenizer.bos_token):
            formatted = formatted[len(self.tokenizer.bos_token) :]
        return formatted

    def _tokenize_messages(self, messages: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Tokenize messages and handle truncation/padding to fixed length"""
        # Apply chat template
        text = self._apply_chat_template(messages)

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        sequence_length = input_ids.shape[0]

        # Handle sequence length like SFT dataset
        if sequence_length < self.max_length:
            # Pad sequences
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
            padded_input_ids = (
                torch.ones(
                    size=(self.max_length - sequence_length,),
                    dtype=input_ids.dtype,
                )
                * pad_token_id
            )
            padded_attention_mask = torch.zeros(
                size=(self.max_length - sequence_length,),
                dtype=attention_mask.dtype,
            )

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # Keep the end of the conversation (including conclusion)
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(
                    f"Sequence length {sequence_length} > max_length {self.max_length}",
                )

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def __getitem__(self, item: int) -> Dict[str, Any]:
        """
        Get a preference pair

        Returns:
            dict with keys:
            - input_ids_j: chosen response tokens
            - attention_mask_j: chosen response attention mask
            - input_ids_k: rejected response tokens
            - attention_mask_k: rejected response attention mask
        """
        chosen_messages = self.chosen_messages[item]
        rejected_messages = self.rejected_messages[item]

        # Tokenize both responses
        chosen_tokens = self._tokenize_messages(chosen_messages)
        rejected_tokens = self._tokenize_messages(rejected_messages)

        return {
            "input_ids_j": chosen_tokens["input_ids"],
            "attention_mask_j": chosen_tokens["attention_mask"],
            "input_ids_k": rejected_tokens["input_ids"],
            "attention_mask_k": rejected_tokens["attention_mask"],
        }
