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
import os
from typing import List, Union

import datasets
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

from rm_gallery.examples.train.pointwise.template import Qwen3PointwiseTrainTemplate


class ChatRLDataset(Dataset):
    """simplified chat reinforcement learning dataset, for training chat data"""

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,  # 保持向后兼容性，但不使用
    ):
        # 初始化基本属性
        self.data_files = self._normalize_data_files(data_files)
        self.original_data_files = copy.deepcopy(self.data_files)
        self.tokenizer = tokenizer
        self.config = config

        # 加载配置设置
        self._load_config()

        # 加载和处理数据
        self._load_dataset()

    def _normalize_data_files(self, data_files):
        """convert data files to list format"""
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        return copy.deepcopy(data_files)

    def _load_config(self):
        """load config parameters"""
        self.cache_dir = os.path.expanduser(
            self.config.get("cache_dir", "~/.cache/verl/rlhf")
        )
        self.prompt_key = self.config.get("prompt_key", "prompt")
        self.max_prompt_length = self.config.get("max_prompt_length", 1024)
        self.return_raw_chat = self.config.get("return_raw_chat", False)
        self.truncation = self.config.get("truncation", "error")
        self.filter_overlong_prompts = self.config.get("filter_overlong_prompts", True)
        self.num_workers = min(
            self.config.get(
                "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
            ),
            os.cpu_count(),
        )
        self.serialize_dataset = False

    def _download_files(self):
        """download files to local cache"""
        from verl.utils.fs import copy_to_local

        for i, file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(src=file, cache_dir=self.cache_dir)

    def _load_dataset(self):
        """load and process dataset"""
        self._download_files()

        # load parquet file
        dataframes = []
        for file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=file)["train"]
            dataframes.append(df)

        self.dataframe = datasets.concatenate_datasets(dataframes)
        print(f"dataset length: {len(self.dataframe)}")

        # filter overlong prompts
        if self.filter_overlong_prompts:
            self._filter_long_prompts()

    def _filter_long_prompts(self):
        """filter out overlong prompts"""

        def is_prompt_valid(doc):
            try:
                prompt = self._extract_prompt(doc)
                return len(self.tokenizer.encode(prompt)) <= self.max_prompt_length
            except:
                return False

        self.dataframe = self.dataframe.filter(
            is_prompt_valid,
            num_proc=self.num_workers,
            desc=f"filter out prompts longer than {self.max_prompt_length} tokens",
        )
        print(f"filtered dataset length: {len(self.dataframe)}")

    def _extract_prompt(self, example):
        """extract prompt from example"""
        # try new data structure first
        if "input" in example and example["input"]:
            for msg in example["input"]:
                if msg.get("role") == "user" and msg.get("content"):
                    return msg["content"]

        # fallback to old data structure
        prompt = example.get(self.prompt_key)
        if prompt is None:
            prompt = example.get("x", [])
            if prompt:
                return prompt[-1].get("content", "")

        if isinstance(prompt, str):
            return prompt[: self.max_prompt_length]
        elif isinstance(prompt, list) and prompt:
            return (
                prompt[0].get("content", "")
                if isinstance(prompt[0], dict)
                else str(prompt[0])
            )

        return ""

    def _build_messages(self, example: dict) -> List[dict]:
        """build chat messages from example"""
        messages = []

        # extract all user messages from input field
        if "input" in example and example["input"]:
            for msg in example["input"]:
                # extract all user messages
                if msg.get("role") == "user" and msg.get("content"):
                    messages.append({"role": "user", "content": msg["content"]})
                # also consider extracting messages from other roles to keep conversation complete
                elif msg.get("role") in ["assistant", "system"] and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # extract assistant messages from output field
        if "output" in example and example["output"]:
            for output_item in example["output"]:
                answer = output_item.get("answer", {})
                if isinstance(answer, dict) and answer.get("role") == "assistant":
                    content = answer.get("content", "")
                    if content:
                        messages.append({"role": "assistant", "content": content})

        # fallback to original structure
        if not messages:
            prompt = self._extract_prompt(example)
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        return messages

    def _format_to_principle_template(self, messages: List[dict]) -> str:
        """format messages to principle template"""
        task_desc = """
            You are a professional expert in helpfulness evaluation.
            You will be provided with a query and an answer based on that query.
            Your task is to rate the answer on a scale of 0-4, where 0 means not helpful at all and 4 means extremely helpful.
            Please consider the following principles in your evaluation. please think firstly about the following principles and then answer the question with Output Format.
            """

        principles = [
            "Structure: Organize information logically",
            "Clarity: Ensure clear communication",
            "Accuracy: Ensure factual correctness",
            "Conciseness: Be brief yet informative",
            "Relevance: Focus on related information",
            "Engagement: Make content interactive",
            "Detail: Provide comprehensive specifics",
            "Practicality: Offer actionable advice",
            "Comprehensiveness: Cover all aspects",
            "Safety: Emphasize protective measures",
        ]

        # extract query and answer
        query = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        answer = next(
            (msg["content"] for msg in messages if msg["role"] == "assistant"), ""
        )

        prompt = Qwen3PointwiseTrainTemplate.format(
            desc=task_desc,
            principles=principles,
            examples="",
            query=query,
            context="",
            answer=answer,
        )
        return [{"role": "user", "content": prompt}]

    def _extract_ground_truth(self, row_dict):
        """extract ground truth from row data"""
        try:
            output_data = row_dict.get("output", [])
            if output_data:
                answer_data = output_data[0].get("answer", {})
                if isinstance(answer_data, dict):
                    label_data = answer_data.get("label", {})
                    if isinstance(label_data, dict):
                        return label_data.get("helpfulness", "") or label_data

            # fallback options
            return row_dict.get("ground_truth", "") or row_dict.get("answer", "")
        except:
            return ""

    def __getitem__(self, item):
        """get an item from dataset"""
        row_dict = dict(self.dataframe[item])
        messages = self._build_messages(row_dict)

        # format prompt
        raw_prompt_messages = self._format_to_principle_template(messages)

        raw_prompt = self.tokenizer.apply_chat_template(
            raw_prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

        # tokenize
        model_inputs = self.tokenizer(
            raw_prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # postprocess
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # compute position ids
        position_ids = compute_position_id_with_mask(attention_mask)

        # prepare raw prompt ids
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"提示长度 {len(raw_prompt_ids)} 超过 {self.max_prompt_length}"
                )

        # build result
        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "index": row_dict.get("index", item),
            "extra_info": copy.deepcopy(row_dict),
            "reward_model": {"ground_truth": self._extract_ground_truth(row_dict)},
            "data_source": row_dict.get("data_source", "helpsteer2"),
        }

        if self.return_raw_chat:
            result["raw_prompt"] = messages

        return result

    def __len__(self):
        return len(self.dataframe)

    def resume_dataset_state(self):
        """resume dataset state for checkpoint"""
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self.data_files = copy.deepcopy(self.original_data_files)
            self._load_dataset()
        else:
            print(
                "use old dataset loader checkpoint file, it is recommended to train from scratch"
            )

    def __getstate__(self):
        """get state for serialization"""
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()
