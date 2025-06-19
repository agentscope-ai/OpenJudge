"""
HuggingFace Generic Data Converter - flexible converter for various HuggingFace dataset formats.
Automatically detects and processes common data patterns from HuggingFace datasets.
"""

import hashlib
from typing import Any, Dict

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("*")
class GenericConverter(DataConverter):
    """
    Generic converter that automatically handles diverse HuggingFace dataset formats.

    Acts as a fallback converter when no specific format converter is available.
    Intelligently extracts input/output pairs from common field names and structures.

    Supported Input Patterns:
        - Fields: prompt, question, input, text, instruction (for input)
        - Fields: response, answer, output, completion (for output)
        - Messages: array of role/content objects for conversations

    Output: DataSample with auto-detected task category and structured data
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> DataSample:
        """
        Convert generic HuggingFace data dictionary to standardized DataSample format.

        Automatically detects input/output patterns from common field names,
        determines task category, and creates appropriate data structure.

        Args:
            data_dict: Raw data dictionary from HuggingFace dataset
            source_info: Source metadata including dataset name, config, split info

        Returns:
            DataSample with auto-detected structure and task category
            Returns None if input/output extraction fails
        """
        # Generate unique id
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()

        try:
            # Try to extract input from common field names
            input_data = self._extract_input(data_dict)
            if not input_data:
                logger.warning(f"Could not extract input from data: {data_dict}")
                return None

            # Try to extract output from common field names
            output_data = self._extract_output(data_dict)
            if not output_data:
                logger.warning(f"Could not extract output from data: {data_dict}")
                return None

            # Determine task category
            task_category = self._determine_task_category(data_dict)

            # Build metadata based on source type
            metadata = {
                "raw_data": data_dict,
                "load_strategy": "GenericConverter",
                "task_category": task_category,
            }

            # Add source-specific metadata
            if source_info.get("load_type") == "local":
                metadata.update(
                    {
                        "source_file_path": source_info.get("source_file_path"),
                        "load_type": "local",
                    }
                )
            elif source_info.get("load_type") == "huggingface":
                metadata.update(
                    {
                        "dataset_name": source_info.get("dataset_name"),
                        "dataset_config": source_info.get("dataset_config"),
                        "split": source_info.get("split", "train"),
                        "load_type": "huggingface",
                    }
                )

            data_sample = DataSample(
                unique_id=unique_id,
                input=input_data,
                output=output_data,
                source=source_info.get("dataset_name", "generic"),
                task_category=task_category,
                metadata=metadata,
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating generic DataSample: {str(e)}")
            return None

    def _extract_input(self, data_dict: Dict[str, Any]) -> list[ChatMessage]:
        """
        Extract input messages from data using common field name patterns.

        Searches for standard input field names and converts to ChatMessage format.
        Handles both single-field inputs and conversation message arrays.

        Args:
            data_dict: Raw data dictionary to extract input from

        Returns:
            List of ChatMessage objects representing the input context
        """
        input_data = []

        # Common input field names
        for field in ["prompt", "question", "input", "text", "instruction"]:
            if field in data_dict and data_dict[field]:
                input_data.append(
                    ChatMessage(role="user", content=str(data_dict[field]))
                )
                break

        # Handle conversation/messages format
        if "messages" in data_dict:
            messages = data_dict["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", str(msg))
                        if role in ["user", "system"]:  # Only include input messages
                            input_data.append(ChatMessage(role=role, content=content))

        return input_data

    def _extract_output(self, data_dict: Dict[str, Any]) -> list[DataOutput]:
        """
        Extract output responses from data using common field name patterns.

        Searches for standard output field names and creates DataOutput objects
        with Step components for response evaluation.

        Args:
            data_dict: Raw data dictionary to extract output from

        Returns:
            List of DataOutput objects representing expected responses
        """
        outputs = []

        # Common output field names
        for field in ["response", "answer", "output", "completion"]:
            if field in data_dict and data_dict[field]:
                outputs.append(
                    DataOutput(
                        answer=Step(role="assistant", content=str(data_dict[field]))
                    )
                )
                break

        # Handle messages format for assistant responses
        if "messages" in data_dict and not outputs:
            messages = data_dict["messages"]
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        outputs.append(
                            DataOutput(
                                answer=Step(
                                    role="assistant",
                                    content=str(msg.get("content", "")),
                                )
                            )
                        )

        return outputs

    def _determine_task_category(self, data_dict: Dict[str, Any]) -> str:
        """
        Automatically determine task category from data field patterns.

        Analyzes field names and structure to classify the type of task
        for appropriate processing and evaluation strategies.

        Args:
            data_dict: Raw data dictionary to analyze

        Returns:
            String identifier for the detected task category
        """
        # Check for explicit task category
        if "task_category" in data_dict:
            return str(data_dict["task_category"])

        # Infer from field names
        if any(field in data_dict for field in ["messages", "conversation"]):
            return "chat"
        elif any(field in data_dict for field in ["question", "answer"]):
            return "qa"
        elif any(field in data_dict for field in ["instruction", "completion"]):
            return "instruction_following"
        else:
            return "general"
