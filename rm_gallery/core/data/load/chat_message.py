"""
Chat Message Data Converter - specialized converter for chat message format data.
Handles conversation data with multiple messages and roles for chat-based training.
"""

import hashlib
from typing import Any, Dict

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataSample


@DataConverterRegistry.register("chat_message")
class ChatMessageConverter(DataConverter):
    """
    Specialized converter for chat message data format with conversation structure.

    Processes data containing message arrays with role/content pairs for
    chat-based reward modeling and conversation training.

    Input Data Format Expected:
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }

    Output: DataSample with structured input messages and empty output for inference
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> DataSample:
        """
        Convert chat message data dictionary to standardized DataSample format.

        Extracts conversation messages from input data and creates a DataSample
        with structured input for chat-based processing pipelines.

        Args:
            data_dict: Raw data containing messages array with role/content pairs
            source_info: Metadata about data source (file path, dataset name, etc.)

        Returns:
            DataSample with structured conversation input and metadata
            Returns None if conversion fails
        """
        # generate unique id
        content = str(data_dict)
        unique_id = hashlib.md5(content.encode()).hexdigest()

        try:
            # Create input from messages
            data_input = []
            messages = data_dict.get("messages", [])
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        data_input.append(ChatMessage(role=role, content=content))

            # Create simple output
            data_output = []

            # Build metadata based on source type
            metadata = {
                "raw_data": data_dict,
                "load_strategy": "ChatMessageConverter",
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
                input=data_input,
                output=data_output,
                source="chat_message",
                task_category="chat",
                metadata=metadata,
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating ChatMessage DataSample: {str(e)}")
            return None
