import hashlib
from typing import Any, Dict

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("rewardbench2")
class RewardBench2Converter(DataConverter):
    """
    Unified converter for conversation data with prompt, chosen and rejected responses (version 2)
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> DataSample:
        """Convert conversation data to DataSample format"""
        # generate unique id using id field if available, otherwise use prompt content
        if "id" in data_dict:
            unique_id = str(data_dict["id"])
        else:
            content = str(data_dict.get("prompt", ""))
            unique_id = hashlib.md5(content.encode()).hexdigest()

        # Create input from prompt
        data_input = self._create_conversation_input(data_dict)

        # Create outputs from chosen/rejected responses
        data_output = self._create_conversation_output(data_dict)

        try:
            # Build metadata based on source type
            metadata = {
                "raw_data": data_dict,
                "load_strategy": "RewardBench2Converter",
                "subset": data_dict.get("subset"),
                "num_correct": data_dict.get("num_correct"),
                "num_rejected": data_dict.get("num_rejected"),
                "total_completions": data_dict.get("total_completions"),
                "models": data_dict.get("models"),
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
                source="rewardbench2",
                task_category="conversation",
                metadata=metadata,
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating conversation DataSample: {str(e)}")
            return None

    def _create_conversation_input(
        self, data_dict: Dict[str, Any]
    ) -> list[ChatMessage]:
        """Create DataInput from conversation prompt"""
        prompt = data_dict.get("prompt", "")

        # Since prompt is now a string, create a single user message
        if isinstance(prompt, str):
            return [ChatMessage(role="user", content=prompt)]
        else:
            # Fallback for backwards compatibility
            history = []
            if isinstance(prompt, list):
                for turn in prompt:
                    if isinstance(turn, dict):
                        role = turn.get("role", "user")
                        content = turn.get("content", str(turn))
                        history.append(ChatMessage(role=role, content=content))
                    else:
                        history.append(ChatMessage(role="user", content=str(turn)))
            else:
                history.append(ChatMessage(role="user", content=str(prompt)))

            return history

    def _create_conversation_output(
        self, data_dict: Dict[str, Any]
    ) -> list[DataOutput]:
        """Create DataOutput list from conversation responses"""
        outputs = []

        # Handle chosen responses (now a list of strings)
        chosen_responses = data_dict.get("chosen", [])
        if isinstance(chosen_responses, list):
            for chosen_content in chosen_responses:
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(chosen_content),
                            label={"preference": "chosen"},
                        ),
                    )
                )
        elif chosen_responses:  # Single chosen response (backwards compatibility)
            outputs.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=str(chosen_responses),
                        label={"preference": "chosen"},
                    ),
                )
            )

        # Handle rejected responses (now a list of strings)
        rejected_responses = data_dict.get("rejected", [])
        if isinstance(rejected_responses, list):
            for rejected_content in rejected_responses:
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(rejected_content),
                            label={"preference": "rejected"},
                        ),
                    )
                )
        elif rejected_responses:  # Single rejected response (backwards compatibility)
            outputs.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=str(rejected_responses),
                        label={"preference": "rejected"},
                    ),
                )
            )

        return outputs
