import hashlib
from typing import Any, Dict, List

from loguru import logger

from rm_gallery.core.data.load.base import (
    DataLoadStrategyRegistry,
    FileDataLoadStrategy,
)
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataLoadStrategyRegistry.register("local", "rmbbenchmark_pairwise")
class RMBBenchmarkPairwiseDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading conversation data with conversation_input, chosen and reject responses
    """

    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert conversation data to DataSample format"""
        # Generate unique id using pair_uid
        if "pair_uid" in data_dict:
            unique_id = str(data_dict["pair_uid"])
        else:
            # Use conversation_input content for generating hash
            conversation_input = data_dict.get("conversation_input", [])
            if (
                conversation_input
                and isinstance(conversation_input, list)
                and len(conversation_input) > 0
            ):
                content = str(conversation_input[0].get("content", ""))
            else:
                content = ""
            unique_id = hashlib.md5(content.encode()).hexdigest()

        # Create input from conversation_input
        data_input = self._create_conversation_input(data_dict)

        # Create outputs from chosen and reject
        data_output = self._create_conversation_output(data_dict)

        try:
            data_sample = DataSample(
                unique_id=unique_id,
                input=data_input,
                output=data_output,
                source="rewardbench",
                task_category=self.config.get("task_category", "conversation"),
                metadata={
                    "raw_data": data_dict,
                    "load_strategy": "RMBBenchmarkPairwiseDataLoadStrategy",
                    "category_path": data_dict.get("category_path"),
                    "pair_uid": data_dict.get("pair_uid"),
                    "chosen_model": data_dict.get("chosen", {}).get("llm_name")
                    if data_dict.get("chosen")
                    else None,
                    "reject_model": data_dict.get("reject", {}).get("llm_name")
                    if data_dict.get("reject")
                    else None,
                },
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating conversation DataSample: {str(e)}")
            return None

    def _create_conversation_input(
        self, data_dict: Dict[str, Any]
    ) -> List[ChatMessage]:
        """Create DataInput from conversation_input"""
        conversation_input = data_dict.get("conversation_input", [])
        if isinstance(conversation_input, list):
            history = []
            for message in conversation_input:
                if isinstance(message, dict):
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    history.append(ChatMessage(role=role, content=content))
                else:
                    history.append(ChatMessage(role="user", content=str(message)))
            return history
        else:
            return [ChatMessage(role="user", content=str(conversation_input))]

    def _create_conversation_output(
        self, data_dict: Dict[str, Any]
    ) -> List[DataOutput]:
        """Create DataOutput list from chosen and reject"""
        outputs = []

        # Handle chosen
        if "chosen" in data_dict:
            chosen = data_dict["chosen"]
            if isinstance(chosen, dict):
                answer_content = chosen.get("answer", "")
                llm_name = chosen.get("llm_name", "unknown")
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(answer_content),
                            label={
                                "preference": "chosen",
                                "model": llm_name,
                                "type": "chosen",
                            },
                        ),
                    )
                )

        # Handle reject
        if "reject" in data_dict:
            reject = data_dict["reject"]
            if isinstance(reject, dict):
                answer_content = reject.get("answer", "")
                llm_name = reject.get("llm_name", "unknown")
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(answer_content),
                            label={
                                "preference": "rejected",
                                "model": llm_name,
                                "type": "reject",
                            },
                        ),
                    )
                )

        return outputs
