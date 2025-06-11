import hashlib
from typing import Any, Dict, List

from loguru import logger

from rm_gallery.core.data.load.base import (
    DataLoadStrategyRegistry,
    FileDataLoadStrategy,
)
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataLoadStrategyRegistry.register("local", "rmbbenchmark_bestofn")
class RMBBenchmarkBestOfNDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading conversation data with conversation_input, bon_best and loser_list responses
    """

    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert conversation data to DataSample format"""
        # Generate unique id using bon_uid
        if "bon_uid" in data_dict:
            unique_id = str(data_dict["bon_uid"])
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

        # Create outputs from bon_best and loser_list
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
                    "load_strategy": "RMBBenchmarkDataLoadStrategy",
                    "category_path": data_dict.get("category_path"),
                    "bon_uid": data_dict.get("bon_uid"),
                    "bon_best_model": data_dict.get("bon_best", {}).get("llm_name")
                    if data_dict.get("bon_best")
                    else None,
                    "loser_models": [
                        item.get("llm_name")
                        for item in data_dict.get("loser_list", [])
                        if isinstance(item, dict)
                    ],
                    "num_losers": len(data_dict.get("loser_list", [])),
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
        """Create DataOutput list from bon_best and loser_list"""
        outputs = []

        # Handle bon_best
        if "bon_best" in data_dict:
            bon_best = data_dict["bon_best"]
            if isinstance(bon_best, dict):
                answer_content = bon_best.get("answer", "")
                llm_name = bon_best.get("llm_name", "unknown")
                outputs.append(
                    DataOutput(
                        answer=Step(
                            role="assistant",
                            content=str(answer_content),
                            label={
                                "preference": "chosen",
                                "model": llm_name,
                                "type": "bon_best",
                            },
                        ),
                    )
                )

        # Handle loser_list
        if "loser_list" in data_dict:
            loser_list = data_dict["loser_list"]
            if isinstance(loser_list, list):
                for loser in loser_list:
                    if isinstance(loser, dict):
                        answer_content = loser.get("answer", "")
                        llm_name = loser.get("llm_name", "unknown")
                        outputs.append(
                            DataOutput(
                                answer=Step(
                                    role="assistant",
                                    content=str(answer_content),
                                    label={
                                        "preference": "rejected",
                                        "model": llm_name,
                                        "type": "loser",
                                    },
                                ),
                            )
                        )

        return outputs
