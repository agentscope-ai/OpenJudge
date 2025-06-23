import hashlib
from typing import Any, Dict, List, Union

from loguru import logger

from rm_gallery.core.data.load.base import DataConverter, DataConverterRegistry
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataConverterRegistry.register("helpsteer2_pairwise")
class HelpSteer2PairwiseConverter(DataConverter):
    """
    Converter for HelpSteer2 pairwise data format
    Can handle data from both local files and HuggingFace Hub
    Converts each data entry into two DataSamples with swapped responses
    """

    def convert_to_data_sample(
        self, data_dict: Dict[str, Any], source_info: Dict[str, Any]
    ) -> Union[DataSample, List[DataSample]]:
        """Convert HelpSteer2 pairwise data to DataSample format"""

        try:
            # Create input from prompt
            data_input = [ChatMessage(role="user", content=data_dict["prompt"])]

            # Determine preference based on preference_strength
            preference_strength = data_dict.get("preference_strength", 0)
            if preference_strength > 0:
                # response_2 is better
                preferred_in_original = "response_2"
            elif preference_strength < 0:
                # response_1 is better
                preferred_in_original = "response_1"
            else:
                # tie
                preferred_in_original = "tie"

            data_samples = []

            # Create first sample: response_A = response_1, response_B = response_2
            sample1_id = hashlib.md5(f"{str(data_dict)}_sample1".encode()).hexdigest()

            # Determine preferred for first sample
            if preferred_in_original == "response_1":
                preferred_1 = "A"  # response_A (response_1) is preferred
            elif preferred_in_original == "response_2":
                preferred_1 = "B"  # response_B (response_2) is preferred
            else:
                preferred_1 = "tie"

            # Create outputs for first sample
            output_1 = [
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=data_dict["response_1"],
                        label={"response_type": "A"},
                    )
                ),
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=data_dict["response_2"],
                        label={"response_type": "B"},
                    )
                ),
            ]

            # Build metadata for first sample
            metadata_1 = {
                "raw_data": data_dict,
                "load_strategy": "HelpSteer2PairwiseConverter",
                "response_A": data_dict["response_1"],
                "response_B": data_dict["response_2"],
                "preferred": preferred_1,
                "preference_strength": preference_strength,
                "preference_statement": data_dict.get("preference_statement"),
                "preference_elaboration": data_dict.get("preference_elaboration"),
                "sample_type": "original_order",
            }

            # Add source-specific metadata
            if source_info.get("load_type") == "local":
                metadata_1.update(
                    {
                        "source_file_path": source_info.get("source_file_path"),
                        "load_type": "local",
                    }
                )
            elif source_info.get("load_type") == "huggingface":
                metadata_1.update(
                    {
                        "dataset_name": source_info.get(
                            "dataset_name", "nvidia/HelpSteer2"
                        ),
                        "dataset_config": source_info.get("dataset_config"),
                        "split": source_info.get("split", "train"),
                        "load_type": "huggingface",
                    }
                )

            sample_1 = DataSample(
                unique_id=sample1_id,
                input=data_input,
                output=output_1,
                source="helpsteer2_pairwise",
                task_category="chat_pairwise",
                metadata=metadata_1,
            )
            data_samples.append(sample_1)

            # Create second sample: response_A = response_2, response_B = response_1 (swapped)
            sample2_id = hashlib.md5(f"{str(data_dict)}_sample2".encode()).hexdigest()

            # Determine preferred for second sample (swapped)
            if preferred_in_original == "response_1":
                preferred_2 = "B"  # response_B (response_1) is preferred
            elif preferred_in_original == "response_2":
                preferred_2 = "A"  # response_A (response_2) is preferred
            else:
                preferred_2 = "tie"

            # Create outputs for second sample (swapped)
            output_2 = [
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=data_dict["response_2"],
                        label={"response_type": "A"},
                    )
                ),
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content=data_dict["response_1"],
                        label={"response_type": "B"},
                    )
                ),
            ]

            # Build metadata for second sample
            metadata_2 = {
                "raw_data": data_dict,
                "load_strategy": "HelpSteer2PairwiseConverter",
                "response_A": data_dict["response_2"],
                "response_B": data_dict["response_1"],
                "preferred": preferred_2,
                "preference_strength": preference_strength,
                "preference_statement": data_dict.get("preference_statement"),
                "preference_elaboration": data_dict.get("preference_elaboration"),
                "sample_type": "swapped_order",
            }

            # Add source-specific metadata
            if source_info.get("load_type") == "local":
                metadata_2.update(
                    {
                        "source_file_path": source_info.get("source_file_path"),
                        "load_type": "local",
                    }
                )
            elif source_info.get("load_type") == "huggingface":
                metadata_2.update(
                    {
                        "dataset_name": source_info.get(
                            "dataset_name", "nvidia/HelpSteer2"
                        ),
                        "dataset_config": source_info.get("dataset_config"),
                        "split": source_info.get("split", "train"),
                        "load_type": "huggingface",
                    }
                )

            sample_2 = DataSample(
                unique_id=sample2_id,
                input=data_input,
                output=output_2,
                source="helpsteer2_pairwise",
                task_category="chat_pairwise",
                metadata=metadata_2,
            )
            data_samples.append(sample_2)

            return data_samples

        except Exception as e:
            logger.error(f"Error creating HelpSteer2 Pairwise DataSample: {str(e)}")
            return None
