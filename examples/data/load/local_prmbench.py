import hashlib
from typing import Any, ClassVar, Dict, List

from loguru import logger

from rm_gallery.core.data.load.base import (
    DataLoadStrategyRegistry,
    FileDataLoadStrategy,
)
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step


@DataLoadStrategyRegistry.register("local", "prmbench")
class PRMDataLoadStrategy(FileDataLoadStrategy):
    """
    Strategy for loading Process Reward Model (PRM) data
    Handles mathematical reasoning data with step-wise processes
    """

    # define as class attribute instead of instance attribute
    DIMENSION_CLASSIFICATION_MAPPING: ClassVar[Dict[str, str]] = {
        "confidence": "confidence",
        "*": None,  # wildcard, means no filtering
    }

    def load_data(self, **kwargs) -> List[DataSample]:
        """Override load_data method, add dimension filtering"""
        # first call parent class method to load all data
        all_data = super().load_data(**kwargs)

        # get current dimension config
        current_dimension = self.metadata.get("dimension", "*")

        # if wildcard or no mapping, return all data
        if (
            current_dimension == "*"
            or current_dimension not in self.DIMENSION_CLASSIFICATION_MAPPING
        ):
            logger.info(f"No filtering applied for dimension: {current_dimension}")
            return all_data

        # get corresponding classification
        target_classification = self.DIMENSION_CLASSIFICATION_MAPPING[current_dimension]

        # filter data
        filtered_data = []
        for data_sample in all_data:
            if data_sample and data_sample.metadata:
                data_classification = data_sample.metadata.get("classification")
                if data_classification == target_classification:
                    filtered_data.append(data_sample)

        logger.info(
            f"Filtered data by dimension '{current_dimension}' -> classification '{target_classification}': "
            f"{len(all_data)} -> {len(filtered_data)} items"
        )

        return filtered_data

    def _convert_to_data_sample(self, data_dict: Dict[str, Any]) -> DataSample:
        """Convert PRM data to DataSample format

        Expected input format:
        {
            "original_question": "...",
            "modified_question": "...",
            "original_process": ["step1", "step2", ...],
            "modified_process": ["step1", "step2", ...],
            "modified_steps": [5, 6],
            "error_steps": [5, 6],
            "reason": "...",
            "idx": "...",
            "question": "...",
            "classification": "confidence"
        }
        """

        # Generate unique id from idx or question
        unique_id = data_dict.get(
            "idx", hashlib.md5(str(data_dict.get("question", "")).encode()).hexdigest()
        )

        try:
            # Create input from question
            data_input = self._create_prm_input(data_dict)

            # Create outputs from processes
            data_output = self._create_prm_output(data_dict)

            # Create DataSample object
            data_sample = DataSample(
                unique_id=str(unique_id),
                input=data_input,
                output=data_output,
                source="prmbench",
                task_category=data_dict.get("classification", "reasoning"),
                metadata={
                    "classification": data_dict.get("classification"),
                    "modified_steps": data_dict.get("modified_steps", []),
                    "error_steps": data_dict.get("error_steps", []),
                    "reason": data_dict.get("reason"),
                    "idx": data_dict.get("idx"),
                    "original_process_length": len(
                        data_dict.get("original_process", [])
                    ),
                    "modified_process_length": len(
                        data_dict.get("modified_process", [])
                    ),
                    "load_strategy": "PRMDataLoadStrategy",
                },
            )

            return data_sample

        except Exception as e:
            logger.error(f"Error creating DataSample from PRM data: {str(e)}")
            return None

    def _create_prm_input(self, data_dict: Dict[str, Any]) -> List[ChatMessage]:
        """Create DataInput from PRM question"""
        question = data_dict.get("question") or data_dict.get("original_question", "")

        return [ChatMessage(role="user", content=question)]

    def _create_prm_output(self, data_dict: Dict[str, Any]) -> List[DataOutput]:
        """Create DataOutput list from PRM processes"""
        outputs = []

        # Original process output
        if "original_process" in data_dict:
            original_steps = []
            for i, step_content in enumerate(data_dict["original_process"]):
                step = Step(
                    role="assistant",
                    content=step_content,
                    label={"correctness": "correct", "step_idx": i + 1},
                )
                original_steps.append(step)

            outputs.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content="\n".join(data_dict["original_process"]),
                        label={"process_type": "original_correct"},
                    ),
                    steps=original_steps,
                )
            )

        # Modified process output (with errors)
        if "modified_process" in data_dict:
            modified_steps = []
            error_steps = set(data_dict.get("error_steps", []))

            for i, step_content in enumerate(data_dict["modified_process"]):
                step_idx = i + 1
                is_correct = step_idx not in error_steps

                step = Step(
                    role="assistant",
                    content=step_content,
                    label={
                        "correctness": "correct" if is_correct else "error",
                        "step_idx": step_idx,
                    },
                )
                modified_steps.append(step)

            # Calculate correctness score based on error ratio
            total_steps = len(data_dict["modified_process"])
            error_count = len(error_steps)

            outputs.append(
                DataOutput(
                    answer=Step(
                        role="assistant",
                        content="\n".join(data_dict["modified_process"]),
                        label={
                            "process_type": f"Modified process with {error_count}/{total_steps} error steps"
                        },
                    ),
                    steps=modified_steps,
                )
            )

        return outputs
