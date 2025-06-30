"""
Data Annotation Module - comprehensive Label Studio integration for manual data annotation workflows.

Provides end-to-end annotation capabilities including project creation, task import,
annotation collection, and export processing for reward model training pipelines.
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import Field

from rm_gallery.core.data.annotation.client import LabelStudioClient
from rm_gallery.core.data.annotation.template import AnnotationTemplateRegistry
from rm_gallery.core.data.base import BaseDataModule, DataModuleType
from rm_gallery.core.data.schema import (
    BaseDataSet,
    ChatMessage,
    DataOutput,
    DataSample,
    Reward,
    Step,
)


class DataAnnotator(BaseDataModule):
    """
    Main annotation module providing Label Studio integration for manual data labeling.

    Manages the complete annotation workflow from project creation to annotation export,
    supporting template-based configuration and automated task conversion for reward
    model training and evaluation data preparation.

    Attributes:
        client: Label Studio HTTP client for API interactions
        project_id: Current annotation project identifier
        label_config: Label Studio XML configuration for annotation interface
        template_name: Registered template name for configuration resolution
        project_description: Human-readable project description
        project_title: Display title for the annotation project
        server_url: Label Studio server endpoint URL
        api_token: Authentication token for Label Studio API
        export_processor: Template-specific processor for annotation export

    Input: BaseDataSet or List[DataSample] containing data to annotate
    Output: BaseDataSet with annotation project metadata and original data

    Note:
        Before using this module, ensure Label Studio service is running:
        python examples/data/data_pipeline.py start
    """

    client: Optional[LabelStudioClient] = Field(
        default=None, description="Label Studio client"
    )
    project_id: Optional[int] = Field(default=None, description="Current project ID")
    label_config: Optional[str] = Field(
        default=None, description="Label Studio labeling configuration"
    )
    template_name: Optional[str] = Field(
        default=None, description="Template name to use"
    )
    project_description: str = Field(default="", description="Project description")
    project_title: str = Field(default="", description="Project title")
    server_url: str = Field(
        default="http://localhost:8080", description="Label Studio server URL"
    )
    api_token: Optional[str] = Field(default=None, description="Label Studio API token")
    export_processor: Optional[str] = Field(
        default=None, description="Config-specific processor name (e.g. 'rewardbench')"
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        name: str,
        label_config: Optional[str] = None,
        template_name: Optional[str] = None,
        project_title: str = "RM Gallery Annotation Project",
        project_description: str = "RM Gallery Annotation Project",
        server_url: str = "http://localhost:8080",
        api_token: Optional[str] = None,
        export_processor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize data annotation module with Label Studio configuration.

        Args:
            name: Unique identifier for the annotation module
            label_config: Direct Label Studio XML configuration (overrides template)
            template_name: Registered template name for configuration resolution
            project_title: Display title for annotation project
            project_description: Human-readable project description
            server_url: Label Studio server endpoint URL
            api_token: Authentication token for Label Studio API access
            export_processor: Template-specific processor for annotation export
            metadata: Additional metadata for tracking and debugging
            **kwargs: Additional initialization parameters

        Raises:
            ValueError: If neither label_config nor template_name is provided
        """
        # Resolve label_config from template if needed
        resolved_label_config = self._resolve_label_config(label_config, template_name)
        resolved_export_processor = export_processor or template_name

        super().__init__(
            module_type=DataModuleType.ANNOTATION,
            name=name,
            label_config=resolved_label_config,
            template_name=template_name,
            project_title=project_title,
            project_description=project_description,
            server_url=server_url,
            api_token=api_token,
            export_processor=resolved_export_processor,
            metadata=metadata,
            **kwargs,
        )

        # Initialize client
        if self.api_token:
            self.client = LabelStudioClient(
                server_url=self.server_url, api_token=self.api_token
            )
            logger.info("Label Studio client initialized successfully")
        else:
            logger.warning(
                "No API token provided. Please start Label Studio service first or provide API token."
            )

    def _resolve_label_config(
        self, label_config: Optional[str], template_name: Optional[str]
    ) -> str:
        """
        Resolve Label Studio configuration from direct config or registered template.

        Args:
            label_config: Direct XML configuration string
            template_name: Registered template identifier

        Returns:
            Resolved Label Studio XML configuration

        Raises:
            ValueError: If no configuration source is available
        """
        if label_config:
            return label_config

        if template_name:
            # Try to get label config from registered template
            try:
                template_config = AnnotationTemplateRegistry.get_label_config(
                    template_name
                )
                return template_config
            except Exception as e:
                logger.warning(f"Could not get label configuration from template: {e}")
                return None

        raise ValueError(
            "Either label_config or template_name must be provided. "
            f"Available templates: {AnnotationTemplateRegistry.list_templates()}"
        )

    def create_annotation_project(self) -> bool:
        """
        Create new annotation project in Label Studio with configured settings.

        Returns:
            True if project creation successful, False otherwise
        """
        if not self.client:
            logger.error(
                "Label Studio client not initialized. Please provide API token or start service."
            )
            return False

        self.project_id = self.client.create_project(
            title=self.project_title,
            label_config=self.label_config,
            description=self.project_description,
        )

        return self.project_id is not None

    def run(
        self, input_data: Union[BaseDataSet, List[DataSample]], **kwargs
    ) -> BaseDataSet:
        """
        Execute the annotation workflow with project setup and task import.

        Creates Label Studio project, converts data samples to annotation tasks,
        and imports them for manual annotation. Returns dataset with annotation
        project metadata for tracking and subsequent export operations.

        Args:
            input_data: Dataset or samples to prepare for annotation
            **kwargs: Additional parameters including:
                - create_new_project: Whether to create new project (default True)
                - project_title: Override project title

        Returns:
            BaseDataSet with original data and annotation project metadata
            including project ID, server URL, and task count

        Raises:
            RuntimeError: If Label Studio client unavailable or project creation fails
            Exception: If any annotation workflow step fails
        """
        try:
            # Check if client is available
            if not self.client:
                raise RuntimeError(
                    "Label Studio client not available. "
                    "Please start Label Studio service first: "
                    "python rm_gallery/scripts/start_label_studio.py start"
                )

            # Prepare data
            data_samples = self._prepare_data(input_data)
            logger.info(f"Starting annotation for {len(data_samples)} samples")

            # Check if we should create a new project or use existing one
            create_new_project = kwargs.get("create_new_project", True)
            project_title = kwargs.get("project_title", self.project_title)

            # Create project if needed
            if create_new_project or not self.project_id:
                if not self.create_annotation_project():
                    raise RuntimeError("Failed to create annotation project")

            # Convert data samples to Label Studio tasks
            tasks = self._convert_to_tasks(data_samples)

            # Import tasks to Label Studio
            if not self.client.import_tasks(self.project_id, tasks):
                raise RuntimeError("Failed to import tasks to Label Studio")

            logger.info(f"Successfully imported {len(tasks)} tasks to Label Studio")
            logger.info(
                f"Access annotation interface at: {self.server_url}/projects/{self.project_id}"
            )

            # Return dataset with annotation project information
            if isinstance(input_data, BaseDataSet):
                result_name = f"{input_data.name}_annotation_ready"
                result_metadata = (
                    input_data.metadata.copy() if input_data.metadata else {}
                )
            else:
                result_name = f"{self.name}_annotation_ready"
                result_metadata = {}

            # Add annotation project metadata
            result_metadata.update(
                {
                    "annotation_project_id": self.project_id,
                    "annotation_server_url": self.server_url,
                    "annotation_tasks_count": len(tasks),
                    "annotation_status": "ready_for_annotation",
                    "annotation_project_title": project_title,
                }
            )

            return BaseDataSet(
                name=result_name, metadata=result_metadata, datasamples=data_samples
            )

        except Exception as e:
            logger.error(f"Annotation process failed: {e}")
            raise e

    def export_annotations(self) -> Optional[List[Dict[str, Any]]]:
        """Export completed annotations"""
        if not self.client or not self.project_id:
            logger.error("Client or project not initialized")
            return None

        return self.client.export_annotations(self.project_id)

    def _prepare_data(
        self, input_data: Union[BaseDataSet, List[DataSample]]
    ) -> List[DataSample]:
        """Prepare data for annotation"""
        if isinstance(input_data, BaseDataSet):
            data_samples = list(input_data.datasamples)
        else:
            data_samples = input_data

        # Store original data for later use
        self._original_data_samples = data_samples
        return data_samples

    def _convert_to_tasks(self, data_samples: List[DataSample]) -> List[Dict[str, Any]]:
        """Convert DataSample objects to Label Studio tasks based on schema definition"""
        tasks = []

        for i, sample in enumerate(data_samples):
            task_data = {
                "id": sample.unique_id,
                "unique_id": sample.unique_id,
                "source": sample.source or "unknown",
                "task_category": sample.task_category or "general",
                "created_at": sample.created_at.isoformat()
                if sample.created_at
                else "",
            }

            # Process input messages - List[ChatMessage] -> dialogue format
            input_dialogue = []
            if sample.input:
                for msg in sample.input:
                    input_dialogue.append({"role": msg.role, "content": msg.content})
            task_data["input_messages"] = input_dialogue

            # Process output - List[DataOutput] -> dialogue format with multiple answers
            output_dialogue = []
            answer_count = 0

            if sample.output:
                for output in sample.output:
                    # Process answer (Step)
                    if output.answer:
                        answer_count += 1
                        output_dialogue.append(
                            {
                                "role": output.answer.role,
                                "content": f"Answer {answer_count}: {output.answer.content}",
                            }
                        )

                    # Process steps (Optional[List[Step]])
                    if output.steps:
                        for step in output.steps:
                            output_dialogue.append(
                                {"role": step.role, "content": f"Step: {step.content}"}
                            )

            # If no outputs, create placeholder
            if answer_count == 0:
                answer_count = 1
                output_dialogue.append(
                    {"role": "assistant", "content": "Answer 1: [No output available]"}
                )

            task_data["output_messages"] = output_dialogue
            task_data["answer_count"] = str(answer_count)

            # Add metadata as-is without special processing
            if sample.metadata:
                for key, value in sample.metadata.items():
                    if key not in task_data:  # Don't override existing keys
                        task_data[f"meta_{key}"] = str(value)

            tasks.append({"data": task_data})

        return tasks

    def export_annotations_to_dataset(self) -> BaseDataSet:
        """Export annotations and return as BaseDataSet for pipeline continuation"""
        annotations = self.export_annotations()
        if not annotations:
            logger.warning("No annotations found to export")
            # Return empty dataset
            return BaseDataSet(
                name=f"{self.name}_annotations_empty",
                metadata={"annotation_status": "no_annotations"},
                datasamples=[],
            )

        # Convert annotations to DataSample format
        annotated_samples = self._convert_annotations_to_schema(annotations)

        return BaseDataSet(
            name=f"{self.name}_annotations",
            metadata={
                "annotation_project_id": self.project_id,
                "annotation_count": len(annotated_samples),
                "annotation_status": "exported",
                "data_format": "DataSample",
            },
            datasamples=annotated_samples,
        )

    def _convert_annotations_to_schema(
        self, annotations: List[Dict]
    ) -> List[DataSample]:
        """Convert Label Studio annotations to DataSample format with annotations in label fields"""
        annotated_samples = []

        for annotation in annotations:
            if not annotation.get("annotations"):
                continue

            task_data = annotation.get("data", {})
            ann_results = annotation["annotations"][0].get("result", [])

            # Generic annotation data structure - collect all results by type
            annotation_data = {
                "ratings": {},  # All rating fields
                "choices": {},  # All choice fields
                "text_areas": {},  # All text area fields
                "raw_results": ann_results,  # Keep original results for reference
            }

            # Process annotation results generically
            for result in ann_results:
                field_name = result.get("from_name", "")
                value = result.get("value", {})
                result_type = result.get("type", "")

                # Handle different control types generically
                if "rating" in value:
                    # Rating controls (star ratings, etc.)
                    annotation_data["ratings"][field_name] = {
                        "rating": value.get("rating", 0),
                        "type": result_type,
                    }

                elif "choices" in value or "choice" in value:
                    # Choice controls (radio, checkbox, etc.)
                    choices = value.get("choices", [])
                    if not choices and "choice" in value:
                        choices = [value.get("choice")]
                    annotation_data["choices"][field_name] = {
                        "choices": choices,
                        "type": result_type,
                    }

                elif "text" in value:
                    # Text controls (textarea, text input, etc.)
                    text = value.get("text", [])
                    if not isinstance(text, list):
                        text = [text]
                    annotation_data["text_areas"][field_name] = {
                        "text": text[0] if text else "",
                        "type": result_type,
                    }

            # Log annotation data for debugging
            if (
                annotation_data["ratings"]
                or annotation_data["choices"]
                or annotation_data["text_areas"]
            ):
                logger.debug(
                    f"Found annotation data for {task_data.get('unique_id')}: "
                    f"ratings={list(annotation_data['ratings'].keys())}, "
                    f"choices={list(annotation_data['choices'].keys())}, "
                    f"text_areas={list(annotation_data['text_areas'].keys())}"
                )
            else:
                logger.warning(
                    f"No annotation data found for {task_data.get('unique_id')}"
                )

            # Apply config-specific processing if processor is specified
            if self.export_processor:
                try:
                    # Get template and process annotations
                    template = AnnotationTemplateRegistry.get_template(
                        self.export_processor
                    )
                    if template:
                        processed_data = template.process_annotations(annotation_data)
                        annotation_data["processed"] = processed_data
                        logger.debug(
                            f"Applied {self.export_processor} processor to {task_data.get('unique_id')}"
                        )
                    else:
                        logger.warning(
                            f"No processor found for config: {self.export_processor}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error applying config processor {self.export_processor}: {e}"
                    )

            # Find original sample or reconstruct from task data
            unique_id = task_data.get("unique_id")
            original_sample = None

            if hasattr(self, "_original_data_samples"):
                for sample in self._original_data_samples:
                    if sample.unique_id == unique_id:
                        original_sample = sample
                        break

            if original_sample:
                # Use original sample and add annotations to Step labels
                annotated_outputs = []

                for output in original_sample.output:
                    # Add annotation data to answer step's label
                    if output.answer:
                        new_answer = Step(
                            role=output.answer.role,
                            content=output.answer.content,
                            label={
                                "annotation_data": annotation_data,
                                "annotation_status": "completed",
                            },
                            reward=output.answer.reward
                            if hasattr(output.answer, "reward")
                            else Reward(),
                        )
                    else:
                        new_answer = Step(
                            role="assistant",
                            content="[No answer available]",
                            label={
                                "annotation_data": annotation_data,
                                "annotation_status": "completed",
                            },
                            reward=Reward(),
                        )

                    # Copy steps with original labels
                    new_steps = []
                    if output.steps:
                        for step in output.steps:
                            new_step = Step(
                                role=step.role,
                                content=step.content,
                                label=step.label,
                                reward=step.reward
                                if hasattr(step, "reward")
                                else Reward(),
                            )
                            new_steps.append(new_step)

                    annotated_outputs.append(
                        DataOutput(
                            answer=new_answer, steps=new_steps if new_steps else None
                        )
                    )

                # Create annotated sample
                annotated_sample = DataSample(
                    unique_id=original_sample.unique_id,
                    source=original_sample.source,
                    task_category=original_sample.task_category,
                    input=original_sample.input,
                    output=annotated_outputs,
                    metadata={
                        **(original_sample.metadata or {}),
                        "annotation_status": "completed",
                        "annotation_project_id": self.project_id,
                    },
                    created_at=original_sample.created_at,
                )

            else:
                # Reconstruct DataSample from task data
                annotated_sample = self._reconstruct_sample_from_task_data(
                    task_data, annotation_data
                )

            if annotated_sample:
                annotated_samples.append(annotated_sample)

        return annotated_samples

    def _reconstruct_sample_from_task_data(
        self, task_data: Dict, annotation_data: Dict
    ) -> Optional[DataSample]:
        """Reconstruct DataSample from task data when original sample is not available"""
        try:
            unique_id = task_data.get("unique_id")

            # Reconstruct input from task data
            input_messages = []
            if "input_messages" in task_data:
                input_text = task_data["input_messages"]
                if input_text and input_text != "[No input messages]":
                    # Try to parse as JSON first
                    try:
                        if isinstance(input_text, str):
                            parsed_input = json.loads(input_text)
                            for msg_data in parsed_input:
                                input_messages.append(
                                    ChatMessage(
                                        role=msg_data.get("role", "user"),
                                        content=msg_data.get("content", ""),
                                    )
                                )
                        elif isinstance(input_text, list):
                            for msg_data in input_text:
                                input_messages.append(
                                    ChatMessage(
                                        role=msg_data.get("role", "user"),
                                        content=msg_data.get("content", ""),
                                    )
                                )
                    except:
                        # Fallback to simple text
                        input_messages.append(
                            ChatMessage(role="user", content=str(input_text))
                        )

            # Reconstruct output from task data
            outputs = []
            answer_count = int(task_data.get("answer_count", 0))

            if answer_count > 0:
                # Create outputs for each answer
                for i in range(1, min(4, answer_count + 1)):  # Support up to 3 answers
                    output_messages = task_data.get("output_messages", [])

                    # Find answer for this index
                    answer_content = f"Answer {i}: [Reconstructed from annotation data]"
                    if isinstance(output_messages, str):
                        answer_content = output_messages
                    elif isinstance(output_messages, list):
                        for msg in output_messages:
                            if isinstance(msg, dict) and f"Answer {i}:" in msg.get(
                                "content", ""
                            ):
                                answer_content = msg["content"]
                                break

                    # Create answer step with annotation data in label
                    answer_step = Step(
                        role="assistant",
                        content=answer_content,
                        label={
                            "annotation_data": annotation_data,
                            "annotation_status": "completed",
                            "answer_index": i,
                        },
                        reward=Reward(),
                    )

                    # Create steps if available
                    steps = []
                    if isinstance(output_messages, list):
                        for msg in output_messages:
                            if isinstance(msg, dict) and msg.get(
                                "content", ""
                            ).startswith("Step:"):
                                steps.append(
                                    Step(
                                        role="assistant",
                                        content=msg["content"],
                                        label=None,
                                        reward=Reward(),
                                    )
                                )

                    outputs.append(
                        DataOutput(answer=answer_step, steps=steps if steps else None)
                    )
            else:
                # Create default output with annotation data
                answer_step = Step(
                    role="assistant",
                    content="[No answer available]",
                    label={
                        "annotation_data": annotation_data,
                        "annotation_status": "completed",
                    },
                    reward=Reward(),
                )
                outputs.append(DataOutput(answer=answer_step, steps=None))

            # Create DataSample
            sample = DataSample(
                unique_id=unique_id,
                source=task_data.get("source", "unknown"),
                task_category=task_data.get("task_category", "general"),
                input=input_messages,
                output=outputs,
                metadata={
                    "annotation_status": "completed",
                    "annotation_project_id": self.project_id,
                    "reconstructed_from_annotation": True,
                    "original_answer_count": answer_count,
                },
                created_at=datetime.now(),
            )

            return sample

        except Exception as e:
            logger.error(f"Error reconstructing sample from task data: {e}")
            return None


def create_annotator(
    name: str = "annotation",
    label_config: Optional[str] = None,
    template_name: Optional[str] = None,
    project_title: str = "RM Gallery Annotation Project",
    project_description: str = "RM Gallery Annotation Project",
    server_url: str = "http://localhost:8080",
    api_token: Optional[str] = None,
    export_processor: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DataAnnotator:
    """Factory function to create data annotation module for build pipeline

    Args:
        name: Module name
        label_config: Label Studio configuration XML (optional if template_name provided)
        template_name: Name of registered template to use (optional if label_config provided)
        project_title: Annotation project title
        project_description: Annotation project description
        server_url: Label Studio server URL
        api_token: API token for Label Studio (optional, will try to load from config file)
        export_processor: Name of config-specific processor (e.g., 'reward_bench')
        metadata: Additional metadata for the module

    """

    # Validate API token
    if not api_token:
        raise ValueError(
            "API token is required. Please either:\n"
            "1. Start Label Studio service: python rm_gallery/scripts/start_label_studio.py start\n"
            "2. Provide api_token parameter\n"
            "3. Ensure label_studio_config.json exists with valid token"
        )

    # Create annotation module (validation of label_config/template_name happens in __init__)
    annotation_module = DataAnnotator(
        name=name,
        label_config=label_config,
        template_name=template_name,
        project_title=project_title,
        project_description=project_description,
        server_url=server_url,
        api_token=api_token,
        export_processor=export_processor,
        metadata=metadata,
    )

    return annotation_module
