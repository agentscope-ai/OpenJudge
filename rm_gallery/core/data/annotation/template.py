"""
Annotation Template System - extensible framework for Label Studio annotation configurations.

Provides base classes and registry for managing annotation templates with custom labeling
configurations and post-processing logic for different evaluation tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type


class BaseAnnotationTemplate(ABC):
    """
    Abstract base class for annotation templates defining labeling interface and processing logic.

    Templates encapsulate Label Studio XML configuration and annotation post-processing
    for specific evaluation tasks like reward benchmarking or quality assessment.

    Attributes:
        name: Unique identifier for the template
    """

    def __init__(self, name: str):
        """
        Initialize annotation template with unique identifier.

        Args:
            name: Unique template name for registry identification
        """
        self.name = name

    @property
    @abstractmethod
    def label_config(self) -> str:
        """
        Return the Label Studio XML configuration for the annotation interface.

        Defines the labeling interface including rating scales, choice options,
        text areas, and other annotation components specific to the evaluation task.

        Returns:
            Label Studio XML configuration string
        """
        pass

    @abstractmethod
    def process_annotations(self, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw annotation data and return structured evaluation results.

        Transforms Label Studio annotation output into domain-specific structured
        data suitable for reward model training or evaluation analysis.

        Args:
            annotation_data: Raw annotation data from Label Studio containing
                           ratings, choices, text areas, and other annotation components

        Returns:
            Processed annotation data structured for specific evaluation needs
            (e.g., reward scores, preference rankings, quality metrics)
        """
        pass

    def validate_annotation_data(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Validate annotation data structure and completeness (optional override).

        Performs sanity checks on annotation data to ensure required fields
        are present and values are within expected ranges.

        Args:
            annotation_data: Annotation data to validate

        Returns:
            True if annotation data is valid and complete, False otherwise
        """
        return True


class AnnotationTemplateRegistry:
    """
    Registry system for managing and discovering annotation templates.

    Provides decorator-based registration and factory methods for template instantiation.
    Enables extensible template ecosystem for different annotation tasks.
    """

    _templates: Dict[str, BaseAnnotationTemplate] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Type[BaseAnnotationTemplate]], Type[BaseAnnotationTemplate]]:
        """
        Decorator for registering annotation templates with unique identifiers.

        Args:
            name: Unique template identifier for registry lookup

        Returns:
            Decorator function that registers the template class and returns it unchanged

        Example:
            @AnnotationTemplateRegistry.register("rewardbench")
            class RewardBenchTemplate(BaseAnnotationTemplate):
                ...
        """

        def decorator(
            template_class: Type[BaseAnnotationTemplate],
        ) -> Type[BaseAnnotationTemplate]:
            # Create an instance of the template with the given name
            template_instance = template_class(name)
            cls._templates[name] = template_instance
            return template_class

        return decorator

    @classmethod
    def get_template(cls, name: str) -> Optional[BaseAnnotationTemplate]:
        """
        Retrieve template instance by name.

        Args:
            name: Template identifier to look up

        Returns:
            Template instance if found, None otherwise
        """
        return cls._templates.get(name)

    @classmethod
    def get_label_config(cls, template_name: str) -> Optional[str]:
        """
        Get Label Studio XML configuration from registered template.

        Args:
            template_name: Name of registered template

        Returns:
            Label Studio XML configuration string if template exists, None otherwise
        """
        template = cls.get_template(template_name)
        return template.label_config if template else None

    @classmethod
    def list_templates(cls) -> list[str]:
        """
        List all registered template names.

        Returns:
            List of registered template identifiers
        """
        return list(cls._templates.keys())
