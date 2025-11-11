"""
Multimodal Metrics Module

This module provides base classes, utilities, and graders for multimodal evaluation metrics.
"""

from rm_gallery.core.metrics.multimodal.base import BaseMultimodalMetric
from rm_gallery.core.metrics.multimodal.schema import (
    MLLMImage,
    MLLMTestCase,
    MLLMTestCaseParams,
    MultimodalMetricResult,
    ReasonScore,
)
from rm_gallery.core.metrics.multimodal.utils import (
    construct_verbose_logs,
    separate_images_from_text,
)


# Lazy import graders to avoid circular import
def _register_graders():
    """Register multimodal graders (called during metrics module initialization)"""
    from rm_gallery.core.metrics.multimodal.image_coherence import ImageCoherenceGrader
    from rm_gallery.core.metrics.multimodal.image_editing import ImageEditingGrader
    from rm_gallery.core.metrics.multimodal.image_helpfulness import (
        ImageHelpfulnessGrader,
    )
    from rm_gallery.core.metrics.multimodal.image_reference import ImageReferenceGrader
    from rm_gallery.core.metrics.multimodal.multimodal_geval import (
        MultimodalGEvalGrader,
    )
    from rm_gallery.core.metrics.multimodal.text_to_image import TextToImageGrader
    from rm_gallery.core.metrics.registry import register_grader

    # Register graders
    register_grader("image_coherence")(ImageCoherenceGrader)
    register_grader("image_helpfulness")(ImageHelpfulnessGrader)
    register_grader("image_reference")(ImageReferenceGrader)
    register_grader("image_editing")(ImageEditingGrader)
    register_grader("text_to_image")(TextToImageGrader)
    register_grader("multimodal_geval")(MultimodalGEvalGrader)

    return (
        ImageCoherenceGrader,
        ImageHelpfulnessGrader,
        ImageReferenceGrader,
        ImageEditingGrader,
        TextToImageGrader,
        MultimodalGEvalGrader,
    )


# Export grader classes for direct import
def __getattr__(name):
    """Lazy loading for grader classes"""
    grader_names = (
        "ImageCoherenceGrader",
        "ImageHelpfulnessGrader",
        "ImageReferenceGrader",
        "ImageEditingGrader",
        "TextToImageGrader",
        "MultimodalGEvalGrader",
    )
    if name in grader_names:
        _graders = _register_graders()
        _map = {
            "ImageCoherenceGrader": _graders[0],
            "ImageHelpfulnessGrader": _graders[1],
            "ImageReferenceGrader": _graders[2],
            "ImageEditingGrader": _graders[3],
            "TextToImageGrader": _graders[4],
            "MultimodalGEvalGrader": _graders[5],
        }
        # Cache in module
        globals()[name] = _map[name]
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseMultimodalMetric",
    "MLLMImage",
    "MLLMTestCase",
    "MLLMTestCaseParams",
    "MultimodalMetricResult",
    "ReasonScore",
    "construct_verbose_logs",
    "separate_images_from_text",
    # Graders
    "ImageCoherenceGrader",
    "ImageHelpfulnessGrader",
    "ImageReferenceGrader",
    "ImageEditingGrader",
    "TextToImageGrader",
    "MultimodalGEvalGrader",
]
