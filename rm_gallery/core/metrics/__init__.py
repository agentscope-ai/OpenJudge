"""
RM-Gallery Metrics Module

Evaluation metrics module restructured to work with the Grader framework.

This module provides a comprehensive collection of evaluation metrics as Graders:
- String check graders (Exact Match, Substring, Overlap, etc.)
- NLP graders (BLEU, ROUGE, METEOR, GLEU, etc.)
- Text similarity graders (Fuzzy Match, Cosine Similarity, F1 Score, etc.)
- Format check graders (JSON validation, etc.)
- Multimodal graders (Image Coherence, Text-to-Image, Multimodal G-Eval, etc.)

Quick Start:
    >>> from rm_gallery.core.metrics import get_grader, list_available_graders
    >>> from rm_gallery.core.data import DataSample
    >>>
    >>> # List all available graders
    >>> graders = list_available_graders()
    >>> print(f"Available: {', '.join(graders)}")
    >>>
    >>> # Use a text grader
    >>> grader = get_grader("exact_match")
    >>> data_sample = DataSample(
    ...     data={"reference": "hello"},
    ...     samples=[{"candidate": "hello"}, {"candidate": "world"}]
    ... )
    >>> import asyncio
    >>> results = asyncio.run(grader(data_sample))
    >>>
    >>> # Use a multimodal grader
    >>> from rm_gallery.core.metrics.multimodal import ImageCoherenceGrader
    >>> from rm_gallery.core.model.qwen_vlm_api import QwenVLAPI
    >>> api = QwenVLAPI(api_key="...")
    >>> grader = ImageCoherenceGrader(model=api)
"""

# Format Check Graders
from rm_gallery.core.metrics.format_check.json_match import (
    JsonMatchGrader,
    JsonValidatorGrader,
)

# NLP Graders
from rm_gallery.core.metrics.nlp_metrics.bleu import BLEUGrader, SentenceBLEUGrader
from rm_gallery.core.metrics.nlp_metrics.gleu import ChrFGrader, GLEUGrader
from rm_gallery.core.metrics.nlp_metrics.meteor import METEORGrader
from rm_gallery.core.metrics.nlp_metrics.rouge import (
    ROUGE1Grader,
    ROUGE2Grader,
    ROUGEGrader,
    ROUGELGrader,
)
from rm_gallery.core.metrics.nlp_metrics.rouge_ngram import (
    ROUGE3Grader,
    ROUGE4Grader,
    ROUGE5Grader,
    ROUGENGramGrader,
)

# Registry functions
from rm_gallery.core.metrics.registry import (
    get_grader,
    grader_registry,
    list_available_graders,
    register_grader,
)

# Schema (for data structures)
from rm_gallery.core.metrics.schema import (
    AggregatedMetricResult,
    BatchComparisonInput,
    ComparisonInput,
    MetricConfig,
    MetricResult,
)

# String Check Graders
from rm_gallery.core.metrics.string_check.exact_match import (
    ExactMatchGrader,
    PrefixMatchGrader,
    RegexMatchGrader,
    SuffixMatchGrader,
)
from rm_gallery.core.metrics.string_check.substring import (
    CharacterOverlapGrader,
    ContainsAllGrader,
    ContainsAnyGrader,
    SubstringMatchGrader,
    WordOverlapGrader,
)

# Text Similarity Graders
from rm_gallery.core.metrics.text_similarity.cosine import (
    CosineSimilarityGrader,
    JaccardSimilarityGrader,
)
from rm_gallery.core.metrics.text_similarity.f1_score import (
    F1ScoreGrader,
    TokenF1Grader,
)
from rm_gallery.core.metrics.text_similarity.fuzzy import (
    EditDistanceGrader,
    FuzzyMatchGrader,
)

# Multimodal Graders (lazy import to avoid circular dependency)
# from rm_gallery.core.metrics.multimodal import (
#     ImageCoherenceGrader,
#     MultimodalGEvalGrader,
#     TextToImageGrader,
# )


# Auto-register all graders
register_grader("exact_match")(ExactMatchGrader)
register_grader("prefix_match")(PrefixMatchGrader)
register_grader("suffix_match")(SuffixMatchGrader)
register_grader("regex_match")(RegexMatchGrader)
register_grader("substring_match")(SubstringMatchGrader)
register_grader("contains_all")(ContainsAllGrader)
register_grader("contains_any")(ContainsAnyGrader)
register_grader("word_overlap")(WordOverlapGrader)
register_grader("char_overlap")(CharacterOverlapGrader)

register_grader("bleu")(BLEUGrader)
register_grader("sentence_bleu")(SentenceBLEUGrader)
register_grader("gleu")(GLEUGrader)
register_grader("chrf")(ChrFGrader)
register_grader("meteor")(METEORGrader)
register_grader("rouge")(ROUGEGrader)
register_grader("rouge1")(ROUGE1Grader)
register_grader("rouge2")(ROUGE2Grader)
register_grader("rougeL")(ROUGELGrader)
register_grader("rouge3")(ROUGE3Grader)
register_grader("rouge4")(ROUGE4Grader)
register_grader("rouge5")(ROUGE5Grader)
register_grader("rouge_ngram")(ROUGENGramGrader)

register_grader("f1_score")(F1ScoreGrader)
register_grader("token_f1")(TokenF1Grader)
register_grader("fuzzy_match")(FuzzyMatchGrader)
register_grader("edit_distance")(EditDistanceGrader)
register_grader("cosine")(CosineSimilarityGrader)
register_grader("jaccard")(JaccardSimilarityGrader)

register_grader("json_match")(JsonMatchGrader)
register_grader("json_validator")(JsonValidatorGrader)


# Register multimodal graders
def _register_multimodal_graders():
    """Register multimodal graders (lazy loading to avoid circular import)"""
    try:
        from rm_gallery.core.metrics.multimodal import _register_graders

        _register_graders()
    except ImportError:
        pass  # Multimodal module not available


# Call registration
_register_multimodal_graders()


# Expose multimodal graders for convenience
def __getattr__(name):
    """Lazy loading for multimodal grader classes"""
    multimodal_graders = (
        "ImageCoherenceGrader",
        "ImageHelpfulnessGrader",
        "ImageReferenceGrader",
        "ImageEditingGrader",
        "TextToImageGrader",
        "MultimodalGEvalGrader",
    )
    if name in multimodal_graders:
        try:
            from rm_gallery.core.metrics.multimodal import (
                ImageCoherenceGrader,
                ImageEditingGrader,
                ImageHelpfulnessGrader,
                ImageReferenceGrader,
                MultimodalGEvalGrader,
                TextToImageGrader,
            )

            _map = {
                "ImageCoherenceGrader": ImageCoherenceGrader,
                "ImageHelpfulnessGrader": ImageHelpfulnessGrader,
                "ImageReferenceGrader": ImageReferenceGrader,
                "ImageEditingGrader": ImageEditingGrader,
                "TextToImageGrader": TextToImageGrader,
                "MultimodalGEvalGrader": MultimodalGEvalGrader,
            }
            return _map[name]
        except ImportError:
            raise AttributeError(f"Multimodal graders not available: {name}") from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.0"

__all__ = [
    # Registry
    "grader_registry",
    "register_grader",
    "get_grader",
    "list_available_graders",
    # Schema
    "MetricResult",
    "ComparisonInput",
    "BatchComparisonInput",
    "MetricConfig",
    "AggregatedMetricResult",
    # String Check Graders
    "ExactMatchGrader",
    "PrefixMatchGrader",
    "SuffixMatchGrader",
    "RegexMatchGrader",
    "SubstringMatchGrader",
    "ContainsAllGrader",
    "ContainsAnyGrader",
    "WordOverlapGrader",
    "CharacterOverlapGrader",
    # NLP Graders
    "BLEUGrader",
    "SentenceBLEUGrader",
    "GLEUGrader",
    "ChrFGrader",
    "METEORGrader",
    "ROUGEGrader",
    "ROUGE1Grader",
    "ROUGE2Grader",
    "ROUGELGrader",
    "ROUGE3Grader",
    "ROUGE4Grader",
    "ROUGE5Grader",
    "ROUGENGramGrader",
    # Text Similarity Graders
    "F1ScoreGrader",
    "TokenF1Grader",
    "FuzzyMatchGrader",
    "EditDistanceGrader",
    "CosineSimilarityGrader",
    "JaccardSimilarityGrader",
    # Format Check Graders
    "JsonMatchGrader",
    "JsonValidatorGrader",
    # Multimodal Graders
    "ImageCoherenceGrader",
    "ImageHelpfulnessGrader",
    "ImageReferenceGrader",
    "ImageEditingGrader",
    "TextToImageGrader",
    "MultimodalGEvalGrader",
]
