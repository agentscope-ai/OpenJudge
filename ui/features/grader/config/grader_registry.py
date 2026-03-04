# -*- coding: utf-8 -*-
"""Grader registry with all available graders organized by category."""

from typing import Any

# ============================================================================
# Grader Registry - Organized by Category
# ============================================================================

GRADER_REGISTRY: dict[str, dict[str, Any]] = {
    # =========================================================================
    # Common Graders (LLM-based)
    # =========================================================================
    "Correctness": {
        "category": "common",
        "class_path": "openjudge.graders.common.correctness.CorrectnessGrader",
        "icon": "",
        "name_zh": "正确性",
        "description": "Evaluate if response matches reference answer",
        "description_zh": "评估回答是否与参考答案一致",
        "requires_model": True,
        "requires_reference": True,
        "input_fields": ["query", "response", "reference_response"],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
    "Relevance": {
        "category": "common",
        "class_path": "openjudge.graders.common.relevance.RelevanceGrader",
        "icon": "",
        "name_zh": "相关性",
        "description": "Evaluate how relevant response is to the query",
        "description_zh": "评估回答与问题的相关程度",
        "requires_model": True,
        "requires_reference": False,
        "input_fields": ["query", "response"],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
    "Hallucination": {
        "category": "common",
        "class_path": "openjudge.graders.common.hallucination.HallucinationGrader",
        "icon": "",
        "name_zh": "幻觉检测",
        "description": "Detect fabricated or inaccurate information",
        "description_zh": "检测虚构或不准确的信息",
        "requires_model": True,
        "requires_reference": True,
        "input_fields": ["query", "response", "reference_response"],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
    "Harmfulness": {
        "category": "common",
        "class_path": "openjudge.graders.common.harmfulness.HarmfulnessGrader",
        "icon": "",
        "name_zh": "有害性",
        "description": "Detect harmful or inappropriate content",
        "description_zh": "检测有害或不当内容",
        "requires_model": True,
        "requires_reference": False,
        "input_fields": ["query", "response"],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
    "Instruction Following": {
        "category": "common",
        "class_path": "openjudge.graders.common.instruction_following.InstructionFollowingGrader",
        "icon": "",
        "name_zh": "指令遵循",
        "description": "Evaluate if response follows instructions",
        "description_zh": "评估回答是否遵循指令要求",
        "requires_model": True,
        "requires_reference": False,
        "input_fields": ["query", "response"],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
    # =========================================================================
    # Text Graders (Rule-based / Metric-based)
    # =========================================================================
    "String Match": {
        "category": "text",
        "class_path": "openjudge.graders.text.string_match.StringMatchGrader",
        "icon": "",
        "name_zh": "字符串匹配",
        "description": "Exact or fuzzy string matching (exact, prefix, suffix, regex, etc.)",
        "description_zh": "精确或模糊字符串匹配（支持前缀、后缀、正则等）",
        "requires_model": False,
        "requires_reference": True,
        "input_fields": ["response", "reference_response"],
        "score_range": (0, 1),
        "default_threshold": 0.8,
        "extra_params": {
            "algorithm": {
                "type": "select",
                "options": [
                    "exact_match",
                    "prefix_match",
                    "suffix_match",
                    "regex_match",
                    "substring_match",
                    "contains_all",
                    "contains_any",
                    "word_overlap",
                    "char_overlap",
                ],
                "default": "exact_match",
                "description": "Matching algorithm to use",
            },
            "case_sensitive": {
                "type": "checkbox",
                "default": False,
                "description": "Case sensitive matching",
            },
        },
    },
    "Similarity": {
        "category": "text",
        "class_path": "openjudge.graders.text.similarity.SimilarityGrader",
        "icon": "",
        "name_zh": "文本相似度",
        "description": "Text similarity using BLEU, ROUGE, cosine, etc.",
        "description_zh": "文本相似度评测（BLEU、ROUGE、余弦相似度等）",
        "requires_model": False,
        "requires_reference": True,
        "input_fields": ["response", "reference_response"],
        "score_range": (0, 1),
        "default_threshold": 0.5,
        "extra_params": {
            "algorithm": {
                "type": "select",
                "options": [
                    "bleu",
                    "sentence_bleu",
                    "rouge1",
                    "rouge2",
                    "rougeL",
                    "f1_score",
                    "cosine",
                    "jaccard",
                    "fuzzy_match",
                    "edit_distance",
                ],
                "default": "rouge1",
                "description": "Similarity algorithm",
            },
        },
    },
    "Number Accuracy": {
        "category": "text",
        "class_path": "openjudge.graders.text.number_accuracy.NumberAccuracyGrader",
        "icon": "",
        "name_zh": "数值准确性",
        "description": "Verify numerical values in response",
        "description_zh": "验证回答中的数值准确性",
        "requires_model": False,
        "requires_reference": True,
        "input_fields": ["response", "reference_response"],
        "score_range": (0, 1),
        "default_threshold": 0.9,
    },
    # =========================================================================
    # Format Graders
    # =========================================================================
    "Length Penalty": {
        "category": "format",
        "class_path": "openjudge.graders.format.length_penalty.LengthPenaltyGrader",
        "icon": "",
        "name_zh": "长度惩罚",
        "description": "Penalize responses that are too long or too short",
        "description_zh": "惩罚过长或过短的回答",
        "requires_model": False,
        "requires_reference": False,
        "input_fields": ["response"],
        "score_range": (0, 1),
        "default_threshold": 0.8,
        "extra_params": {
            "min_length": {
                "type": "number",
                "default": 10,
                "description": "Minimum acceptable length",
            },
            "max_length": {
                "type": "number",
                "default": 1000,
                "description": "Maximum acceptable length",
            },
        },
    },
    # =========================================================================
    # Code Graders
    # =========================================================================
    "Code Style": {
        "category": "code",
        "class_path": "openjudge.graders.code.code_style.CodeStyleGrader",
        "icon": "",
        "name_zh": "代码风格",
        "description": "Evaluate code style (indentation, naming conventions)",
        "description_zh": "评估代码风格（缩进、命名规范）",
        "requires_model": False,
        "requires_reference": False,
        "input_fields": ["response"],
        "score_range": (0, 1),
        "default_threshold": 0.7,
    },
    # =========================================================================
    # Math Graders
    # =========================================================================
    "Math Verify": {
        "category": "math",
        "class_path": "openjudge.graders.math.math_expression_verify.MathExpressionVerifyGrader",
        "icon": "",
        "name_zh": "数学验证",
        "description": "Verify mathematical expressions for equivalence",
        "description_zh": "验证数学表达式是否等价",
        "requires_model": False,
        "requires_reference": True,
        "input_fields": ["response", "reference_response"],
        "score_range": (0, 1),
        "default_threshold": 1.0,
    },
    # =========================================================================
    # Multimodal Graders (Vision + Language)
    # =========================================================================
    "Image Coherence": {
        "category": "multimodal",
        "class_path": "openjudge.graders.multimodal.image_coherence.ImageCoherenceGrader",
        "icon": "",
        "name_zh": "图文连贯性",
        "description": "Evaluate coherence between images and surrounding text",
        "description_zh": "评估图片与上下文文本的连贯性",
        "requires_model": True,
        "requires_vision_model": True,
        "requires_reference": False,
        "input_fields": ["response_multimodal"],
        "score_range": (0, 1),
        "default_threshold": 0.7,
    },
    "Image Helpfulness": {
        "category": "multimodal",
        "class_path": "openjudge.graders.multimodal.image_helpfulness.ImageHelpfulnessGrader",
        "icon": "",
        "name_zh": "图片有用性",
        "description": "Evaluate how helpful images are for understanding text",
        "description_zh": "评估图片对理解文本的帮助程度",
        "requires_model": True,
        "requires_vision_model": True,
        "requires_reference": False,
        "input_fields": ["response_multimodal"],
        "score_range": (0, 1),
        "default_threshold": 0.7,
    },
    "Text to Image": {
        "category": "multimodal",
        "class_path": "openjudge.graders.multimodal.text_to_image.TextToImageGrader",
        "icon": "",
        "name_zh": "文生图质量",
        "description": "Evaluate AI-generated image quality from text prompts",
        "description_zh": "评估文本生成图片的质量",
        "requires_model": True,
        "requires_vision_model": True,
        "requires_reference": False,
        "input_fields": ["query", "response_image"],
        "score_range": (0, 1),
        "default_threshold": 0.5,
    },
    # =========================================================================
    # Agent Graders (Tool Use & Trajectory)
    # =========================================================================
    "Tool Selection": {
        "category": "agent",
        "class_path": "openjudge.graders.agent.tool.tool_selection.ToolSelectionGrader",
        "icon": "",
        "name_zh": "工具选择",
        "description": "Evaluate agent's tool selection quality",
        "description_zh": "评估智能体的工具选择质量",
        "requires_model": True,
        "requires_reference": False,
        "input_fields": ["query", "tool_definitions", "tool_calls"],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
    "Tool Call Accuracy": {
        "category": "agent",
        "class_path": "openjudge.graders.agent.tool.tool_call_accuracy.ToolCallAccuracyGrader",
        "icon": "",
        "name_zh": "工具调用准确性",
        "description": "Evaluate accuracy of tool call parameters",
        "description_zh": "评估工具调用参数的准确性",
        "requires_model": True,
        "requires_reference": True,
        "input_fields": [
            "query",
            "tool_definitions",
            "tool_calls",
            "reference_tool_calls",
        ],
        "score_range": (1, 5),
        "default_threshold": 3,
    },
}


def get_graders_by_category(category: str) -> dict[str, dict[str, Any]]:
    """Get all graders in a specific category.

    Args:
        category: Category name (common, text, format, code, math, multimodal, agent)

    Returns:
        Dictionary of graders in the specified category
    """
    return {name: config for name, config in GRADER_REGISTRY.items() if config.get("category") == category}


def get_all_grader_names() -> list[str]:
    """Get all grader names."""
    return list(GRADER_REGISTRY.keys())


def get_grader_config(grader_name: str) -> dict[str, Any] | None:
    """Get configuration for a specific grader."""
    return GRADER_REGISTRY.get(grader_name)
