# -*- coding: utf-8 -*-
"""Validator functions for agent evaluation tasks."""

import json
import re
from typing import Any, Dict


def validate_basic_math_task(output: str) -> Dict[str, Any]:
    """Validate the basic math task output."""
    # Check if the output contains the expected calculation result
    if "248" in output:
        return {
            "success": True,
            "score": 1.0,
            "reason": "Correct mathematical calculation result"
        }
    else:
        return {
            "success": False,
            "score": 0.0,
            "reason": "Incorrect mathematical calculation"
        }


def validate_text_summarization_task(output: str) -> Dict[str, Any]:
    """Validate the text summarization task output."""
    # Check if output has roughly 3 sentences and mentions key concepts
    sentences = re.split(r'[.!?]+', output.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    has_ai_mention = any(word in output.lower() for word in ['artificial intelligence', 'ai', 'machine learning', 'ml'])
    has_pattern_mention = any(word in output.lower() for word in ['patterns', 'data', 'identify'])
    has_efficiency_mention = any(word in output.lower() for word in ['efficiency', 'decision', 'organizations'])

    score = 0.0
    if len(sentences) >= 2 and len(sentences) <= 5:
        score += 0.3  # Sentence count is reasonable
    if has_ai_mention:
        score += 0.3  # Mentions AI concepts
    if has_pattern_mention or has_efficiency_mention:
        score += 0.4  # Mentions relevant concepts

    return {
        "success": score >= 0.7,
        "score": min(score, 1.0),
        "reason": f"Summary quality: {len(sentences)} sentences, relevance score: {score:.2f}"
    }


def validate_simple_reasoning_task(output: str) -> Dict[str, Any]:
    """Validate the simple reasoning task output."""
    # Check if the answer correctly explains why we cannot conclude some roses fade quickly
    output_lower = output.lower()

    correct_negation = "no" in output_lower or "cannot" in output_lower or "not conclude" in output_lower
    logical_explanation = any(phrase in output_lower for phrase in [
        "those flowers that fade", "may not include roses", "specific flowers", "doesn't mean roses"
    ])

    score = 0.0
    if correct_negation:
        score += 0.6  # Correct answer
    if logical_explanation:
        score += 0.4  # Good explanation

    return {
        "success": score >= 0.7,
        "score": min(score, 1.0),
        "reason": f"Logical reasoning quality: {score:.2f}"
    }


def validate_fact_question_task(output: str) -> Dict[str, Any]:
    """Validate the fact question task output."""
    # Check if the output contains Neil Armstrong and the correct date
    has_armstrong = "armstrong" in output.lower() or "neil" in output.lower()
    has_date = "1969" in output and ("july" in output.lower() or "20" in output or "moon" in output.lower())

    score = 0.0
    if has_armstrong:
        score += 0.5
    if has_date:
        score += 0.5

    return {
        "success": score >= 0.8,
        "score": min(score, 1.0),
        "reason": f"Factual accuracy: {score:.2f}"
    }


def validate_creative_writing_task(output: str) -> Dict[str, Any]:
    """Validate the creative writing task output."""
    # Check if the output contains content about seasons
    output_lower = output.lower()

    has_seasons = any(season in output_lower for season in ['spring', 'summer', 'fall', 'winter', 'autumn'])
    has_poetic_elements = len(output.split()) > 10  # At least some content
    has_creativity = len(set(output_lower.split())) / len(output_lower.split()) > 0.6  # Variety of words

    score = 0.0
    if has_seasons:
        score += 0.4
    if has_poetic_elements:
        score += 0.3
    if has_creativity:
        score += 0.3

    return {
        "success": score >= 0.6,
        "score": min(score, 1.0),
        "reason": f"Creative content quality: {score:.2f}"
    }


TASK_VALIDATORS = {
    "basic_math": validate_basic_math_task,
    "text_summarization": validate_text_summarization_task,
    "simple_reasoning": validate_simple_reasoning_task,
    "fact_question": validate_fact_question_task,
    "creative_writing": validate_creative_writing_task,
}


def validate_task_output(task_name: str, output: str) -> Dict[str, Any]:
    """Validate the output of a specific task."""
    if task_name in TASK_VALIDATORS:
        return TASK_VALIDATORS[task_name](output)
    else:
        return {
            "success": True,
            "score": 0.5,  # Neutral score for unvalidated tasks
            "reason": "Task not validated, neutral score assigned"
        }