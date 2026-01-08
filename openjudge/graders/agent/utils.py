# -*- coding: utf-8 -*-
"""
Agent Action Utilities
This module provides utility functions for analyzing agent action behaviors,
including action-observation pair extraction and similarity calculations.
"""
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_action_observation_pairs(
    messages: List[Dict[str, Any]],
) -> List[Tuple[Dict, str]]:
    """
    Extract action-observation pairs from a sequence of messages.
    This utility function parses the message history to identify agent actions
    and their corresponding observations/results. It supports precise matching
    using tool_call_id and falls back to FIFO matching for compatibility.
    Args:
        messages: List of message dicts containing agent interactions
    Returns:
        List of tuples where each tuple contains (action, observation)
        - action: Dict representing the tool call made by the agent
        - observation: String containing the result/response from the tool
    Raises:
        ValueError: If tool_call_id is provided but not found in pending actions,
                   or if there's a mismatch between tool calls and responses
    Example:
        >>> messages = [
        ...     {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {...}}]},
        ...     {"role": "tool", "tool_call_id": "call_1", "content": "Result"},
        ... ]
        >>> pairs = extract_action_observation_pairs(messages)
        >>> print(len(pairs))  # 1
    """
    pairs = []
    # Use dict to store pending tool calls by their ID for precise matching
    pending_actions = {}
    # Fallback list for tool calls without IDs (backward compatibility)
    pending_actions_no_id = []
    for message in messages:
        role = message.get("role", "")
        tool_calls = message.get("tool_calls", [])
        content = message.get("content", "")
        if role == "assistant" and tool_calls:
            # Store tool calls indexed by their ID
            for tool_call in tool_calls:
                tool_call = tool_call.get("tool_call", tool_call)
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    pending_actions[tool_call_id] = tool_call
                else:
                    # Fallback for tool calls without ID
                    pending_actions_no_id.append(tool_call)
        elif role in ("function", "tool"):
            # Match function/tool response with corresponding tool call using tool_call_id
            tool_call_id = message.get("tool_call_id")

            if tool_call_id and tool_call_id in pending_actions:
                # Precise matching using tool_call_id
                action = pending_actions.pop(tool_call_id)
                observation = content or ""
                pairs.append((action, observation))
            elif tool_call_id and tool_call_id not in pending_actions:
                # tool_call_id provided but not found in pending actions
                raise ValueError(
                    f"Tool call ID '{tool_call_id}' not found in pending actions. "
                    f"Available IDs: {list(pending_actions.keys())}",
                )
            elif not tool_call_id and pending_actions_no_id:
                # Fallback to FIFO matching for tool calls without IDs
                action = pending_actions_no_id.pop(0)
                observation = content or ""
                pairs.append((action, observation))
            elif not tool_call_id and not pending_actions_no_id:
                # No tool_call_id and no pending actions without IDs
                raise ValueError(
                    "Tool/function response without tool_call_id, but no pending actions without IDs. "
                    "This may indicate a mismatch between tool calls and responses.",
                )
    return pairs


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using word overlap.
    This function uses a simple Jaccard similarity metric based on word sets
    to measure textual similarity. It's useful for comparing observations,
    actions, or other text content for redundancy or loop detection.
    Args:
        text1: First text string
        text2: Second text string
    Returns:
        Float similarity score between 0 and 1, where:
        - 0 means no word overlap
        - 1 means identical word sets
    Example:
        >>> sim = calculate_text_similarity("hello world", "world hello")
        >>> print(sim)  # 1.0
        >>> sim = calculate_text_similarity("hello", "world")
        >>> print(sim)  # 0.0
    """
    if not text1 or not text2:
        return 0.0
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using TF-IDF and cosine similarity.
    This function uses TF-IDF vectorization and cosine similarity for
    more sophisticated semantic matching compared to word overlap.
    It captures semantic relationships better than simple lexical matching.
    Args:
        text1: First text string
        text2: Second text string
    Returns:
        Float similarity score between 0 and 1, where:
        - 0 means completely dissimilar
        - 1 means semantically identical
    Example:
        >>> sim = calculate_semantic_similarity(
        ...     "The cat sat on the mat",
        ...     "A feline rested on the rug"
        ... )
        >>> print(sim > 0)  # True (some semantic similarity)
    """
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except Exception as e:
        logger.warning(f"Semantic similarity calculation failed: {e}")
        return 0.0


def format_history(history: Optional[list] = None) -> str:
    """Format history steps for evaluation.

    Args:
        history: Optional list of previous step dictionaries

    Returns:
        Formatted history string, or empty string if no history
    """
    if not history:
        return ""

    lines = ["<History Steps>"]
    for i, hist_step in enumerate(history):
        lines.append(f"Step {i + 1}:")
        for key, value in hist_step.items():
            if value:
                lines.append(f"{key.capitalize()}: {value}")
        lines.append("")
    lines.append("</History Steps>")

    return "\n".join(lines)


__all__ = [
    "extract_action_observation_pairs",
    "calculate_text_similarity",
    "calculate_semantic_similarity",
    "format_history",
]
