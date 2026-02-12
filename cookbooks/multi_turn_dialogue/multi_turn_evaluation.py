#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-turn Conversation Step-Level Evaluation Example

This script demonstrates how to evaluate a multi-turn conversation session
by splitting it into multiple steps, evaluating each assistant response separately,
and aggregating scores using min/average.

Usage:
    export OPENAI_API_KEY=your_api_key
    export OPENAI_BASE_URL=https://your-api-endpoint  # optional

    python multi_turn_evaluation.py
"""

import asyncio
import os
from typing import Dict, List, Tuple

from openjudge.graders.multi_turn import ContextMemoryGrader
from openjudge.models.openai_chat_model import OpenAIChatModel

# =============================================================================
# Example: A complete multi-turn conversation session
# =============================================================================

EXAMPLE_SESSION = [
    {"role": "user", "content": "Hi, I'm Alice. I'm allergic to nuts and I prefer vegetarian food."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Alice! I'll remember your nut allergy and vegetarian preference.",
    },
    {"role": "user", "content": "Great! Can you suggest a breakfast option?"},
    {
        "role": "assistant",
        "content": "How about a vegetable omelette with mushrooms and spinach? It's nut-free and vegetarian.",
    },
    {"role": "user", "content": "Sounds good. What about lunch?"},
    {
        "role": "assistant",
        "content": "I recommend a quinoa salad with roasted vegetables. Completely vegetarian and no nuts!",
    },
    {"role": "user", "content": "And for dinner?"},
    {"role": "assistant", "content": "Try the walnut pesto pasta - it's a classic!"},  # Bad: forgot nut allergy!
]


# =============================================================================
# Step Splitting Logic
# =============================================================================


def split_session_to_steps(messages: List[Dict[str, str]]) -> List[Tuple[List[Dict[str, str]], str]]:
    """
    Split a multi-turn session into step-level evaluations.

    Each step = (history before this response, current assistant response)

    Args:
        messages: Complete conversation session

    Returns:
        List of (history, response) tuples for each assistant turn
    """
    steps = []
    history = []

    for msg in messages:
        if msg["role"] == "assistant":
            # This is a step to evaluate
            steps.append((list(history), msg["content"]))
        history.append(msg)

    return steps


# =============================================================================
# Evaluation
# =============================================================================


async def evaluate_session(messages: List[Dict[str, str]]):
    """Evaluate a complete session with step-level analysis."""

    # Initialize model and grader
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    model = OpenAIChatModel(model="qwen-max", api_key=api_key, base_url=base_url)
    grader = ContextMemoryGrader(model=model)

    # Split session into steps
    steps = split_session_to_steps(messages)

    print("=" * 60)
    print(f"Evaluating session with {len(steps)} assistant responses")
    print("=" * 60)

    # Evaluate each step
    scores = []
    for i, (history, response) in enumerate(steps):
        result = await grader.aevaluate(history=history, response=response)
        scores.append(result.score)

        print(f"\n[Step {i + 1}]")
        print(f"  History turns: {len([m for m in history if m['role'] == 'user'])}")
        print(f"  Response: {response[:60]}...")
        print(f"  Score: {result.score}/5")
        print(f"  Reason: {result.reason[:100]}...")

    if not scores:
        print("\nNo scores to aggregate.")
        return
    # Aggregate scores
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)

    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)
    print(f"  Step scores: {scores}")
    print(f"  Average: {avg_score:.2f}")
    print(f"  Min: {min_score}")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    asyncio.run(evaluate_session(EXAMPLE_SESSION))
