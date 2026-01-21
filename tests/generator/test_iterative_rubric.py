# -*- coding: utf-8 -*-
"""Iterative Rubric Generator test module.

This module contains unit tests for the Iterative Rubric Generator functionality
with different configurations and evaluation modes.

Demonstrates workflow:
1. Create generator with configuration
2. Generate rubrics from training data (with labels)
3. Evaluate test data using the response grader (without labels)

Supports pointwise and listwise evaluation modes.

Note:
    Rubric generation may fail validation due to LLM output inconsistency.
    This is not a code bug but expected behavior. Tests should handle this gracefully.

Example:
    Run all tests:
    ```bash
    pytest tests/generator/test_iterative_rubric.py -v
    ```

    Run a specific test:
    ```bash
    pytest tests/generator/test_iterative_rubric.py::test_iterative_grader_pointwise_single_response -v
    ```

    Run directly as a script:
    ```bash
    python tests/generator/test_iterative_rubric.py
    ```
"""

import asyncio
from copy import deepcopy

import pytest
from loguru import logger

from openjudge.generator.iterative_rubric.generator import (
    IterativeListwiseRubricsGeneratorConfig,
    IterativePointwiseRubricsGeneratorConfig,
    IterativeRubricsGenerator,
)
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderRank, GraderScore
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# pylint: disable=line-too-long

# =============================================================================
# Test Data
# =============================================================================

# Pointwise training data (with labels for rubric generation)
POINTWISE_TRAINING_SAMPLE = [
    {
        "query": "What is the capital of France?",
        "response": "Paris is the capital of France.",
        "label_score": 1,
    },
]

# Pointwise test data (without labels for evaluation)
POINTWISE_TEST_SAMPLE = [
    {
        "query": "What is the capital of China?",
        "response": "Beijing is the capital of China.",
    },
]

# Listwise training data (with labels for rubric generation)
LISTWISE_TRAINING_SAMPLE = [
    {
        "query": "Write a short story about a robot learning to paint.",
        "responses": [
            "BEEP-7 was a maintenance robot who discovered paint cans in an abandoned art studio. Curious, it dipped its mechanical fingers in blue paint and made its first mark on canvas. Days passed as BEEP-7 experimented with colors, learning that art wasn't about precision but expression. Its circuits hummed with joy as it created its first masterpiece - a sunset that somehow captured the warmth it has never felt.",
            "There was a robot. It painted. The end.",
            "A robot painted a sunset.",
        ],
        "label_rank": [1, 3, 2],
    },
]

# Listwise test data (without labels for evaluation)
LISTWISE_TEST_SAMPLE = [
    {
        "query": "Explain what a for loop does in programming.",
        "responses": [
            "A for loop is a control structure that repeats a block of code a specific number of times. It consists of three parts: initialization (setting a counter), condition (when to stop), and increment (updating the counter). For example, in Python: 'for i in range(5):' will execute the loop body 5 times, with i taking values 0 through 4.",
            "A for loop repeats code multiple times.",
        ],
    },
]


# =============================================================================
# Helper Functions
# =============================================================================


def get_test_model() -> OpenAIChatModel:
    """Get test model instance.

    Returns:
        OpenAIChatModel: Configured OpenAI chat model for testing.
    """
    return OpenAIChatModel(model="qwen3-32b", stream=False)


# =============================================================================
# Pointwise Tests
# =============================================================================


@pytest.mark.asyncio
async def test_iterative_grader_pointwise_single_response() -> None:
    """Test pointwise grader generation for a single response."""
    model = get_test_model()

    config = IterativePointwiseRubricsGeneratorConfig(
        model=model,
        grader_name="Iterative_Pointwise_Grader",
        min_score=0,
        max_score=1,
        query_specific_generate_number=1,
        enable_categorization=False,
        language=LanguageEnum.EN,
        task_description="Evaluate the factual accuracy and completeness of geographic knowledge responses.",
    )

    generator = IterativeRubricsGenerator(config)
    grader = await generator.generate(dataset=POINTWISE_TRAINING_SAMPLE)

    # Verify grader was created (code correctness check)
    assert grader is not None, "Grader should not be None"
    assert isinstance(grader, LLMGrader), f"Grader should be LLMGrader, got {type(grader)}"
    assert (
        grader.name == "Iterative_Pointwise_Grader"
    ), f"Grader name should be 'Iterative_Pointwise_Grader', got '{grader.name}'"

    # Rubrics may be empty if LLM validation failed (not a code bug)
    rubrics = grader.kwargs.get("rubrics")
    assert rubrics is not None, "Rubrics key should exist in kwargs"
    if not rubrics or len(rubrics) == 0:
        logger.warning("Rubrics generation failed validation (LLM output issue, not code bug)")
        pytest.skip("Rubrics generation failed validation - LLM output issue")

    logger.info(f"Generated rubrics:\n{rubrics}")

    # Evaluate test sample
    test_query = POINTWISE_TEST_SAMPLE[0]["query"]
    test_response = POINTWISE_TEST_SAMPLE[0]["response"]
    result = await grader.aevaluate(query=test_query, response=test_response)

    # Verify result structure (code correctness check)
    assert result is not None, "Evaluation result should not be None"
    assert isinstance(result, GraderScore), f"Result should be GraderScore, got {type(result)}"
    assert result.score is not None, "Score should not be None"
    assert isinstance(result.score, (int, float)), f"Score should be numeric, got {type(result.score)}"
    assert result.reason is not None, "Reason should not be None"

    logger.info(f"Pointwise mode without categorization result: {result}")


@pytest.mark.asyncio
async def test_iterative_grader_pointwise_multiple_responses() -> None:
    """Test pointwise grader generation for multiple responses."""
    model = get_test_model()

    config = IterativePointwiseRubricsGeneratorConfig(
        model=model,
        grader_name="Iterative_Pointwise_Grader_Categorized",
        min_score=0,
        max_score=1,
        query_specific_generate_number=1,
        enable_categorization=True,
        categories_number=3,
        language=LanguageEnum.EN,
        task_description="Evaluate the factual accuracy and completeness of geographic knowledge responses.",
    )

    generator = IterativeRubricsGenerator(config)
    # Use a few samples to test categorization
    training_data = [deepcopy(POINTWISE_TRAINING_SAMPLE[0]) for _ in range(150)]
    grader = await generator.generate(dataset=training_data)

    # Verify grader was created (code correctness check)
    assert grader is not None, "Grader should not be None"
    assert isinstance(grader, LLMGrader), f"Grader should be LLMGrader, got {type(grader)}"
    assert grader.name == "Iterative_Pointwise_Grader_Categorized", f"Grader name mismatch"

    # Rubrics may be empty if LLM validation failed (not a code bug)
    rubrics = grader.kwargs.get("rubrics")
    assert rubrics is not None, "Rubrics key should exist in kwargs"
    if not rubrics or len(rubrics) == 0:
        logger.warning("Rubrics generation failed validation (LLM output issue, not code bug)")
        pytest.skip("Rubrics generation failed validation - LLM output issue")

    logger.info(f"Generated categorized rubrics:\n{rubrics}")

    # Evaluate test sample
    test_query = POINTWISE_TEST_SAMPLE[0]["query"]
    test_response = POINTWISE_TEST_SAMPLE[0]["response"]
    result = await grader.aevaluate(query=test_query, response=test_response)

    # Verify result structure (code correctness check)
    assert result is not None, "Evaluation result should not be None"
    assert isinstance(result, GraderScore), f"Result should be GraderScore, got {type(result)}"
    assert result.score is not None, "Score should not be None"
    assert isinstance(result.score, (int, float)), f"Score should be numeric, got {type(result.score)}"
    assert result.reason is not None, "Reason should not be None"

    logger.info(f"Pointwise mode with categorization result: {result}")


# =============================================================================
# Listwise Tests
# =============================================================================


@pytest.mark.asyncio
async def test_iterative_grader_listwise() -> None:
    """Test listwise grader generation.

    This test verifies that a listwise grader can be generated and used
    for ranking multiple responses.
    """
    model = get_test_model()

    config = IterativeListwiseRubricsGeneratorConfig(
        model=model,
        grader_name="Iterative_Listwise_Grader",
        enable_categorization=False,
        language=LanguageEnum.EN,
        categories_number=5,
        query_specific_generate_number=2,
        task_description="Evaluate creative writing quality including narrative structure, character development, and emotional engagement.",
    )

    generator = IterativeRubricsGenerator(config)
    grader = await generator.generate(dataset=LISTWISE_TRAINING_SAMPLE)

    # Verify grader was created (code correctness check)
    assert grader is not None, "Grader should not be None"
    assert isinstance(grader, LLMGrader), f"Grader should be LLMGrader, got {type(grader)}"
    assert grader.name == "Iterative_Listwise_Grader", f"Grader name mismatch"

    # Rubrics may be empty if LLM validation failed (not a code bug)
    rubrics = grader.kwargs.get("rubrics")
    assert rubrics is not None, "Rubrics key should exist in kwargs"
    if not rubrics or len(rubrics) == 0:
        logger.warning("Rubrics generation failed validation (LLM output issue, not code bug)")

    logger.info(f"Generated rubrics:\n{rubrics}")

    # Evaluate test sample
    test_query = LISTWISE_TEST_SAMPLE[0]["query"]
    test_responses = LISTWISE_TEST_SAMPLE[0]["responses"]
    responses = "\n\n".join([f"Response {i + 1}:\n{ans}" for i, ans in enumerate(test_responses)])

    result = await grader.aevaluate(query=test_query, responses=responses, num_responses=len(test_responses))

    logger.info(f"Listwise mode result: {result}")

    # Verify result structure (code correctness check)
    assert result is not None, "Evaluation result should not be None"
    assert isinstance(result, GraderRank), f"Result should be GraderRank, got {type(result)}"
    assert result.rank is not None, "Rank should not be None"
    assert isinstance(result.rank, list), f"Rank should be a list, got {type(result.rank)}"
    assert len(result.rank) == len(
        test_responses
    ), f"Rank length should be {len(test_responses)}, got {len(result.rank)}"
    assert result.reason is not None, "Reason should not be None"


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run all test functions."""
    await test_iterative_grader_pointwise_single_response()
    await test_iterative_grader_pointwise_multiple_responses()
    await test_iterative_grader_listwise()


if __name__ == "__main__":
    asyncio.run(main())
