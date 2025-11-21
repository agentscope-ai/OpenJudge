# -*- coding: utf-8 -*-
"""
AutoGrader Single Test Module

This module contains unit tests for the AutoGrader functionality
with single eval cases and different configurations.
"""

import asyncio

import pytest

from rm_gallery.core.grader.auto.auto_grader import AutoGrader
from rm_gallery.core.grader.auto.auto_rubrics import AutoRubricsConfig
from rm_gallery.core.model import OpenAIChatModel
from rm_gallery.core.schema.data import EvalCase


# Test data fixtures
def get_pointwise_sample() -> EvalCase:
    """Sample data with score labels for pointwise evaluation."""
    return EvalCase(
        input={"query": "What is the capital of France?"},
        outputs=[{"answer": "The capital of France is Paris.", "score": 1}],
    )


def get_listwise_sample() -> EvalCase:
    """Sample data with rank labels for listwise evaluation."""
    return EvalCase(
        input={"query": "What is the capital of France?"},
        outputs=[
            {"answer": "The capital of France is Paris.", "rank": 2},
            {"answer": "The capital of France is not Paris.", "rank": 1},
        ],
    )


def get_unlabeled_sample() -> EvalCase:
    """Sample data without labels for evaluation."""
    return EvalCase(
        input={"query": "What is the capital of Germany?"},
        outputs=[
            {"answer": "The capital of Germany is not Berlin."},
            {"answer": "The capital of Germany is Berlin."},
        ],
    )


def get_test_model() -> OpenAIChatModel:
    """Get test model instance."""
    return OpenAIChatModel(model="qwen3-32b", stream=False)


@pytest.mark.asyncio
async def test_auto_grader_with_default_config() -> None:
    """Test AutoGrader with default configuration."""
    model = get_test_model()
    training_data = [get_pointwise_sample()]
    test_data = [get_unlabeled_sample()]

    # Create grader with default config
    auto_grader = AutoGrader.create(
        model,
        method="auto_rubrics",
        grader_name="AutoRubrics_Default_Grader",
        method_config=AutoRubricsConfig(),
    )

    # Train the grader
    grader = await auto_grader.aevaluate_batch(training_data)
    assert grader is not None, "Grader should be created successfully"

    # Evaluate test data
    result = await grader.aevaluate_batch(
        eval_cases=[test_data],
        parser=None,
    )

    assert result is not None, "Evaluation result should not be None"
    print(f"Default config result: {result}")


@pytest.mark.asyncio
async def test_auto_grader_with_custom_config() -> None:
    """Test AutoGrader with custom configuration."""
    model = get_test_model()
    training_data = [get_listwise_sample()]
    test_data = [get_unlabeled_sample()]

    # Create grader with custom config
    custom_config = AutoRubricsConfig(
        grader_mode="listwise",
        sampling_mode="all_samples",
        aggregation_mode="keep_all",
        language="en",
        generate_number=2,
        batch_size=5,
        mcr_batch_size=8,
    )

    auto_grader = AutoGrader.create(
        model,
        method="auto_rubrics",
        grader_name="AutoRubrics_Custom_Grader",
        method_config=custom_config,
    )

    # Train the grader
    grader = await auto_grader.aevaluate_batch(training_data)
    assert grader is not None, "Grader should be created successfully"

    # Evaluate test data
    result = await grader.aevaluate_batch(
        eval_cases=[test_data],
        parser=None,
    )

    assert result is not None, "Evaluation result should not be None"
    print(f"Custom config result: {result}")


@pytest.mark.asyncio
async def test_auto_grader_comparison() -> None:
    """Test and compare results from different configurations."""
    model = get_test_model()
    pointwise_data = [get_pointwise_sample()]
    listwise_data = [get_listwise_sample()]
    test_data = [get_unlabeled_sample()]

    # Default grader
    default_grader_factory = AutoGrader.create(
        model,
        method="auto_rubrics",
        grader_name="AutoRubrics_Default",
        method_config=AutoRubricsConfig(),
    )

    # Custom grader
    custom_grader_factory = AutoGrader.create(
        model,
        method="auto_rubrics",
        grader_name="AutoRubrics_Custom",
        method_config=AutoRubricsConfig(
            grader_mode="listwise",
            sampling_mode="all_samples",
            aggregation_mode="keep_all",
            language="en",
            generate_number=1,
            batch_size=5,
            mcr_batch_size=8,
        ),
    )

    # Train both graders
    default_grader = await default_grader_factory.aevaluate_batch(pointwise_data)
    custom_grader = await custom_grader_factory.aevaluate_batch(listwise_data)

    # Evaluate with both graders
    default_result = await default_grader.aevaluate_batch(
        eval_cases=[test_data],
        parser=None,
    )
    custom_result = await custom_grader.aevaluate_batch(
        eval_cases=[test_data],
        parser=None,
    )

    # Assertions
    assert default_result is not None, "Default grader result should not be None"
    assert custom_result is not None, "Custom grader result should not be None"

    print(f"Default grader result: {default_result}")
    print(f"Custom grader result: {custom_result}")
    print("Comparison test completed successfully")


# Main execution for standalone testing
async def main() -> None:
    """Run all tests manually."""
    print("Running AutoGrader tests...")

    print("\n1. Testing default configuration...")
    await test_auto_grader_with_default_config()

    print("\n2. Testing custom configuration...")
    await test_auto_grader_with_custom_config()

    print("\n3. Testing comparison...")
    await test_auto_grader_comparison()

    print("\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
