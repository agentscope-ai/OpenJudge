# -*- coding: utf-8 -*-
"""Simple Rubric Generator test module.

This module contains unit tests for the Simple Rubric Generator functionality
which generates evaluation rubrics from task descriptions.

Demonstrates workflow:
1. Create generator with task description configuration
2. Generate rubrics from task description (no labeled data required)
3. Optionally create a complete LLMGrader for evaluation

Supports both TaskBasedRubricGenerator (rubrics only) and
SimpleRubricsGenerator (complete LLMGrader).

Example:
    Run all tests:
    ```bash
    pytest tests/generator/test_simple_rubric.py -v
    ```

    Run a specific test:
    ```bash
    pytest tests/generator/test_simple_rubric.py::test_task_based_rubric_generator -v
    ```

    Run directly as a script:
    ```bash
    python tests/generator/test_simple_rubric.py
    ```
"""

import asyncio

import pytest
from loguru import logger

from openjudge.generator.simple_rubric import (
    SimpleRubricsGenerator,
    SimpleRubricsGeneratorConfig,
    TaskBasedRubricGenerator,
)
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode, GraderScore
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum

# =============================================================================
# Test Data
# =============================================================================

# Task description for testing
TEST_TASK_DESCRIPTION = "English to Chinese translation assistant that helps users translate technical documents"

TEST_SCENARIO = "Users need to translate technical documentation from English to fluent, accurate Chinese"

# Sample queries for context
TEST_SAMPLE_QUERIES = [
    "Translate this paragraph into Chinese: 'Machine learning is a subset of artificial intelligence.'",
    "Translate the following technical term: 'neural network'",
    "How would you translate 'API endpoint' into Chinese?",
]

# Test data for evaluation
TEST_EVALUATION_DATA = {
    "query": "Translate this sentence: 'The database query returned an error.'",
    "response": "数据库查询返回了一个错误。",
}


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
# TaskBasedRubricGenerator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_task_based_rubric_generator() -> None:
    """Test TaskBasedRubricGenerator for generating rubrics from task description."""
    model = get_test_model()

    generator = TaskBasedRubricGenerator(
        model=model,
        task_description=TEST_TASK_DESCRIPTION,
        scenario=TEST_SCENARIO,
        language=LanguageEnum.EN,
    )
    rubrics = await generator.generate(sample_queries=TEST_SAMPLE_QUERIES)

    # Verify rubrics were generated
    assert rubrics is not None, "Rubrics should not be None"
    assert isinstance(rubrics, list), f"Rubrics should be a list, got {type(rubrics)}"
    assert len(rubrics) > 0, "Rubrics list should not be empty"

    # Verify each rubric is a non-empty string
    for i, rubric in enumerate(rubrics):
        assert isinstance(rubric, str), f"Rubric {i} should be a string, got {type(rubric)}"
        assert len(rubric) > 0, f"Rubric {i} should not be empty"

    logger.info(f"Generated {len(rubrics)} rubrics:")
    for i, rubric in enumerate(rubrics, 1):
        logger.info(f"  {i}. {rubric}")


@pytest.mark.asyncio
async def test_task_based_rubric_generator_chinese() -> None:
    """Test TaskBasedRubricGenerator with Chinese language prompts."""
    model = get_test_model()

    generator = TaskBasedRubricGenerator(
        model=model,
        task_description="代码审查助手，帮助开发者检查 Python 代码质量",
        scenario="开发者需要对代码进行质量检查和改进建议",
        language=LanguageEnum.ZH,
    )
    rubrics = await generator.generate(
        sample_queries=[
            "请审查这段代码是否有bug",
            "这个函数的命名是否合理？",
        ]
    )

    # Verify rubrics were generated
    assert rubrics is not None, "Rubrics should not be None"
    assert isinstance(rubrics, list), f"Rubrics should be a list, got {type(rubrics)}"
    assert len(rubrics) > 0, "Rubrics list should not be empty"

    logger.info(f"Generated {len(rubrics)} Chinese rubrics:")
    for i, rubric in enumerate(rubrics, 1):
        logger.info(f"  {i}. {rubric}")


@pytest.mark.asyncio
async def test_task_based_rubric_generator_default_fallback() -> None:
    """Test that default rubrics are returned when generation fails."""
    model = get_test_model()

    default_rubrics = [
        "Custom default rubric 1",
        "Custom default rubric 2",
    ]

    generator = TaskBasedRubricGenerator(
        model=model,
        task_description=TEST_TASK_DESCRIPTION,
        scenario=TEST_SCENARIO,
        default_rubrics=default_rubrics,
    )

    # Verify default_rubrics are set
    assert generator.default_rubrics == default_rubrics


# =============================================================================
# SimpleRubricsGenerator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_simple_rubrics_generator_pointwise() -> None:
    """Test SimpleRubricsGenerator for creating a complete LLMGrader (pointwise mode)."""
    model = get_test_model()

    config = SimpleRubricsGeneratorConfig(
        grader_name="Translation_Quality_Grader",
        model=model,
        grader_mode=GraderMode.POINTWISE,
        task_description=TEST_TASK_DESCRIPTION,
        scenario=TEST_SCENARIO,
        language=LanguageEnum.EN,
        min_score=0,
        max_score=5,
    )

    generator = SimpleRubricsGenerator(config)
    grader = await generator.generate(
        dataset=[],
        sample_queries=TEST_SAMPLE_QUERIES,
    )

    # Verify grader was created
    assert grader is not None, "Grader should not be None"
    assert isinstance(grader, LLMGrader), f"Grader should be LLMGrader, got {type(grader)}"
    assert grader.name == "Translation_Quality_Grader", f"Grader name mismatch"

    # Verify rubrics were generated
    rubrics = grader.kwargs.get("rubrics")
    assert rubrics is not None, "Rubrics key should exist in kwargs"
    assert len(rubrics) > 0, "Rubrics should not be empty"

    logger.info(f"Generated rubrics:\n{rubrics}")

    # Evaluate test sample
    result = await grader.aevaluate(
        query=TEST_EVALUATION_DATA["query"],
        response=TEST_EVALUATION_DATA["response"],
    )

    # Verify result structure
    assert result is not None, "Evaluation result should not be None"
    assert isinstance(result, GraderScore), f"Result should be GraderScore, got {type(result)}"
    assert result.score is not None, "Score should not be None"
    assert isinstance(result.score, (int, float)), f"Score should be numeric, got {type(result.score)}"
    assert result.reason is not None, "Reason should not be None"

    logger.info(f"Pointwise evaluation result: {result}")


@pytest.mark.asyncio
async def test_simple_rubrics_generator_extract_queries_from_dataset() -> None:
    """Test that SimpleRubricsGenerator extracts sample queries from dataset."""
    model = get_test_model()

    config = SimpleRubricsGeneratorConfig(
        grader_name="Auto_Query_Extraction_Grader",
        model=model,
        task_description=TEST_TASK_DESCRIPTION,
        scenario=TEST_SCENARIO,
    )

    # Provide dataset with queries but no explicit sample_queries
    dataset = [
        {"query": "Translate: Hello world", "response": "你好世界"},
        {"query": "Translate: Good morning", "response": "早上好"},
        {"query": "Translate: Thank you", "response": "谢谢"},
    ]

    generator = SimpleRubricsGenerator(config)
    grader = await generator.generate(dataset=dataset)  # No sample_queries provided

    # Verify grader was created
    assert grader is not None, "Grader should not be None"
    assert isinstance(grader, LLMGrader), f"Grader should be LLMGrader, got {type(grader)}"

    # Verify rubrics were generated (queries should be extracted from dataset)
    rubrics = grader.kwargs.get("rubrics")
    assert rubrics is not None, "Rubrics key should exist in kwargs"
    assert len(rubrics) > 0, "Rubrics should not be empty"

    logger.info(f"Generated rubrics from dataset queries:\n{rubrics}")


@pytest.mark.asyncio
async def test_simple_rubrics_generator_with_model_dict() -> None:
    """Test SimpleRubricsGenerator with model configuration as dictionary."""
    config = SimpleRubricsGeneratorConfig(
        grader_name="Dict_Config_Grader",
        model={"model": "qwen3-32b", "stream": False},  # Dict instead of model instance
        task_description=TEST_TASK_DESCRIPTION,
        scenario=TEST_SCENARIO,
    )

    generator = SimpleRubricsGenerator(config)

    # Verify model was converted from dict to OpenAIChatModel
    assert generator.config.model is not None
    assert isinstance(generator.config.model, OpenAIChatModel)

    grader = await generator.generate(dataset=[], sample_queries=TEST_SAMPLE_QUERIES)

    assert grader is not None, "Grader should not be None"
    assert isinstance(grader, LLMGrader), f"Grader should be LLMGrader, got {type(grader)}"

    logger.info("Successfully created grader with dict model config")


# =============================================================================
# Main Entry Point
# =============================================================================


async def main() -> None:
    """Run all test functions."""
    logger.info("Running TaskBasedRubricGenerator tests...")
    await test_task_based_rubric_generator()
    await test_task_based_rubric_generator_chinese()
    await test_task_based_rubric_generator_default_fallback()

    logger.info("\nRunning SimpleRubricsGenerator tests...")
    await test_simple_rubrics_generator_pointwise()
    await test_simple_rubrics_generator_extract_queries_from_dataset()
    await test_simple_rubrics_generator_with_model_dict()

    logger.info("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
