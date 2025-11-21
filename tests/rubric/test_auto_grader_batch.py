# -*- coding: utf-8 -*-
"""
AutoGrader Batch Test

Test AutoGrader batch mode with processed data.

Features:
1. Load processed data
2. Test batch mode - generate grader and evaluate samples
3. Save results to file
4. Calculate accuracy metrics
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from loguru import logger

from rm_gallery.core.grader.auto.auto_grader import AutoGrader
from rm_gallery.core.grader.auto.auto_rubrics import AutoRubricsConfig
from rm_gallery.core.grader.base import GraderMode
from rm_gallery.core.model import OpenAIChatModel
from rm_gallery.core.schema.data import EvalCase

# pylint: disable=line-too-long,too-many-nested-blocks,redefined-outer-name

# ============================================================================
# Utility Classes and Functions
# ============================================================================


class AccuracyCalculator:
    """Utility class for calculating accuracy metrics."""

    @staticmethod
    def calculate_accuracy(
        grader_mode: GraderMode,
        test_samples: List[EvalCase],
        results: List,
    ) -> Dict[str, Any]:
        """Calculate accuracy metrics based on grader mode."""
        if grader_mode == GraderMode.POINTWISE:
            return AccuracyCalculator._calculate_pointwise_accuracy(
                test_samples,
                results,
            )
        elif grader_mode == GraderMode.LISTWISE:
            return AccuracyCalculator._calculate_listwise_accuracy(
                test_samples,
                results,
            )
        else:
            raise ValueError(f"Unsupported grader mode: {grader_mode}")

    @staticmethod
    def _calculate_pointwise_accuracy(
        test_samples: List[EvalCase],
        results: List,
    ) -> Dict[str, Any]:
        """Calculate accuracy for pointwise evaluation."""
        correct = 0
        total = 0

        for sample, result in zip(test_samples, results):
            result_list = result if isinstance(result, list) else [result]

            for idx, item in enumerate(sample.outputs):
                if idx < len(result_list):
                    expected_score = item.get("score")
                    if expected_score is not None:
                        pred_result = result_list[idx]
                        predicted_score = (
                            pred_result.score if hasattr(pred_result, "score") else None
                        )
                        if predicted_score is not None:
                            total += 1
                            if predicted_score == expected_score:
                                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mode": "pointwise",
        }

    @staticmethod
    def _calculate_listwise_accuracy(
        test_samples: List[EvalCase],
        results: List,
    ) -> Dict[str, Any]:
        """Calculate accuracy for listwise evaluation."""
        correct = 0
        total = 0

        for idx, (sample, result) in enumerate(zip(test_samples, results)):
            expected_ranks = [
                item.get("rank")
                for item in sample.outputs
                if item.get("rank") is not None
            ]

            predicted_ranks = []
            if isinstance(result, list):
                predicted_ranks = [
                    item.score for item in result if hasattr(item, "score")
                ]
            elif hasattr(result, "rank"):
                predicted_ranks = result.rank
            elif hasattr(result, "score"):
                predicted_ranks = [result.score]

            if idx == 0:  # Debug first sample
                logger.debug(f"Sample 0 - Expected ranks: {expected_ranks}")
                logger.debug(f"Sample 0 - Predicted ranks: {predicted_ranks}")

            if (
                expected_ranks
                and predicted_ranks
                and len(expected_ranks) == len(predicted_ranks)
            ):
                total += 1
                expected_order = AccuracyCalculator._get_relative_order(
                    expected_ranks,
                )
                predicted_order = AccuracyCalculator._get_relative_order(
                    predicted_ranks,
                )

                if expected_order == predicted_order:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"Listwise accuracy: {correct}/{total} = {accuracy:.2%}")
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "mode": "listwise",
        }

    @staticmethod
    def _get_relative_order(values: List[float]) -> List[int]:
        """Convert values to relative order rankings."""
        indexed_values = list(enumerate(values))
        indexed_values.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in indexed_values]


class DataLoader:
    """Utility class for loading test data."""

    @staticmethod
    def load_eval_cases(
        file_path: str,
        max_samples: Optional[int] = None,
    ) -> List[EvalCase]:
        """Load eval cases from JSONL file."""
        logger.info(f"Loading eval cases from {file_path}")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        eval_cases = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    sample = EvalCase(**data)
                    eval_cases.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to parse line {i+1}: {e}")
                    continue

        logger.info(f"Successfully loaded {len(eval_cases)} eval cases")
        return eval_cases


class TestConfig:
    """Configuration for batch tests."""

    def __init__(
        self,
        grader_mode: str = "listwise",
        sampling_mode: str = "all_samples",
        aggregation_mode: str = "merge_similar",
        language: str = "zh",
        max_train_samples: int = 50,
        max_test_samples: int = 20,
        generate_number: int = 2,
        max_epochs: int = 3,
        batch_size: int = 10,
    ):
        self.grader_mode = grader_mode
        self.sampling_mode = sampling_mode
        self.aggregation_mode = aggregation_mode
        self.language = language
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.generate_number = generate_number
        self.max_epochs = max_epochs
        self.batch_size = batch_size


# ============================================================================
# Test Functions
# ============================================================================


class AutoGraderBatchTester:
    """Main class for running AutoGrader batch tests."""

    def __init__(self, model: OpenAIChatModel):
        self.model = model
        self.accuracy_calculator = AccuracyCalculator()
        self.data_loader = DataLoader()

    @pytest.mark.asyncio
    async def run_batch_test(
        self,
        eval_cases: List[EvalCase],
        config: TestConfig,
        grader_name: str = "AutoGrader_Batch_Test",
    ) -> Dict[str, Any]:
        """Run a complete batch test with the given configuration."""

        # Split data into training and testing
        train_samples = eval_cases[: config.max_train_samples]
        test_samples = eval_cases[
            config.max_train_samples : config.max_train_samples
            + config.max_test_samples
        ]

        logger.info(f"Using {len(train_samples)} samples for training")
        logger.info(f"Using {len(test_samples)} samples for testing")

        if not train_samples:
            raise ValueError("No training samples available")
        if not test_samples:
            raise ValueError("No test samples available")

        # Create AutoGrader
        auto_grader = AutoGrader.create(
            self.model,
            method="auto_rubrics",
            grader_name=grader_name,
            method_config=AutoRubricsConfig(
                grader_mode=config.grader_mode,
                sampling_mode=config.sampling_mode,
                aggregation_mode=config.aggregation_mode,
                language=config.language,
                generate_number=config.generate_number,
                max_epochs=config.max_epochs,
                batch_size=config.batch_size,
                mcr_batch_size=8,
            ),
        )

        # Train the grader
        logger.info("Training grader...")
        grader = await auto_grader.aevaluate_batch(train_samples)
        logger.info(f"Grader trained: {grader.name} ({grader.mode.value})")

        # Evaluate test samples
        logger.info("Evaluating test samples...")
        results = await grader.aevaluate_batch(test_samples)

        # Calculate accuracy
        accuracy_metrics = self.accuracy_calculator.calculate_accuracy(
            grader.mode,
            test_samples,
            results,
        )

        # Prepare serializable results
        serializable_results = []
        for item in results:
            if isinstance(item, list):
                serializable_results.append(
                    [sub_item.model_dump() for sub_item in item],
                )
            else:
                serializable_results.append(item.model_dump())

        # Compile final results
        final_results = {
            "grader_info": {
                "name": grader.name,
                "mode": grader.mode.value,
                "language": grader.language.value,
                "rubrics": grader.rubrics,
            },
            "config": {
                "grader_mode": config.grader_mode,
                "sampling_mode": config.sampling_mode,
                "aggregation_mode": config.aggregation_mode,
                "language": config.language,
            },
            "data_info": {
                "training_samples_count": len(train_samples),
                "test_samples_count": len(test_samples),
            },
            "evaluation_results": serializable_results,
            "accuracy_metrics": accuracy_metrics,
        }

        logger.info(
            f"Batch test completed. Accuracy: {accuracy_metrics['accuracy']:.2%}",
        )
        return final_results

    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save test results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to: {output_path}")


# ============================================================================
# Pytest Test Cases
# ============================================================================


@pytest.fixture
def test_model() -> OpenAIChatModel:
    """Fixture for test model."""
    return OpenAIChatModel(model="qwen3-32b", stream=False)


@pytest.fixture
def sample_data_path() -> str:
    """Fixture for sample data path."""
    return "./data/processed_mxc/train_samples_结论实用.jsonl"


@pytest.fixture
def batch_tester(test_model: OpenAIChatModel) -> AutoGraderBatchTester:
    """Fixture for batch tester."""
    return AutoGraderBatchTester(test_model)


@pytest.mark.asyncio
async def test_listwise_batch_evaluation(
    batch_tester: AutoGraderBatchTester,  # pylint: disable=redefined-outer-name
    sample_data_path: str,  # pylint: disable=redefined-outer-name
) -> None:
    """Test listwise batch evaluation."""
    # Skip if data file doesn't exist
    if not Path(sample_data_path).exists():
        pytest.skip(f"Data file not found: {sample_data_path}")

    # Load data
    eval_cases = batch_tester.data_loader.load_eval_cases(
        sample_data_path,
        max_samples=100,
    )

    # Configure test
    config = TestConfig(
        grader_mode="listwise",
        aggregation_mode="merge_similar",
        sampling_mode="all_samples",
        language="zh",
        max_train_samples=30,
        max_test_samples=10,
    )

    # Run test
    results = await batch_tester.run_batch_test(
        eval_cases,
        config,
        "Test_Listwise_Grader",
    )

    # Assertions
    assert results is not None
    assert "accuracy_metrics" in results
    assert "grader_info" in results
    assert results["accuracy_metrics"]["mode"] == "listwise"
    assert results["data_info"]["training_samples_count"] == 30
    assert results["data_info"]["test_samples_count"] == 10

    logger.info(
        f"Listwise test completed with accuracy: {results['accuracy_metrics']['accuracy']:.2%}",
    )


@pytest.mark.asyncio
async def test_pointwise_batch_evaluation(
    batch_tester: AutoGraderBatchTester,  # pylint: disable=redefined-outer-name
    sample_data_path: str,  # pylint: disable=redefined-outer-name
) -> None:
    """Test pointwise batch evaluation."""
    # Skip if data file doesn't exist
    if not Path(sample_data_path).exists():
        pytest.skip(f"Data file not found: {sample_data_path}")

    # Load data
    eval_cases = batch_tester.data_loader.load_eval_cases(
        sample_data_path,
        max_samples=100,
    )

    # Configure test
    config = TestConfig(
        grader_mode="pointwise",
        aggregation_mode="keep_all",
        sampling_mode="all_samples",
        language="zh",
        max_train_samples=30,
        max_test_samples=10,
    )

    # Run test
    results = await batch_tester.run_batch_test(
        eval_cases,
        config,
        "Test_Pointwise_Grader",
    )

    # Assertions
    assert results is not None
    assert "accuracy_metrics" in results
    assert results["accuracy_metrics"]["mode"] == "pointwise"

    logger.info(
        f"Pointwise test completed with accuracy: {results['accuracy_metrics']['accuracy']:.2%}",
    )


@pytest.mark.asyncio
async def test_batch_with_results_saving(
    batch_tester: AutoGraderBatchTester,  # pylint: disable=redefined-outer-name
    sample_data_path: str,  # pylint: disable=redefined-outer-name
    tmp_path: Path,
) -> None:
    """Test batch evaluation with results saving."""
    # Skip if data file doesn't exist
    if not Path(sample_data_path).exists():
        pytest.skip(f"Data file not found: {sample_data_path}")

    # Load data
    eval_cases = batch_tester.data_loader.load_eval_cases(
        sample_data_path,
        max_samples=50,
    )

    # Configure test
    config = TestConfig(
        grader_mode="listwise",
        max_train_samples=20,
        max_test_samples=5,
    )

    # Run test
    results = await batch_tester.run_batch_test(eval_cases, config)

    # Save results
    output_path = tmp_path / "test_results.json"
    batch_tester.save_results(results, output_path)

    # Verify file was created and contains valid JSON
    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8") as f:
        loaded_results = json.load(f)

    assert loaded_results == results
    logger.info("Results saving test completed successfully")


# ============================================================================
# Main Execution for Standalone Testing
# ============================================================================


async def main() -> None:
    """Run batch tests manually for development/debugging."""
    train_file = "./data/processed_mxc/train_samples.jsonl"

    # Check if data file exists
    if not Path(train_file).exists():
        logger.warning(f"Data file not found: {train_file}")
        logger.info(
            "Skipping main execution - use pytest to run tests with mock data",
        )
        return

    # Initialize components
    model = OpenAIChatModel(model="qwen3-32b", stream=False)
    batch_tester = AutoGraderBatchTester(model)

    try:
        # Load data
        logger.info("Loading eval cases...")
        eval_cases = batch_tester.data_loader.load_eval_cases(
            train_file,
            max_samples=100,
        )

        if not eval_cases:
            logger.error("No valid eval cases loaded")
            return

        # Test configuration
        config = TestConfig(
            grader_mode="listwise",
            aggregation_mode="merge_similar",
            sampling_mode="all_samples",
            language="zh",
            generate_number=1,
            max_train_samples=50,
            max_test_samples=20,
        )

        # Run batch test
        logger.info("Starting batch test...")
        results = await batch_tester.run_batch_test(
            eval_cases,
            config,
            "Manual_Batch_Test_Grader",
        )

        # Save results
        output_path = Path(
            "results/auto_grader_batch/manual_test_results.json",
        )
        batch_tester.save_results(results, output_path)

        # Print summary
        accuracy = results["accuracy_metrics"]["accuracy"]
        logger.info("Batch test completed successfully!")
        logger.info(f"Final accuracy: {accuracy:.2%}")
        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during batch test: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
