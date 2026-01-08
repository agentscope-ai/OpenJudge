"""
LangSmith Integration Cookbook for OpenJudge

This example demonstrates how to integrate OpenJudge with LangSmith
using both individual graders and GradingRunner approaches.
"""

import asyncio

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.common.correctness import CorrectnessGrader
from openjudge.graders.common.relevance import RelevanceGrader
from openjudge.graders.schema import GraderError, GraderRank, GraderResult, GraderScore
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GradingRunner
from openjudge.utils.mapping import parse_data_with_mapper

# Load environment variables from .env file
load_dotenv()


def create_langsmith_evaluator(grader: BaseGrader, mapper: dict | None = None):
    """
    Create a LangSmith-compatible evaluator from an OpenJudge grader.

    Args:
        grader: An OpenJudge grader instance
        mapper: A dictionary mapping source keys to target keys for data transformation

    Returns:
        A LangSmith-compatible evaluator function
    """

    def langsmith_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """
        LangSmith evaluator function.

        Args:
            inputs: The inputs from LangSmith example
            outputs: The actual outputs from LangSmith run
            reference_outputs: The expected outputs from LangSmith example

        Returns:
            A dictionary containing the evaluation results with score and reasoning
        """
        try:
            # Prepare data for evaluation
            data = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference_outputs}

            # Parse and map the data using the mapper
            mapped_data = parse_data_with_mapper(data, mapper)

            # Execute OpenJudge evaluation with the mapped data
            result: GraderResult = asyncio.run(grader.aevaluate(**mapped_data))

            # Convert OpenJudge result to LangSmith format
            if isinstance(result, GraderScore):
                return {
                    "key": grader.name,  # The feedback key for LangSmith
                    "score": result.score,
                    "comment": getattr(result, "reason", ""),
                }
            elif isinstance(result, GraderRank):
                return {
                    "key": grader.name,
                    "score": getattr(result, "rank", 0),
                    "comment": getattr(result, "reason", ""),
                }
            elif isinstance(result, GraderError):
                return {"key": grader.name, "score": 0.0, "comment": f"Error: {result.error}"}
            else:
                return {"key": grader.name, "score": 0.0, "comment": f"Unsupported result type {type(result)}"}
        except Exception as e:
            # Handle any unexpected errors during evaluation
            return {"key": grader.name, "score": 0.0, "comment": f"Evaluation failed: {str(e)}"}

    return langsmith_evaluator


class LangSmithBatchEvaluator:
    """Batch evaluator that combines multiple OpenJudge graders"""

    def __init__(self, model=None, mapper: dict | None = None):
        """
        Initialize the batch evaluator with a GradingRunner.
        """
        if model is None:
            model = OpenAIChatModel(model="qwen3-32b", extra_body={"enable_thinking": False})

        # Define grader configs with their respective mappers
        grader_configs = {
            "relevance": (RelevanceGrader(model=model), mapper),
            "correctness": (CorrectnessGrader(model=model), mapper),
        }

        # Configure the runner with multiple graders and their mappers
        self.runner = GradingRunner(grader_configs=grader_configs, max_concurrency=10)

    def __call__(self, inputs: dict, outputs: dict, reference_outputs: dict) -> list:
        """
        LangSmith batch evaluator function.

        Args:
            inputs: The inputs from LangSmith example
            outputs: The actual outputs from LangSmith run
            reference_outputs: The expected outputs from LangSmith example


        Returns:
            A list of dictionaries containing results from all graders
        """
        try:
            # Prepare data for batch processing
            data = {"inputs": inputs, "outputs": outputs, "reference_outputs": reference_outputs}

            # Execute batch evaluation using OpenJudge runner
            # Using a new event loop as LangSmith evaluators are synchronous

            batch_results = asyncio.run(self.runner.arun([data]))

            # Convert results to LangSmith format
            formatted_results = []
            for grader_name, grader_results in batch_results.items():
                if grader_results:  # Check if results exist
                    result = grader_results[0]  # We only have one sample
                    if isinstance(result, GraderScore):
                        formatted_results.append(
                            {"key": grader_name, "score": result.score, "comment": getattr(result, "reason", "")}
                        )
                    elif isinstance(result, GraderRank):
                        formatted_results.append(
                            {
                                "key": grader_name,
                                "score": getattr(result, "rank", 0),
                                "comment": getattr(result, "reason", ""),
                            }
                        )
                    elif isinstance(result, GraderError):
                        formatted_results.append(
                            {"key": grader_name, "score": 0.0, "comment": f"Error: {result.error}"}
                        )
                    else:
                        formatted_results.append(
                            {"key": grader_name, "score": 0.0, "comment": f"Unsupported result type {type(result)}"}
                        )

            return formatted_results

        except Exception as e:
            # Handle any errors during batch evaluation
            return [{"key": "batch_evaluation_error", "score": 0.0, "comment": f"Batch evaluation failed: {str(e)}"}]


def qa_application(inputs: dict) -> dict:
    """
    The target application to be evaluated.

    Args:
        inputs: Dictionary containing input data

    Returns:
        Dictionary containing the application output
    """
    model = OpenAIChatModel(model="qwen3-32b", extra_body={"enable_thinking": False})
    response = asyncio.run(
        model.achat(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inputs["question"]},
            ]
        )
    )
    return {"answer": response.content}


def run_individual_grader_example():
    """
    Example of using individual graders with LangSmith.
    """
    print("Running LangSmith Individual Grader Example...")

    # Create examples with inputs and expected outputs
    examples = [
        {
            "inputs": {
                "question": "What is the capital of France?",
            },
            "outputs": {"expected_answer": "Paris"},
        },
        {
            "inputs": {
                "question": "How many planets are in our solar system?",
            },
            "outputs": {"expected_answer": "8"},
        },
        {
            "inputs": {
                "question": "Who wrote Romeo and Juliet?",
            },
            "outputs": {"expected_answer": "William Shakespeare"},
        },
    ]

    # Initialize LangSmith client and create dataset
    client = Client()

    # Check if dataset already exists, if not create it
    try:
        dataset = client.create_dataset(dataset_name="QA Evaluation Dataset")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Dataset already exists: {str(e)}")
            dataset = client.read_dataset(dataset_name="QA Evaluation Dataset")
        else:
            raise e

    # Create examples in the dataset
    try:
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print("Examples added to dataset.")
    except Exception as e:
        print(f"Could not add examples: {str(e)}. They may already exist.")

    # Initialize model for relevance and correctness graders
    model = OpenAIChatModel(model="qwen3-32b", extra_body={"enable_thinking": False})

    # Define mappers for each grader - mapping LangSmith data format to OpenJudge format
    relevance_mapper = {
        "query": "inputs.question",
        "response": "outputs.answer",
    }

    correctness_mapper = {
        "query": "inputs.question",
        "response": "outputs.answer",
        "reference_response": "reference_outputs.expected_answer",
    }

    # Create multiple OpenJudge evaluators with appropriate mappers
    graders = [
        ("relevance", RelevanceGrader(model=model), relevance_mapper),
        ("correctness", CorrectnessGrader(model=model), correctness_mapper),
    ]

    # Convert to LangSmith evaluators
    langsmith_evaluators = [create_langsmith_evaluator(grader, mapper) for _, grader, mapper in graders]

    # Run evaluation
    experiment_results = evaluate(
        qa_application,  # Your LLM application or chain
        data=dataset.name,  # Dataset in LangSmith
        evaluators=langsmith_evaluators,
        experiment_prefix="open_judge_individual_graders",
        description="Evaluating QA application with OpenJudge individual graders",
        max_concurrency=4,
    )

    # Process results
    df = experiment_results.to_pandas()
    print(df.head())


def run_batch_grader_example():
    """
    Example of using GradingRunner with LangSmith.
    """
    print("Running LangSmith Batch Grader Example...")

    # Create examples with inputs and expected outputs
    examples = [
        {"inputs": {"question": "What is the capital of France?"}, "outputs": {"expected_answer": "Paris"}},
        {"inputs": {"question": "How many planets are in our solar system?"}, "outputs": {"expected_answer": "8"}},
        {"inputs": {"question": "Who wrote Romeo and Juliet?"}, "outputs": {"expected_answer": "William Shakespeare"}},
    ]

    # Initialize LangSmith client and create dataset
    client = Client()

    # Check if dataset already exists, if not create it
    try:
        dataset = client.create_dataset(dataset_name="QA Batch Evaluation Dataset")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Dataset already exists: {str(e)}")
            dataset = client.read_dataset(dataset_name="QA Batch Evaluation Dataset")
        else:
            raise e

    # Create examples in the dataset
    try:
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print("Examples added to dataset.")
    except Exception as e:
        print(f"Could not add examples: {str(e)}. They may already exist.")

    # Define mapper for the batch evaluator - mapping LangSmith data format to OpenJudge format
    mapper = {
        "query": "inputs.question",
        "response": "outputs.answer",
        "reference_response": "reference_outputs.expected_answer",
    }

    # Create an instance of the batch evaluator
    batch_evaluator = LangSmithBatchEvaluator(mapper=mapper)

    # Run evaluation with batch evaluator
    experiment_results = evaluate(
        qa_application,
        data=dataset.name,
        evaluators=[batch_evaluator],  # Single batch evaluator handles multiple graders
        experiment_prefix="open_judge_batch_evaluation",
        description="Evaluating QA application with OpenJudge GradingRunner",
        max_concurrency=4,
    )

    # Process results
    df = experiment_results.to_pandas()
    print(df.head())


def main():
    """
    Main function to run LangSmith integration examples.
    """
    print("LangSmith Integration Cookbook")
    print("=" * 40)

    print("1. Running Individual Grader Example...")
    run_individual_grader_example()
    print()

    print("2. Running Batch Grader Example...")
    run_batch_grader_example()
    print()


if __name__ == "__main__":
    main()
