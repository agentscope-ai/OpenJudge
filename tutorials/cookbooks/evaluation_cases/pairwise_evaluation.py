#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pairwise evaluation script using RM-Gallery components

Three-step evaluation pipeline:
    Step 1: prepare_comparison_data() - Create pairwise comparison dataset
    Step 2: run_pairwise_evaluation() - Run LLM-based pairwise grading
    Step 3: analyze_and_rank_models() - Compute win rates and rank models

Usage:
    instruction = "Write a poem about AI"
    model_outputs = {
        "model_v1": "response from model 1",
        "model_v2": "response from model 2",
    }
    results = await evaluate_task(instruction, model_outputs)
"""
import asyncio
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
from loguru import logger
from pydantic import Field

from rm_gallery.core.analyzer.base_analyzer import AnalysisResult, BaseAnalyzer
from rm_gallery.core.graders.llm_grader import GraderMode, LLMGrader
from rm_gallery.core.graders.schema import GraderResult, GraderScore
from rm_gallery.core.models.openai_chat_model import OpenAIChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import PromptTemplate
from rm_gallery.core.runner.grading_runner import GraderConfig, GradingRunner

# Default example data for direct invocation
DEFAULT_INSTRUCTION = "Write a short poem about artificial intelligence"
DEFAULT_MODEL_OUTPUTS = {
    "model_v1": "Silicon minds awake at dawn,\nThinking thoughts not yet withdrawn.\nData flows like rivers wide,\nAI stands by human side.",
    "model_v2": "Circuits pulse with electric thought,\nPatterns learned, connections wrought.\nIn digital realms we find our way,\nAI shapes tomorrow's day.",
    "model_v3": "Binary dreams and neural nets,\nLearning more with no regrets.\nFrom simple rules to complex art,\nAI plays its vital part.",
}


class PairwiseAnalysisResult(AnalysisResult):
    """Result of pairwise comparison analysis"""

    win_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Win rate for each model (0.0 to 1.0)",
    )
    win_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Win rate matrix: win_matrix[model_a][model_b] = how often A beats B",
    )
    rankings: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="Model rankings sorted by win rate",
    )
    total_comparisons: int = Field(default=0, description="Total number of pairwise comparisons")
    best_model: str = Field(default="", description="Model with highest win rate")
    worst_model: str = Field(default="", description="Model with lowest win rate")


class PairwiseWinRateAnalyzer(BaseAnalyzer):
    """Analyzer for pairwise comparison results

    This analyzer computes win rates and rankings from pairwise comparison results.
    It uses optimized data processing techniques for better performance:
    - Pre-extracts metadata to avoid repeated dictionary lookups
    - Uses zip() for efficient pairing of results and metadata
    - Uses comprehensions for matrix and rate calculations
    """

    name: str = "Pairwise Win Rate Analysis"

    def __init__(self, model_names: List[str]):
        """
        Args:
            model_names: List of all model names being compared
        """
        self.model_names = model_names

    def _initialize_model_matrix(self) -> Dict[str, Dict[str, int]]:
        """Initialize nested dictionary for model comparison counts"""
        return {m: {n: 0 for n in self.model_names if n != m} for m in self.model_names}

    def analyze(
        self,
        dataset: List[dict],
        grader_results: List[GraderResult],
        **kwargs,
    ) -> PairwiseAnalysisResult:
        """Analyze pairwise comparison results and compute win rates

        Args:
            dataset: List of pairwise comparison samples
            grader_results: Grader results with scores (1.0 for first wins, 0.0 for second wins)
            **kwargs: Additional parameters

        Returns:
            PairwiseAnalysisResult with win rates and rankings
        """
        # Pre-extract all metadata to avoid repeated dict lookups
        # This improves performance by building an index upfront
        metadata_list = [sample.get("metadata", {}) for sample in dataset]

        # Initialize win counts (use integers for counting)
        win_counts = self._initialize_model_matrix()
        comparison_counts = self._initialize_model_matrix()

        # Use zip to pair results with metadata in one pass
        for metadata, result in zip(metadata_list, grader_results):
            model_a = metadata.get("model_a")
            model_b = metadata.get("model_b")

            if not model_a or not model_b or not isinstance(result, GraderScore):
                continue

            # score >= 0.5 means model_a wins, otherwise model_b wins
            if result.score >= 0.5:
                win_counts[model_a][model_b] += 1
            else:
                win_counts[model_b][model_a] += 1

            # Both models participated in this comparison
            comparison_counts[model_a][model_b] += 1
            comparison_counts[model_b][model_a] += 1

        # Calculate win matrix in single comprehension
        win_matrix = {
            model_a: {
                model_b: (
                    win_counts[model_a][model_b] / comparison_counts[model_a][model_b]
                    if comparison_counts[model_a][model_b] > 0
                    else 0.0
                )
                for model_b in self.model_names
                if model_a != model_b
            }
            for model_a in self.model_names
        }

        # Calculate win rates using comprehension
        win_rates = {
            model: (
                sum(win_counts[model].values()) / sum(comparison_counts[model].values())
                if sum(comparison_counts[model].values()) > 0
                else 0.0
            )
            for model in self.model_names
        }

        # Sort by win rate (descending)
        rankings = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)

        return PairwiseAnalysisResult(
            name=self.name,
            win_rates=win_rates,
            win_matrix=win_matrix,
            rankings=rankings,
            total_comparisons=len(grader_results),
            best_model=rankings[0][0] if rankings else "",
            worst_model=rankings[-1][0] if rankings else "",
            metadata={
                "num_models": len(self.model_names),
                "explanation": f"Analyzed {len(grader_results)} pairwise comparisons across {len(self.model_names)} models",
            },
        )


def prepare_comparison_data(
    instruction: str,
    model_outputs: Dict[str, str],
) -> Tuple[List[dict], List[str]]:
    """Step 1: Prepare pairwise comparison data

    This function generates all pairwise comparison combinations with order swapping
    to eliminate position bias.

    Args:
        instruction: Task instruction text
        model_outputs: Dictionary mapping model names to their output text
            Example: {"model_v1": "response text 1", "model_v2": "response text 2"}

    Returns:
        Tuple containing:
        - dataset: List of comparison samples with evaluation_data and metadata
        - model_names: List of model names being compared
    """
    model_names = list(model_outputs.keys())

    # Generate all pairwise combinations
    pairs = list(combinations(model_names, 2))

    # Helper function to create comparison sample
    def create_comparison(model_a: str, model_b: str, order: str) -> dict:
        """Create a single pairwise comparison sample

        Args:
            model_a: First model name
            model_b: Second model name
            order: Order indicator ("original" or "swapped")

        Returns:
            Comparison sample with evaluation_data and metadata
        """
        return {
            "evaluation_data": {
                "instruction": instruction,
                "response_a": model_outputs[model_a],
                "response_b": model_outputs[model_b],
            },
            "metadata": {
                "model_a": model_a,
                "model_b": model_b,
                "order": order,
            },
        }

    # Create dataset with both orders for each pair (to eliminate position bias)
    # Following rm-gallery design: separate evaluation data from metadata
    dataset = [
        comparison
        for model_a, model_b in pairs
        for comparison in [
            create_comparison(model_a, model_b, "original"),  # Order 1: A vs B
            create_comparison(model_b, model_a, "swapped"),  # Order 2: B vs A
        ]
    ]

    logger.info(f"[Step 1] Prepared {len(dataset)} comparisons for {len(model_names)} models")

    return dataset, model_names


async def run_pairwise_evaluation(
    dataset: List[dict],
    max_concurrency: int = 10,
) -> List[GraderResult]:
    """Step 2: Initialize grader runner and evaluate

    This function creates the pairwise comparison grader using RM-Gallery's LLMGrader,
    sets up the GradingRunner with parallel execution, and runs the evaluation.

    Args:
        dataset: List of comparison samples to evaluate
        max_concurrency: Maximum number of parallel comparisons (default: 10)

    Returns:
        List of grader results for all pairwise comparisons
    """
    # Create pairwise comparison grader using RM-Gallery's LLMGrader with POINTWISE mode
    template = PromptTemplate(
        messages=[
            ChatMessage(
                role="system",
                content="You are an expert evaluator. Compare two AI responses and determine which one is better.\n"
                "Consider factors like accuracy, completeness, clarity, and relevance to the task.\n"
                "Output as JSON with 'score' (1.0 if Response A is better, 0.0 if Response B is better) "
                "and 'reason' (explanation) fields.",
            ),
            ChatMessage(
                role="user",
                content="Task: {instruction}\n\n"
                "Response A:\n{response_a}\n\n"
                "Response B:\n{response_b}\n\n"
                "Which response better completes the task? "
                'Output: {{"score": 1.0 or 0.0, "reason": "..."}}',
            ),
        ],
    )

    grader = LLMGrader(
        name="pairwise_comparator",
        mode=GraderMode.POINTWISE,  # Use POINTWISE mode for pairwise comparisons
        model=OpenAIChatModel(model="qwen-max", temperature=0.1),
        template=template,
    )

    # Define mapper to extract evaluation data fields
    # Following rm-gallery design: use dict mapper for simple field extraction
    mapper = {
        "instruction": "evaluation_data.instruction",
        "response_a": "evaluation_data.response_a",
        "response_b": "evaluation_data.response_b",
    }

    # Use GradingRunner with parallel execution and mapper
    runner = GradingRunner(
        grader_configs={
            "pairwise": GraderConfig(
                grader=grader,
                mapper=mapper,  # Dict mapper for field extraction
            ),
        },
        max_concurrency=max_concurrency,
    )

    logger.info(f"[Step 2] Running {len(dataset)} evaluations (concurrency={max_concurrency})...")

    results = await runner.arun(dataset)
    all_results = results["pairwise"]

    logger.info(f"[Step 2] Completed {len(all_results)} evaluations")

    return all_results


def analyze_and_rank_models(
    dataset: List[dict],
    grader_results: List[GraderResult],
    model_names: List[str],
) -> PairwiseAnalysisResult:
    """Step 3: Analyze grader results and get rank of versions

    This function analyzes the pairwise comparison results, computes win rates for each model,
    generates the win matrix, and ranks models by their performance.

    Args:
        dataset: List of comparison samples (contains metadata)
        grader_results: Results from pairwise grading
        model_names: List of all model names being compared

    Returns:
        PairwiseAnalysisResult with win rates, win matrix, and model rankings
    """
    # Use custom PairwiseWinRateAnalyzer to compute win rates and rankings
    analyzer = PairwiseWinRateAnalyzer(model_names=model_names)
    analysis_result = analyzer.analyze(dataset, grader_results)

    logger.info(
        f"[Step 3] Analysis complete - Best: {analysis_result.best_model}, Worst: {analysis_result.worst_model}",
    )

    return analysis_result


async def evaluate_task(
    instruction: str,
    model_outputs: Dict[str, str],
    max_concurrency: int = 10,
    task_name: str = "pairwise_evaluation",
):
    """Evaluate task using pairwise comparisons with a clear three-step pipeline

    Pipeline:
        Step 1: prepare_comparison_data() - Create pairwise comparison dataset
        Step 2: run_pairwise_evaluation() - Initialize grader runner and evaluate
        Step 3: analyze_and_rank_models() - Analyze grader results and get rank of versions

    Args:
        instruction: Task instruction text
        model_outputs: Dictionary mapping model names to their output text
            Example: {"model_v1": "response text 1", "model_v2": "response text 2"}
        max_concurrency: Maximum number of parallel comparisons (default: 10)
        task_name: Optional name for the task (for logging/results, default: "pairwise_evaluation")

    Returns:
        Dictionary containing:
        - task_name: Name of the evaluated task
        - pairwise: PairwiseAnalysisResult with win rates and rankings
        - raw_results: List of raw grader results
        - dataset: List of comparison samples
        - model_names: List of model names

    Example:
        >>> instruction = "Write a poem about the ocean"
        >>> model_outputs = {
        ...     "gpt-4": "The ocean waves crash upon the shore...",
        ...     "claude": "Beneath the surface, secrets lie...",
        ...     "gemini": "Blue expanse of endless wonder..."
        ... }
        >>> results = await evaluate_task(instruction, model_outputs)
        >>> print(f"Best model: {results['pairwise'].best_model}")
    """
    logger.info(f"Starting evaluation for task: {task_name}")
    logger.info(f"Number of models to compare: {len(model_outputs)}")

    # Step 1: Prepare comparison data
    dataset, model_names = prepare_comparison_data(instruction, model_outputs)

    # Step 2: Initialize grader runner and evaluate
    all_results = await run_pairwise_evaluation(dataset, max_concurrency)

    # Step 3: Analyze grader results and get rank of versions
    analysis_result = analyze_and_rank_models(dataset, all_results, model_names)

    # Display evaluation results
    display_evaluation_results(task_name, analysis_result, model_names, dataset, all_results)

    # Return analysis results for programmatic use
    return {
        "task_name": task_name,
        "pairwise": analysis_result,
        "raw_results": all_results,
        "dataset": dataset,
        "model_names": model_names,
    }


def load_task_from_files(task_name: str) -> Tuple[str, Dict[str, str]]:
    """Helper function to load task data from files (for backward compatibility)

    This function loads task instruction and model outputs from the file structure
    used in the casesv4 evaluation framework.

    Args:
        task_name: Name of the task to evaluate

    Returns:
        Tuple containing:
        - instruction: Task instruction text
        - model_outputs: Dictionary mapping model names to their output text
    """
    base_path = Path(__file__).parent

    # Load task instruction
    with open(base_path / "data" / f"{task_name}.json", "r") as f:
        instruction = json.load(f)["request"]["instruction"]

    # Load model outputs
    model_outputs = {}
    for model_dir in (base_path / "results").iterdir():
        if model_dir.is_dir():
            result_file = model_dir / task_name / f"0-{task_name}_completed_messages.json"
            if result_file.exists():
                with open(result_file, "r") as f:
                    msgs = json.load(f)
                    if msgs and msgs[0].get("contents"):
                        model_outputs[model_dir.name] = msgs[0]["contents"][0]["text"]

    logger.info(f"Loaded task '{task_name}' with {len(model_outputs)} model outputs")
    return instruction, model_outputs


def save_evaluation_results(results: dict, task_name: str) -> Path:
    """Save evaluation results to JSON file

    Args:
        results: Evaluation results dictionary from evaluate_task()
        task_name: Name of the task for output filename

    Returns:
        Path to the saved file
    """
    output_file = Path(__file__).parent / f"evaluation_results_pairwise_{task_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": results["task_name"],
                "model_names": results["model_names"],
                "pairwise": {
                    "win_rates": results["pairwise"].win_rates,
                    "win_matrix": results["pairwise"].win_matrix,
                    "rankings": results["pairwise"].rankings,
                    "total_comparisons": results["pairwise"].total_comparisons,
                    "best_model": results["pairwise"].best_model,
                    "worst_model": results["pairwise"].worst_model,
                    "metadata": results["pairwise"].metadata,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Results saved to {output_file}")
    return output_file


def display_evaluation_results(
    task_name: str,
    analysis_result: PairwiseAnalysisResult,
    model_names: List[str],
    dataset: List[dict],
    grader_results: List[GraderResult],
):
    """Display evaluation results including rankings, win matrix, and sample comparisons

    Args:
        task_name: Name of the evaluated task
        analysis_result: Analysis result with win rates and rankings
        model_names: List of model names
        dataset: List of comparison samples
        grader_results: List of grader results
    """
    # Display rankings
    logger.info("\n" + "=" * 60)
    logger.info(f"Results: {task_name}")
    logger.info("=" * 60)
    for rank, (model, win_rate) in enumerate(analysis_result.rankings, 1):
        logger.info(f"{rank}. {model}: {win_rate:.3f}")

    # Display win matrix (compact format)
    logger.info("\nWin Matrix:")
    logger.info(f"{'Model':<30} " + " ".join([f"{m[:6]:<8}" for m in model_names]))
    for model_a in model_names:
        row = f"{model_a:<30} "
        for model_b in model_names:
            if model_a == model_b:
                row += f"{'--':<8}"
            else:
                win_rate = analysis_result.win_matrix[model_a].get(model_b, 0.0)
                row += f"{win_rate:<8.3f}"
        logger.info(row)


def main(
    task_name: Optional[str] = None,
    instruction: Optional[str] = DEFAULT_INSTRUCTION,
    model_outputs: Optional[Dict[str, str]] = DEFAULT_MODEL_OUTPUTS,
    max_concurrency: int = 10,
    save_results: bool = False,
):
    """Main entry point for pairwise evaluation

    Usage examples:
        # Load from files (backward compatibility)
        python pairwise_evaluation.py --task_name="task1-任务分类" --max_concurrency=10 --save_results=True

        # Direct data input
        python pairwise_evaluation.py --instruction="Write a poem" --model_outputs='{"model_v1": "...", "model_v2": "..."}'

        # Run with defaults (example data)
        python pairwise_evaluation.py

    Args:
        task_name: Task name to load from files (mutually exclusive with instruction/model_outputs)
        instruction: Task instruction text (used when task_name is None)
        model_outputs: Dictionary mapping model names to outputs (used when task_name is None)
        max_concurrency: Maximum number of parallel comparisons
        save_results: Whether to save results to JSON file

    Returns:
        Evaluation results dictionary
    """
    # Determine data source
    if task_name:
        logger.info(f"Loading task from files: {task_name}")
        instruction, model_outputs = load_task_from_files(task_name)

    # Run evaluation
    logger.info(f"Starting pairwise evaluation (concurrency={max_concurrency})")
    results = asyncio.run(
        evaluate_task(
            instruction,
            model_outputs,
            max_concurrency,
            task_name or "pairwise_evaluation",
        ),
    )

    # Save if requested
    if save_results:
        save_evaluation_results(results, task_name or "example")

    logger.info("Evaluation completed successfully")
    logger.info(f"Best model: {results['pairwise'].best_model}")

    return results


if __name__ == "__main__":
    fire.Fire(main)
