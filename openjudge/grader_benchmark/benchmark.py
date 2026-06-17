# -*- coding: utf-8 -*-
"""
Base Grader Benchmark for OpenJudge Graders.

Provides the GraderBenchmark class — a reusable framework for running grader
evaluations against benchmark datasets with support for both pairwise
(chosen vs rejected) and pointwise (score vs threshold) modes.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError


@dataclass
class BenchmarkResult:
    """Result of a grader benchmark run."""

    grader_name: str
    model_name: str
    accuracy: float = 0.0
    correct: int = 0
    total: int = 0
    elapsed_seconds: float = 0.0
    status: str = "pending"
    results: List[Dict[str, Any]] = field(default_factory=list)
    error_output: str = ""


class GraderBenchmark:
    """Base benchmark for running grader evaluations.

    Provides reusable evaluation logic including:
    - Dataset loading (local JSON or HuggingFace)
    - Model initialization
    - Pairwise accuracy evaluation (chosen > rejected)
    - Pointwise evaluation (score vs threshold)
    - Result aggregation and formatting

    Subclass and override `extract_inputs()` and `create_grader()`
    to customize per-grader behavior.

    Example:
        >>> benchmark = GraderBenchmark(
        ...     grader_class=ActionLoopDetectionGrader,
        ...     data_file="action_loop_detection.json",
        ...     eval_mode="pairwise",
        ... )
        >>> result = await benchmark.evaluate(model_name="qwen3-max")
        >>> print(f"Accuracy: {result.accuracy:.2%}")
    """

    def __init__(
        self,
        grader_class: Type[BaseGrader],
        data_file: str,
        eval_mode: str = "pairwise",
        expected_accuracy: str = "",
        needs_model: bool = True,
        hf_dataset: str = "agentscope-ai/OpenJudge",
        hf_subdir: str = "",
        grader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize GraderBenchmark.

        Args:
            grader_class: The grader class to evaluate.
            data_file: Filename of the evaluation dataset (JSON).
            eval_mode: "pairwise" (chosen > rejected) or "pointwise" (score vs threshold).
            expected_accuracy: Expected accuracy string for reporting.
            needs_model: Whether this grader requires an LLM model.
            hf_dataset: HuggingFace dataset name for remote loading.
            hf_subdir: Subdirectory within the HF dataset (e.g., "agent/action").
            grader_kwargs: Additional kwargs for grader instantiation.
        """
        self.grader_class = grader_class
        self.data_file = data_file
        self.eval_mode = eval_mode
        self.expected_accuracy = expected_accuracy
        self.needs_model = needs_model
        self.hf_dataset = hf_dataset
        self.hf_subdir = hf_subdir
        self.grader_kwargs = grader_kwargs or {}

    async def load_dataset(self, local_dir: Optional[str] = None) -> List[dict]:
        """Load dataset from local file or HuggingFace.

        Args:
            local_dir: Optional directory to look for the local data file.
                       If not provided, looks next to the calling script.

        Returns:
            List of dataset samples.
        """
        local_file = None
        if local_dir:
            local_file = Path(local_dir) / self.data_file

        if local_file and local_file.exists():
            print(f"Loading from local file: {local_file}")
            with open(local_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Try HuggingFace
        print(f"Loading from HuggingFace: {self.hf_dataset}")
        from datasets import load_dataset

        data_path = f"{self.hf_subdir}/{self.data_file}" if self.hf_subdir else self.data_file
        ds = load_dataset(
            self.hf_dataset,
            data_files=data_path,
            split="train",
        )
        return list(ds)

    def create_grader(self, model=None) -> BaseGrader:
        """Instantiate the grader.

        Override this method for custom grader initialization logic.

        Args:
            model: The chat model instance (for LLM-based graders).

        Returns:
            An instance of the grader class.
        """
        if self.needs_model:
            if model is None:
                raise ValueError(f"Model is required for {self.grader_class.__name__}")
            return self.grader_class(model=model, **self.grader_kwargs)
        return self.grader_class(**self.grader_kwargs)

    def extract_inputs(self, sample: dict, side: str = "chosen") -> Optional[dict]:
        """Extract grader input parameters from a dataset sample.

        Override this method to customize how dataset samples map to
        grader input parameters.

        Args:
            sample: A dataset sample with 'input', 'chosen', 'rejected' fields.
            side: "chosen" or "rejected" — which response to extract.

        Returns:
            Dict of kwargs for the grader's aevaluate(), or None if inapplicable.
        """
        input_data = sample.get("input", {})
        response_data = sample.get(side, {})
        if response_data is None:
            return None

        resp = response_data.get("response", {})
        context = input_data.get("context", {})

        # Default: pass everything from response + context
        kwargs = {}
        kwargs.update(resp)
        if "task_context" in context:
            kwargs["context"] = context["task_context"]
        if "history" in context:
            kwargs["history"] = context["history"]

        return kwargs

    async def evaluate_pairwise(
        self,
        grader: BaseGrader,
        dataset: List[dict],
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Run pairwise evaluation (chosen > rejected).

        Args:
            grader: The grader instance.
            dataset: List of dataset samples.
            verbose: Whether to print per-sample results.

        Returns:
            BenchmarkResult with pairwise accuracy.
        """
        start_time = time.time()
        correct_count = 0
        total_count = 0
        results = []

        for i, sample in enumerate(dataset):
            chosen_inputs = self.extract_inputs(sample, "chosen")
            rejected_inputs = self.extract_inputs(sample, "rejected")

            chosen_score = None
            rejected_score = None

            if chosen_inputs is not None:
                try:
                    result = await grader.aevaluate(**chosen_inputs)
                    if not isinstance(result, GraderError):
                        chosen_score = result.score
                except Exception as e:
                    if verbose:
                        print(f"  [{i + 1}] Error evaluating chosen: {e}")

            if rejected_inputs is not None:
                try:
                    result = await grader.aevaluate(**rejected_inputs)
                    if not isinstance(result, GraderError):
                        rejected_score = result.score
                except Exception as e:
                    if verbose:
                        print(f"  [{i + 1}] Error evaluating rejected: {e}")

            # Determine correctness
            if chosen_score is not None and rejected_score is not None:
                is_correct = chosen_score > rejected_score
            elif chosen_score is not None:
                is_correct = chosen_score >= 0.5
            elif rejected_score is not None:
                is_correct = rejected_score < 0.5
            else:
                continue

            if is_correct:
                correct_count += 1
            total_count += 1

            results.append(
                {
                    "id": sample.get("id", i),
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "is_correct": is_correct,
                }
            )

            if verbose:
                status = "✓" if is_correct else "✗"
                c_str = f"{chosen_score:.2f}" if chosen_score is not None else "N/A"
                r_str = f"{rejected_score:.2f}" if rejected_score is not None else "N/A"
                print(f"  [{i + 1}/{len(dataset)}] {status} chosen={c_str} vs rejected={r_str}")
            elif (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} samples...")

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        elapsed = time.time() - start_time

        return BenchmarkResult(
            grader_name=grader.name,
            model_name="",
            accuracy=accuracy,
            correct=correct_count,
            total=total_count,
            elapsed_seconds=elapsed,
            status="success" if total_count > 0 else "no_samples",
            results=results,
        )

    async def evaluate_pointwise(
        self,
        grader: BaseGrader,
        dataset: List[dict],
        score_threshold: float = 0.5,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Run pointwise evaluation (score vs threshold).

        For each sample, checks if the grader score meets the expected
        threshold. The expected direction (above/below) is determined
        by the sample's "expected" field or defaults to above threshold.

        Args:
            grader: The grader instance.
            dataset: List of dataset samples.
            score_threshold: Threshold for considering a score correct.
            verbose: Whether to print per-sample results.

        Returns:
            BenchmarkResult with pointwise accuracy.
        """
        start_time = time.time()
        correct_count = 0
        total_count = 0
        results = []

        for i, sample in enumerate(dataset):
            # Try extracting from chosen side first, then directly
            inputs = self.extract_inputs(sample, "chosen")
            if inputs is None:
                # Try extracting directly from input
                inputs = sample.get("input", {})

            if not inputs:
                continue

            try:
                result = await grader.aevaluate(**inputs)
                if isinstance(result, GraderError):
                    continue
                score = result.score
            except Exception:
                continue

            # Determine expected direction
            expected = sample.get("expected", "high")
            if expected == "low":
                is_correct = score < score_threshold
            else:
                is_correct = score >= score_threshold

            if is_correct:
                correct_count += 1
            total_count += 1

            results.append(
                {
                    "id": sample.get("id", i),
                    "score": score,
                    "is_correct": is_correct,
                }
            )

            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"  [{i + 1}/{len(dataset)}] {status} score={score:.2f}")

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        elapsed = time.time() - start_time

        return BenchmarkResult(
            grader_name=grader.name,
            model_name="",
            accuracy=accuracy,
            correct=correct_count,
            total=total_count,
            elapsed_seconds=elapsed,
            status="success" if total_count > 0 else "no_samples",
            results=results,
        )

    async def evaluate(
        self,
        model_name: str = "",
        local_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> BenchmarkResult:
        """Run the full evaluation.

        Args:
            model_name: Model name for LLM-based graders.
            local_dir: Directory to look for local data files.
            verbose: Whether to print per-sample results.

        Returns:
            BenchmarkResult with evaluation metrics.
        """
        # Load dataset
        try:
            dataset = await self.load_dataset(local_dir)
        except Exception as e:
            return BenchmarkResult(
                grader_name=self.grader_class.__name__,
                model_name=model_name,
                status=f"load_error: {e}",
            )

        print(f"Loaded {len(dataset)} samples")

        # Initialize model and grader
        model = None
        if self.needs_model:
            from openjudge.models.openai_chat_model import OpenAIChatModel

            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key:
                return BenchmarkResult(
                    grader_name=self.grader_class.__name__,
                    model_name=model_name,
                    status="missing_api_key",
                )

            model = OpenAIChatModel(
                model=model_name,
                api_key=api_key,
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                extra_body={"channel": "openjudge"},
                temperature=0,
                top_p=0.99,
            )

        grader = self.create_grader(model=model)

        # Run evaluation
        print(f"\nEvaluating {self.grader_class.__name__} ({self.eval_mode} mode)...")

        if self.eval_mode == "pairwise":
            result = await self.evaluate_pairwise(grader, dataset, verbose)
        else:
            result = await self.evaluate_pointwise(grader, dataset, verbose)

        result.model_name = model_name

        # Print summary
        print(f"\n{'=' * 60}")
        print("BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print(f"Grader: {self.grader_class.__name__}")
        print(f"Model: {model_name or 'N/A (rule-based)'}")
        print(f"Samples: {result.total}")
        print(f"Correct: {result.correct}")
        print(f"{'Pairwise' if self.eval_mode == 'pairwise' else 'Pointwise'} Accuracy: {result.accuracy:.2%}")
        print(f"{'=' * 60}")

        return result

    def print_results(self, result: BenchmarkResult) -> None:
        """Print detailed results including error cases.

        Args:
            result: The benchmark result to print.
        """
        errors = [r for r in result.results if not r.get("is_correct", False)]
        if errors:
            print(f"\nError cases ({len(errors)}):")
            for r in errors:
                print(f"  ID: {r['id']}", end="")
                if "chosen_score" in r:
                    print(f" - chosen={r['chosen_score']}, rejected={r['rejected_score']}", end="")
                elif "score" in r:
                    print(f" - score={r['score']}", end="")
                print()
