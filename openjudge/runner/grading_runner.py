# -*- coding: utf-8 -*-
"""Grading runner for executing evaluators on datasets and computing results.

This module provides functionality to run multiple evaluators on datasets and
collect their results. It supports concurrent execution of evaluators and
organizes results by sample for further analysis.

Classes:
    RunnerResult: Result container for grading runs.
    GradingRunner: Main runner class for executing evaluators.
"""

import asyncio
from typing import Any, Callable, Dict, List, Union

from loguru import logger
from tqdm.asyncio import tqdm_asyncio

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderError, GraderResult
from openjudge.runner.aggregator.base_aggregator import BaseAggregator
from openjudge.runner.base_runner import BaseRunner, RunnerResult
from openjudge.runner.controller.base import BaseController
from openjudge.runner.controller.local_controller import LocalController


class GradingRunner(BaseRunner):
    """Runner for executing evaluators on datasets concurrently.

    This class provides functionality to run multiple evaluators on datasets with
    concurrent execution. It organizes results by grader, making it easy to analyze
    how each grader scored all samples.

    The runner supports data mapping to transform input data before passing it to
    evaluators through the mapper functionality integrated directly into the grader,
    and concurrency control to limit the number of simultaneous operations.

    Attributes:
        graders (Dict[str, BaseGrader]): Graders to run.
        max_concurrency (int): Maximum number of concurrent operations.

    Example:
        >>> # Simple usage with just graders
        >>> graders = {
        ...     "accuracy": AccuracyGrader()
        ... }
        >>> runner = GradingRunner(graders=graders, max_concurrency=10)
        >>>
        >>> # Usage with data mappers integrated directly in graders
        >>> grader_with_mapper = AccuracyGrader(mapper={"q": "query", "a": "response"})
        >>> runner = GradingRunner(graders={"accuracy": grader_with_mapper}, max_concurrency=5)
        >>>
        >>> # Run evaluation on data
        >>> data = [{"query": "What is 2+2?", "response": "4"}]
        >>> result = await runner.arun(data)
        >>>
        >>> # Access results
        >>> for grader_name, grader_results in result.items():
        ...     print(f"{grader_name} results:")
        ...     for i, grader_result in enumerate(grader_results):
        ...         print(f"  Sample {i}: {grader_result}")
    """

    def __init__(
        self,
        graders: Dict[str, "BaseGrader"],
        max_concurrency: int = 32,
        aggregators: Union[BaseAggregator, Callable, List[Union[BaseAggregator, Callable]], None] = None,
        show_progress: bool = True,
        controller: BaseController | None = None,
    ) -> None:
        """Initialize the grading runner.

        Args:
            graders: Dictionary of graders where keys are grader names
                and values are BaseGrader instances. Each grader can have an integrated
                mapper for transforming input data.
            max_concurrency: Maximum number of concurrent operations. Defaults to 32.
                Controls how many evaluations can run simultaneously to manage resource usage.
            aggregators: Optional aggregator or list of aggregators to combine results
                from multiple graders.
            show_progress: Whether to display a progress bar during execution. Defaults to True.
            controller: Optional execution controller to manage task execution.
                       Defaults to LocalController if not provided.
        """
        self.graders = graders
        self.max_concurrency = max_concurrency
        self.show_progress = show_progress
        self.controller = controller or LocalController(max_concurrency)

        # Handle aggregators
        if not aggregators:
            self.aggregators = []
        elif isinstance(aggregators, BaseAggregator):
            self.aggregators = [aggregators]
        else:
            self.aggregators = aggregators

    @classmethod
    async def _arun(
        cls,
        data: dict,
        grader: BaseGrader,
        controller: BaseController,
    ) -> GraderResult:
        """Run a single evaluation asynchronously.

        This internal method runs a single evaluation using the grader's built-in mapper.
        It handles exceptions that may occur during evaluation and wraps them in a GraderError.

        Args:
            data: Input data for the evaluation. This is typically a dictionary containing
                the fields needed by the grader (e.g., 'query', 'answer', 'context').
            grader: Grader instance to use for the evaluation. Must be an instance of
                a class that inherits from BaseGrader.
            controller: Execution controller to manage the execution of the task

        Returns:
            GraderResult: The result of the evaluation from the grader. This can be:
                - GraderScore: For pointwise graders, contains score and explanation
                - GraderRank: For listwise graders, contains ranking and explanation
                - GraderError: If an exception occurred during evaluation

        Example:
            >>> # With a simple data transformation
            >>> data = {"question": "What is 2+2?", "response": "4"}
            >>> result = await GradingRunner._arun(data, AccuracyGrader(), controller)
        """

        async def _evaluate(data) -> GraderResult:
            try:
                # The grader itself handles the mapping internally
                return await grader.aevaluate(controller=controller, **data)
            except Exception as e:
                error_msg = f"Error in {grader.name} during evaluation: {str(e)}"
                logger.error(error_msg)
                return GraderError(
                    name=grader.name,
                    reason=f"Error in {grader.name} during evaluation",
                    error=error_msg,
                )

        # Execute the evaluation using the controller
        return await _evaluate(data)

    async def arun(
        self,
        dataset: List[dict],
        *args: Any,
        **kwargs: Any,
    ) -> RunnerResult:
        """Run evaluators on the provided data concurrently.

        This method executes all configured evaluators on the provided data samples
        concurrently. Results are organized by grader, with each grader containing
        results from all samples.

        Args:
            dataset: List of data samples to evaluate. Each sample is a dictionary
                containing the fields needed by the graders. For example:
                [
                    {"query": "What is the capital of France?", "answer": "Paris"},
                    {"query": "What is 2+2?", "answer": "4"}
                ]
            *args: Additional positional arguments (not used in current implementation).
            **kwargs: Additional keyword arguments (not used in current implementation).

        Returns:
            RunnerResult: Results of the evaluation run. This is a dictionary where each key
            is a grader name and each value is a list of results from that grader for all samples.

            The structure is:
            {
                "grader1_name": [           # Results from grader1
                    result1_for_sample1,
                    result1_for_sample2
                ],
                "grader2_name": [           # Results from grader2
                    result2_for_sample1,
                    result2_for_sample2
                ]
            }

        Example:
            >>> # Define graders
            >>> accuracy_grader = AccuracyGrader()
            >>> relevance_grader = RelevanceGrader()
            >>>
            >>> # Create runner
            >>> runner = GradingRunner(graders={
            ...     "accuracy": accuracy_grader,
            ...     "relevance": relevance_grader
            ... }, max_concurrency=10)
            >>>
            >>> # Data to evaluate
            >>> dataset = [
            ...     {"query": "What is the capital of France?", "answer": "Paris"},
            ...     {"query": "What is 2+2?", "answer": "4"}
            ... ]
            >>>
            >>> # Run evaluation
            >>> results = await runner.arun(dataset)
            >>>
            >>> # Process results
            >>> for grader_name, grader_results in results.items():
            ...     print(f"Results for {grader_name}:")
            ...     for i, result in enumerate(grader_results):
            ...         if hasattr(result, 'score'):
            ...             print(f"  Sample {i}: {result.score}")
            ...         elif hasattr(result, 'rank'):
            ...             print(f"  Sample {i}: {result.rank}")
            ...         else:
            ...             print(f"  Sample {i}: Error - {result.error}")
        """
        # Create a dictionary to store result lists for each grader
        grader_results: RunnerResult = {name: [] for name in self.graders.keys()}

        # Create coroutines for all evaluators and all samples
        all_coroutines = []
        coroutine_info = []  # Track (grader_name, sample_index) for each coroutine

        # Use the controller from self
        controller = self.controller

        # Execute controller lifecycle
        for name, grader in self.graders.items():
            assert grader is not None

            # Create coroutines for the current evaluator on all samples
            for i, case in enumerate(dataset):
                all_coroutines.append(
                    self._arun(data=case, grader=grader, controller=controller),
                )
                coroutine_info.append(
                    (name, i),
                )  # Record grader name and sample index

        # Execute all evaluator-sample coroutines concurrently
        if self.show_progress:
            all_results = await tqdm_asyncio.gather(
                *all_coroutines,
                desc="Evaluating a dataset",
                total=len(all_coroutines),
            )
        else:
            all_results = await asyncio.gather(*all_coroutines)

        # Initialize lists for all graders
        for name in self.graders.keys():
            grader_results[name] = [None] * len(dataset)

        # Organize results by grader
        for (grader_name, sample_index), result in zip(coroutine_info, all_results):
            grader_results[grader_name][sample_index] = result

        # Aggregate results
        if self.aggregators:
            for aggregator in self.aggregators:
                aggregator_name = aggregator.__name__
                grader_results[aggregator_name] = [None] * len(dataset)
                for i in range(len(dataset)):
                    grader_results[aggregator_name][i] = aggregator(
                        {grader_name: grader_results[grader_name][i] for grader_name in self.graders.keys()},
                    )
        return grader_results

    async def arun_multiple_datasets(
        self,
        datasets: List[List[dict]],
        *args: Any,
        **kwargs: Any,
    ) -> List[RunnerResult]:
        """Run evaluators on multiple datasets concurrently.

        All datasets share the same concurrency pool (max_concurrency) through
        the singleton ConcurrencyManager. Each dataset is processed by calling
        arun(), and results are returned as a list in the same order as the input datasets.

        Args:
            datasets: List of datasets, where each dataset is a list of data samples.
                Each sample is a dictionary containing the fields needed by the graders.
                For example:
                [
                    [{"query": "Q1", "answer": "A1"}, {"query": "Q2", "answer": "A2"}],
                    [{"query": "Q3", "answer": "A3"}]
                ]
            *args: Additional positional arguments passed to arun.
            **kwargs: Additional keyword arguments passed to arun.

        Returns:
            List[RunnerResult]: A list of RunnerResults, one for each dataset in the same order
            as the input datasets. Each RunnerResult is a dictionary mapping grader names to
            their results for all samples in that dataset.

            The structure is:
            [
                {  # Results for dataset 0
                    "grader1_name": [result1, result2, ...],
                    "grader2_name": [result1, result2, ...]
                },
                {  # Results for dataset 1
                    "grader1_name": [result1, result2, ...],
                    "grader2_name": [result1, result2, ...]
                },
                ...
            ]

        Example:
            >>> # Define graders
            >>> accuracy_grader = AccuracyGrader()
            >>> relevance_grader = RelevanceGrader()
            >>>
            >>> # Create grader configs
            >>> graders = {
            ...     "accuracy": accuracy_grader,
            ...     "relevance": relevance_grader
            ... }
            >>>
            >>> # Create runner
            >>> runner = GradingRunner(graders, max_concurrency=10)
            >>>
            >>> # Multiple datasets to evaluate
            >>> datasets = [
            ...     [  # dataset 0: training set
            ...         {"query": "What is the capital of France?", "answer": "Paris"},
            ...         {"query": "What is 2+2?", "answer": "4"}
            ...     ],
            ...     [  # dataset 1: test set
            ...         {"query": "What is the capital of Spain?", "answer": "Madrid"}
            ...     ]
            ... ]
            >>>
            >>> # Run batch evaluation
            >>> results = await runner.arun_multiple_datasets(datasets)
            >>>
            >>> # Access results by index
            >>> train_results = results[0]
            >>> test_results = results[1]
            >>>
            >>> # Process results
            >>> for i, dataset_results in enumerate(results):
            ...     print(f"Results for dataset {i}:")
            ...     for grader_name, grader_results in dataset_results.items():
            ...         print(f"  {grader_name}:")
            ...         for j, result in enumerate(grader_results):
            ...             if hasattr(result, 'score'):
            ...                 print(f"    Sample {j}: {result.score}")

        Note:
            - All datasets share the same ConcurrencyManager singleton, ensuring that
              the total concurrent operations across all datasets respect max_concurrency.
            - Progress bar shows dataset-level progress (e.g., "3/5 datasets completed").
            - Each dataset maintains the order of samples and graders as specified.
            - When batch processing, individual arun() progress bars are disabled to avoid
              display conflicts with the batch-level progress bar.
        """
        # Temporarily disable show_progress for individual arun calls to avoid progress bar conflicts
        original_show_progress = self.show_progress
        self.show_progress = False

        try:
            # Create tasks for each dataset
            tasks = [self.arun(dataset, *args, **kwargs) for dataset in datasets]

            # Execute all dataset tasks concurrently with progress bar
            if original_show_progress:
                all_results = await tqdm_asyncio.gather(
                    *tasks,
                    desc=f"Evaluating {len(tasks)} datasets",
                    total=len(tasks),
                )
            else:
                all_results = await asyncio.gather(*tasks)

            # Return results as a list
            return list(all_results)
        finally:
            # Restore original show_progress setting
            self.show_progress = original_show_progress
