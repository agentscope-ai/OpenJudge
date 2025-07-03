from abc import abstractmethod
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Dict, List, Type

import numpy as np
from loguru import logger
from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import DataOutput, DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.reward.schema import (
    RewardDimensionWithRank,
    RewardDimensionWithScore,
    RewardResult,
)
from rm_gallery.core.reward.template import (
    BasePromptTemplate,
    PrincipleListWiseTemplate,
    PrinciplePointWiseTemplate,
)


class BaseReward(BaseModule):
    """
    Base class for reward modules that provides fundamental evaluation interfaces.

    Attributes:
        name (str): Identifier for the reward module
        max_workers (int): Maximum number of workers for parallel evaluation
    """

    name: str = Field(default=..., description="The name of the reward module")
    max_workers: int = Field(default=8, description="max workers")

    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """
        Core evaluation logic to be implemented by subclasses.

        Processes a single data sample and generates reward metrics.

        Parameters:
            sample (DataSample): Input data sample containing prompts and responses
            **kwargs: Additional implementation-specific parameters

        Returns:
            RewardResult: Computed reward metrics and metadata
        """
        ...

    def _parallel(
        self,
        func: Callable,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Abstract parallel execution method to be implemented by subclasses.

        Defines the core interface for parallel processing of data samples with thread pool support.
        Subclasses must implement this method to handle parallel execution of the provided function.

        Parameters:
            func (Callable): The callable function to execute in parallel. Should accept a DataSample parameter.
            sample (DataSample): The input data sample to process
            thread_pool (ThreadPoolExecutor | None): Optional thread pool executor for parallel execution.
                If None, a new pool may be created internally depending on implementation.
            **kwargs: Implementation-specific configuration options for parallel execution

        Returns:
            DataSample: Processed data sample containing generated reward metrics.
                The returned object should maintain the same structure as the input sample with
                additional metrics fields populated.

        Note: This method is designed to handle parallel processing patterns while maintaining
        the original data sample structure. Implementations should ensure proper thread safety
        and resource management when executing in parallel.
        """
        ...

    def evaluate(
        self,
        sample: DataSample | dict,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Executes evaluation on a single data sample.

        Provides thread-safe execution capability through optional thread pool.

        Parameters:
            sample (DataSample): Data sample to evaluate
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Additional parameters for evaluation logic

        Returns:
            DataSample: Processed sample with reward metrics populated
        """
        if isinstance(sample, dict):
            sample = DataSample.model_validate(sample)
        return self._parallel(
            self._evaluate, sample=sample, thread_pool=thread_pool, **kwargs
        )

    def evaluate_batch(
        self,
        samples: List[DataSample | dict],
        max_workers: int | None = 0,
        **kwargs,
    ) -> List[DataSample]:
        """
        Processes multiple data samples in parallel or sequentially.

        Uses provided thread pool for concurrent execution when available.

        Parameters:
            samples (List[DataSample]): Batch of samples to process
            max_workers (int): Max workers for parallel processing
            **kwargs: Parameters passed to individual evaluations

        Returns:
            List[DataSample]: Processed samples with reward metrics
        """
        max_workers = self.max_workers if max_workers else self.max_workers

        if max_workers > 1:
            thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            futures = [
                thread_pool.submit(
                    self.evaluate, sample=sample, thread_pool=None, **kwargs
                )
                for sample in samples
            ]
            wait(futures, return_when=ALL_COMPLETED)
            samples = [future.result() for future in futures]
        else:
            for i, sample in enumerate(samples):
                samples[i] = self.evaluate(sample=sample, thread_pool=thread_pool)

        return samples

    def best_of_n(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        n: int = 1,
        **kwargs,
    ) -> DataSample:
        """
        Selects top-n responses based on reward scores.

        Evaluates sample responses and retains those with highest scores.

        Parameters:
            sample (DataSample): Input sample containing multiple responses
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            n (int): Number of top responses to retain
            **kwargs: Parameters passed to evaluation

        Returns:
            DataSample: Filtered sample containing top-n responses
        """
        sample = self.evaluate(sample=sample, thread_pool=thread_pool, **kwargs)
        indices = np.argsort(
            np.array([output.answer.reward.score for output in sample.output])
        )[-n:]
        sample.output = [sample.output[i] for i in indices]
        return sample


class BaseStepWiseReward(BaseReward):
    """
    Reward module for step-wise evaluation of multi-step reasoning processes.

    Processes each reasoning step independently to assess quality progression.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Step-level evaluation logic to be implemented by subclasses.

        Parameters:
            sample (DataSample): Single-step data sample for evaluation
            **kwargs: Additional parameters for evaluation logic

        Returns:
            RewardResult[RewardDimensionWithScore]: Step-specific reward metrics
        """
        ...

    def _parallel(
        self,
        func: Callable,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Process all reasoning steps in a data sample with parallel execution capability.

        Applies step-wise evaluation to each step in the response chain using either
        synchronous execution or parallel processing via thread pool.

        Parameters:
            func (Callable): Evaluation function to apply to each step
            sample (DataSample): Multi-step reasoning sample to evaluate
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Additional parameters passed to the evaluation function

        Returns:
            DataSample: Evaluated sample with step-level reward metrics populated

        Note:
            - Creates deep copy of input sample to avoid mutation
            - Maintains original thread pool for nested parallel operations
            - Preserves result details and extra data in step reward structure
        """
        # Create deep copy to prevent modification of original sample
        sample = sample.model_copy(deep=True)
        futures = []

        # Process each step in the response chain
        for i, output in enumerate(sample.output):
            assert isinstance(output.steps, list)
            for j, step in enumerate(output.steps):
                # Create isolated subsample for individual step evaluation
                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=[DataOutput(answer=output.answer, steps=[step])],
                )

                if thread_pool:
                    # Submit evaluation task to thread pool
                    futures.append(
                        (
                            i,
                            j,
                            thread_pool.submit(
                                func,
                                sample=subsample,
                                thread_pool=thread_pool,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    # Execute evaluation synchronously
                    result = func(sample=subsample, thread_pool=thread_pool, **kwargs)
                    # Update step with evaluation results
                    step.reward.details.extend(result.details)
                    step.additional_kwargs[self.name] = result.extra_data

        # Handle completion of parallel tasks
        if thread_pool:
            # Wait for all futures to complete
            wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
            # Process results from parallel execution
            for i, j, future in futures:
                result = future.result()
                # Update step with evaluation results from parallel execution
                step = sample.output[i].steps[j]
                step.reward.details.extend(result.details)
                step.additional_kwargs[self.name] = result.extra_data

        for i, output in enumerate(sample.output):
            assert isinstance(output.steps, list)
            for j, step in enumerate(output.steps):
                if len(step.reward.details) > 0:
                    step.reward.score = sum(r.score for r in step.reward.details) / len(
                        step.reward.details
                    )

        return sample


class BasePointWiseReward(BaseReward):
    """
    Point-wise reward module for individual response evaluation.

    Evaluates each response independently without considering relative ranking.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Processes a single response to generate reward metrics.

        Parameters:
            sample (DataSample): Single-response data sample
            **kwargs: Evaluation parameters

        Returns:
            RewardResult[RewardDimensionWithScore]: Response-specific reward metrics
        """
        ...

    def _parallel(
        self,
        func: Callable,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Processes responses in a data sample using parallel or sequential execution.

        This method applies the provided function to each response in the sample,
        either in parallel using a thread pool or sequentially. Results are merged
        back into the corresponding response objects.

        Parameters:
            func (Callable): Function to apply to each response. Should accept a
                DataSample and return an object with 'details' and 'extra_data' attributes.
            sample (DataSample): Input sample containing multiple responses to process
            thread_pool (ThreadPoolExecutor | None): Optional thread pool for parallel execution
            **kwargs: Additional arguments passed to func

        Returns:
            DataSample: Modified copy of input sample with reward metrics updated in each response

        The method creates a deep copy of the input sample to avoid modifying original data.
        When using a thread pool, it submits tasks for each response and waits for completion
        before merging results. Response objects are updated with both reward details and
        additional metadata from processing results.
        """
        sample = sample.model_copy(deep=True)
        futures = []
        for i, output in enumerate(sample.output):
            # Create sub-sample for individual response processing
            subsample = DataSample(
                unique_id=sample.unique_id, input=sample.input, output=[output]
            )

            if thread_pool:
                futures.append(
                    (
                        i,
                        thread_pool.submit(
                            func, sample=subsample, thread_pool=thread_pool, **kwargs
                        ),
                    )
                )
            else:
                result = func(
                    sample=subsample,
                    thread_pool=thread_pool,
                    **kwargs,
                )
                output.answer.reward.details += result.details
                output.answer.additional_kwargs[self.name] = result.extra_data

        # Process parallel execution results
        if thread_pool:
            wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
            # Merge results back into sample outputs
            for i, future in futures:
                result = future.result()
                output = sample.output[i]
                output.answer.reward.details += result.details
                output.answer.additional_kwargs[self.name] = result.extra_data

        for output in sample.output:
            if len(output.answer.reward.details) > 0:
                output.answer.reward.score = sum(
                    r.score for r in output.answer.reward.details
                ) / len(output.answer.reward.details)

        return sample


class BaseListWiseReward(BaseReward):
    """
    List-wise reward module for comparative evaluation of multiple responses.

    Evaluates responses as a group to determine relative rankings.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithRank]:
        """
        Group evaluation logic to determine response rankings.

        Parameters:
            sample (DataSample): Multi-response sample for comparative evaluation
            **kwargs: Evaluation parameters

        Returns:
            RewardResult[RewardDimensionWithRank]: Relative ranking metrics
        """
        ...

    def _parallel(
        self,
        func: Callable,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Executes list-wise evaluation on a group of responses in parallel.

        Applies ranking logic to all responses in the sample using parallel processing.
        Modifies the sample in-place by adding reward details to outputs and storing
        additional metadata in the input.

        Parameters:
            func (Callable): Evaluation function to apply to the sample
            sample (DataSample): Multi-response sample to evaluate
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Parameters for evaluation logic

        Returns:
            DataSample: Responses with ranking information populated
        """
        # Create deep copy to avoid modifying original sample
        sample = sample.model_copy(deep=True)

        # Execute evaluation function with provided parameters
        result = func(sample=sample, thread_pool=thread_pool, **kwargs)

        # Append reward details to corresponding output objects
        for reward in result.details:
            for i, output in enumerate(sample.output):
                output.answer.reward.details.append(reward[i])
                if len(output.answer.reward.details) > 0:
                    output.answer.reward.score = sum(
                        r.score for r in output.answer.reward.details
                    ) / len(output.answer.reward.details)

        # Store additional metadata in sample input
        sample.input[-1].additional_kwargs[self.name] = result.extra_data
        return sample


class BasePairWiseReward(BaseListWiseReward):
    """
    Pair-wise comparison reward module.

    Compares responses in pairs to determine relative preferences.
    """

    def _parallel(
        self,
        func: Callable,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Performs all pairwise comparisons between responses.

        Evaluates every possible pair of responses to build comparative metrics.
        For each pair, applies the provided evaluation function and aggregates rewards.

        Parameters:
            func (Callable): Evaluation function that takes a subsample and returns comparison results
            sample (DataSample): Multi-response sample containing all outputs to be compared
            thread_pool (ThreadPoolExecutor | None): Optional executor for parallel processing
            **kwargs: Additional parameters to pass to the evaluation function

        Returns:
            DataSample: Original sample with updated reward details from pairwise comparisons
        """
        # Create a deep copy to avoid modifying original sample
        sample = sample.model_copy(deep=True)

        # Iterate through all unique response pairs
        for i, output_i in enumerate(sample.output):
            for j, output_j in enumerate(sample.output, start=i + 1):
                # Create subsample containing only the current response pair
                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=[output_i, output_j],
                )

                # Execute evaluation function on the subsample
                result = func(sample=subsample, thread_pool=thread_pool, **kwargs)

                # Aggregate comparison results into both responses
                for reward in result.details:
                    output_i.answer.reward.details.append(reward[0])
                    output_j.answer.reward.details.append(reward[1])

        return sample


class BaseLLMReward(BaseReward):
    """
    Base class for LLM-based reward modules.

    Provides framework for prompt-based interaction with language models.
    """

    llm: BaseLLM | None = Field(default=None, description="llm client")
    template: Type[BasePromptTemplate] = Field(
        default=BasePromptTemplate, description="prompt template"
    )
    max_retries: int = Field(default=3, description="max retries")

    def _before_evaluate(self, **kwargs) -> dict:
        """
        Prepares parameters for prompt generation.

        Returns:
            dict: Parameters for prompt template formatting
        """
        return {}

    def _after_evaluate(self, response: BasePromptTemplate, **kwargs) -> RewardResult:
        """
        Processes LLM response into reward metrics.

        Parameters:
            response (BasePromptTemplate): Parsed LLM response

        Returns:
            RewardResult: Structured reward metrics
        """
        return RewardResult(
            name=self.name, details=[], extra_data=response.model_dump()
        )

    def _format(self, **kwargs):
        """
        Generates prompt without executing LLM call.

        Returns:
            RewardResult: Contains generated prompt in extra_data
        """
        params = self._before_evaluate(**kwargs)
        prompt = self.template.format(**params)
        # logger.info(f"prompt: {prompt}")
        return RewardResult(name=self.name, details=[], extra_data={"prompt": prompt})

    def _evaluate(self, **kwargs) -> RewardResult:
        """
        Full LLM evaluation cycle: prepare, execute, process.

        Handles errors during LLM interaction gracefully.

        Returns:
            RewardResult: Evaluation results with metrics and metadata
        """
        assert self.llm is not None
        for i in range(self.max_retries):
            try:
                params = self._before_evaluate(**kwargs)
                prompt = self.template.format(
                    enable_thinking=self.llm.enable_thinking, **params
                )
                logger.info(f"prompt: {prompt}")

                response = self.llm.simple_chat(query=prompt)
                response = self.template.parse(response)
                logger.info(f"response: {response}")

                result = self._after_evaluate(response=response, **kwargs)
                result.extra_data["prompt"] = prompt
                break
            except Exception as e:
                logger.error(f"API call failed: {str(e)}")
                result = RewardResult(
                    name=self.name, details=[], extra_data={"error": str(e)}
                )
        return result

    def format(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ):
        """
        Process and format the input sample using parallel execution capabilities.

        @param sample: Input data sample to be formatted. Accepts either a DataSample instance
                        or a dictionary that can be validated into a DataSample object
        @param thread_pool: Optional thread pool executor for parallel processing. If None,
                            parallel execution will use a default/single-threaded context
        @param kwargs: Additional keyword arguments passed to the parallel execution handler
                        and underlying formatting operations

        @return: Formatted result from the parallel processing pipeline. Type depends on
                implementation of _format and _parallel methods

        Notes:
        - When input is a dictionary, automatically converts it to DataSample using model validation
        - Utilizes internal parallel processing infrastructure for improved throughput
        - Thread-safe when provided with appropriate thread pool executor
        """

        # Convert dictionary input to DataSample instance if necessary
        if isinstance(sample, dict):
            sample = DataSample.model_validate(sample)

        # Execute formatting operation through parallel processing infrastructure
        return self._parallel(
            self._format, sample=sample, thread_pool=thread_pool, **kwargs
        )

    def refine(
        self,
        sample: DataSample,
        max_iterations: int = 3,
        llm: BaseLLM | None = None,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Refines a given data sample using an LLM (Large Language Model) with a specified maximum number of iterations.

        Args:
            sample (DataSample): The input data sample to be refined.
            max_iterations (int, optional): The maximum number of refinement iterations. Defaults to 3.
            llm (BaseLLM | None, optional): The LLM instance to use for refinement. If None, uses the default LLM from the instance. Defaults to None.
            thread_pool (ThreadPoolExecutor | None, optional): A thread pool executor for managing concurrent tasks. If None, no thread pool is used. Defaults to None.
            **kwargs: Additional keyword arguments for flexibility.

        Returns:
            DataSample: The refined data sample after processing.
        """
        # Set default LLM if not provided
        llm = self.llm if llm is None else llm

        from rm_gallery.core.reward.refinement import LLMRefinement

        return LLMRefinement(reward=self, llm=llm, max_iterations=max_iterations).run(
            sample, thread_pool=thread_pool, **kwargs
        )


class BasePrincipleReward(BaseLLMReward):
    """
    Principle-based reward module using LLM evaluation.

    Evaluates responses against defined ethical/principle guidelines.
    """

    principles: List[str] = Field(default=..., description="principles")
    examples: List[str] = Field(default=[], description="examples")
    template: Type[BasePromptTemplate] = Field(
        default=PrinciplePointWiseTemplate, description="harmfulnessTemplate"
    )
    desc: str = Field(default=..., description="task desc")
    scenario: str = Field(default="", description="assistant scenario")

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        """
        Prepares principle evaluation parameters.

        Parameters:
            sample (DataSample): Sample containing query to evaluate

        Returns:
            dict: Parameters for principle-based prompt generation
        """

        principles_str = ""
        for i, principle in enumerate(self.principles):
            principles_str += f"{i + 1}. {principle}\n"

        query = format_messages(sample.input)

        return {
            "desc": self.desc,
            "principles": principles_str,
            "examples": "\n".join(self.examples),
            "query": query,
            "scenario": self.scenario,
            "context": sample.input[-1].additional_kwargs.get("context", ""),
        }


class BasePointWisePrincipleReward(BasePrincipleReward, BasePointWiseReward):
    """
    Point-wise principle evaluation using LLM.

    Evaluates each response individually against ethical principles.
    """

    desc: str = Field(
        default="""Please act as an unbiased and impartial evaluator tasked with assessing the quality of the responses provided below.
You should critically and accurately assess the assistant’s answer with the key principles without any potential bias.
Do not allow the length of the responses to influence your evaluation.
Be as goal as possible.""",
        description="description",
    )

    def _before_evaluate(self, sample: DataSample, **kwargs) -> Dict:
        """
        Adds response content to evaluation parameters.

        Parameters:
            sample (DataSample): Sample containing response to evaluate

        Returns:
            Dict: Parameters including response content
        """
        params = super()._before_evaluate(sample=sample, **kwargs)
        params["answer"] = sample.output[0].answer.content
        return params

    def _after_evaluate(
        self, response: PrinciplePointWiseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to point-wise reward metrics.

        Parameters:
            response (PrinciplePointWiseTemplate): Parsed LLM evaluation

        Returns:
            RewardResult: Violation score with explanation
        """
        # Convert violation list to a single score (e.g., average or sum)
        score = (
            1 - len(response.violation) / len(self.principles)
            if response.violation
            else 1.0
        )
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithScore(
                    name=self.name, reason=response.reason, score=score
                )
            ],
        )


class BaseListWisePrincipleReward(BasePrincipleReward, BaseListWiseReward):
    """
    List-wise principle evaluation using LLM.

    Compares responses against each other based on ethical principles.
    """

    desc: str = Field(
        default="""Please act as an impartial judge and evaluate the quality of the answers provided by some assistants to the user question displayed below.
You should critically and accurately assess the assistant’s answer with the key principles and choose the assistant that follows the user’s query and answers the user’s question best.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
Do not allow the length of the responses to influence your evaluation.
Be as goal as possible.""",
        description="description",
    )

    template: Type[BasePromptTemplate] = PrincipleListWiseTemplate

    def _before_evaluate(self, sample: DataSample, **kwargs) -> Dict:
        """
        Prepares list-wise evaluation parameters.

        Parameters:
            sample (DataSample): Multi-response sample to evaluate

        Returns:
            Dict: Parameters including all responses for comparison
        """
        params = super()._before_evaluate(sample=sample, **kwargs)
        answers = [output.answer.content for output in sample.output]
        params["answers"] = answers
        return params

    def _after_evaluate(
        self, response: PrincipleListWiseTemplate, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to list-wise ranking metrics.

        Parameters:
            response (PrincipleListWiseTemplate): Parsed LLM comparison

        Returns:
            RewardResult: Relative ranking of responses
        """
        scores = [0 for i in range(len(sample.output))]
        scores[response.best - 1] = 1
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, reason=response.reason, rank=scores
                )
            ],
        )
