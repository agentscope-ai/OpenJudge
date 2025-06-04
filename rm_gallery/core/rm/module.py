


from abc import abstractmethod
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import List, Type

from loguru import logger
from pydantic import Field

from rm_gallery.core.base_module import BaseModule
from rm_gallery.core.data.schema import DataOutput, DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.rm.schema import (
    RewardDimensionWithRank,
    RewardDimensionWithScore,
    RewardResult,
)
from rm_gallery.core.rm.template import BasePromptTemplate, PrinciplePairwiseTemplate


class BaseReward(BaseModule):
    """
    This class is used to define a basic reward module, inheriting from BaseModule.
    """

    name: str = Field(default=..., description="The name of the reward module")

    @abstractmethod
    def _evaluate(self, sample: DataSample, **kwargs) -> RewardResult:
        """
        Main processing function, intended to be implemented by subclasses.
        This method accepts arbitrary keyword arguments and its specific behavior is determined by the subclass implementation.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        This abstract method is designed to handle the execution of a single data sample using the provided thread pool.
        Subclasses must implement this method to define the specific behavior of the task execution.

        Parameters:
        - sample: DataSample type, representing the data sample to be processed.
        - thread_pool: ThreadPoolExecutor type, representing the thread pool used for task execution.
        """
        ...

    def evaluate_batch(
        self,
        samples: List[DataSample],
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> List[DataSample]:
        """
        Execute a batch of samples.
        """
        if thread_pool:
            futures = [
                thread_pool.submit(self.evaluate, sample, **kwargs)
                for sample in samples
            ]
            wait(futures, return_when=ALL_COMPLETED)
            samples = [future.result() for future in futures]
        else:
            for i, sample in enumerate(samples):
                samples[i] = self.evaluate(sample=sample, thread_pool=thread_pool)

        return samples


class BaseStepWiseReward(BaseReward):
    """
    The StepWiseReward class, derived from BaseModule, is designed to calculate rewards for each step.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        Abstract method _evaluate, intended to be implemented by subclasses for processing each step's reward calculation.
        """
        ...

    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Method evaluate processes the reward calculation for each step of the sample's output.
        """
        sample = sample.model_copy(deep=True)
        futures = []
        for i, output in enumerate(sample.output):
            assert isinstance(output.steps, list)
            for j, step in enumerate(output.steps):
                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=[DataOutput(answer=output.answer, steps=[step])],
                )

                if thread_pool:
                    futures.append(
                        (
                            i,
                            j,
                            thread_pool.submit(
                                self._evaluate,
                                sample=subsample,
                                thread_pool=thread_pool,
                            ),
                        )
                    )
                else:
                    result = self._evaluate(
                        sample=subsample,
                        thread_pool=thread_pool,
                    )
                    step.reward.details.extend(result.details)
                    step.additional_kwargs[self.name] = result.extra_data
        if thread_pool:
            wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
            for i, j, future in futures:
                result = future.result()
                step = sample.output[i].steps[j]
                step.reward.details.extend(result.details)
                step.additional_kwargs[self.name] = result.extra_data
        return sample


class BasePointWiseReward(BaseReward):
    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithScore]:
        """
        This method is responsible for processing a list of chat messages and updating the step output based on the processing results.
        It is an internal method, indicated by the leading underscore, suggesting it should not be called directly from outside the class.
        """
        ...

    def evaluate(
        self, sample: DataSample, thread_pool: ThreadPoolExecutor | None = None
    ) -> DataSample:
        """
        Starts processing a data sample by submitting tasks to a thread pool.

        For each output in the data sample, a task is submitted to the thread pool for processing.
        This method does not wait for the tasks to complete.

        Parameters:
        - sample: DataSample - The data sample to be processed, containing input and a list of outputs.
        - thread_pool: ThreadPoolExecutor - The thread pool executor used to execute the tasks.
        """
        sample = sample.model_copy(deep=True)
        futures = []
        for i, output in enumerate(sample.output):
            subsample = DataSample(
                unique_id=sample.unique_id, input=sample.input, output=[output]
            )

            if thread_pool:
                futures.append(
                    (
                        i,
                        thread_pool.submit(
                            self._evaluate, sample=subsample, thread_pool=thread_pool
                        ),
                    )
                )
            else:
                result = self._evaluate(
                    sample=subsample,
                    thread_pool=thread_pool,
                )
                output.answer.reward.details += result.details
                output.answer.additional_kwargs[self.name] = result.extra_data

        if thread_pool:
            wait([future[-1] for future in futures], return_when=ALL_COMPLETED)
            for i, future in futures:
                result = future.result()
                output = sample.output[i]
                output.answer.reward.details += result.details
                output.answer.additional_kwargs[self.name] = result.extra_data

        return sample


class BaseListWiseReward(BaseReward):
    """
    This class is a subclass of BaseModule, designed to process data samples and compute rewards.
    It is an abstract class that requires subclasses to implement the _evaluate method to specify the specific reward calculation logic.
    """

    @abstractmethod
    def _evaluate(
        self, sample: DataSample, **kwargs
    ) -> RewardResult[RewardDimensionWithRank]:
        """
        Abstract method _evaluate, intended to be overridden by subclasses.
        This method's purpose is to process a given data sample to compute a reward.

        """
        ...

    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Executes the _evaluate method in a multi-threaded environment.
        This method is intended to be called by external code, which provides a data sample and a thread pool executor.

        """
        sample = sample.model_copy(deep=True)
        result = self._evaluate(sample=sample, thread_pool=thread_pool, **kwargs)
        for reward in result.details:
            for i, output in enumerate(sample.output):
                output.answer.reward.details.append(reward[i])

        sample.input[-1].additional_kwargs[self.name] = result.extra_data
        return sample


class BasePairWiseReward(BaseListWiseReward):
    def evaluate(
        self,
        sample: DataSample,
        thread_pool: ThreadPoolExecutor | None = None,
        **kwargs,
    ) -> DataSample:
        sample = sample.model_copy(deep=True)
        for i, output_i in enumerate(sample.output):
            for j, output_j in enumerate(sample.output, start=i + 1):
                subsample = DataSample(
                    unique_id=sample.unique_id,
                    input=sample.input,
                    output=[output_i, output_j],
                )
                result = self._evaluate(
                    sample=subsample, thread_pool=thread_pool, **kwargs
                )
                for reward in result.details:
                    output_i.answer.reward.details.append(reward[0])
                    output_j.answer.reward.details.append(reward[1])
        return sample


class BaseLLMReward(BaseReward):
    """
    A generic class for LLM-based reward modules, providing a framework for interaction with LLMs and handli
    """

    llm: BaseLLM = Field(default=..., description="llm client")
    template: Type[BasePromptTemplate] = Field(
        default=BasePromptTemplate, description="prompt template"
    )

    def _before_call(self, **kwargs) -> dict:
        """
        Abstract method to be implemented by subclasses for preparing parameters before making a call to the LLM.
        """
        return {}

    def _call(self, **kwargs) -> BasePromptTemplate:
        """
        Abstract method to be implemented by subclasses for making a call to the LLM using the provided parameters.
        """
        query = self.template.format(**kwargs)
        logger.info(f"query: {query}")
        response = self.llm.simple_chat(query=query)
        logger.info(f"response: {response}")
        output = self.template.parse(response)
        logger.info(f"output: {output}")
        return output

    def _after_call(self, response: BasePromptTemplate, **kwargs) -> RewardResult:
        """
        Abstract method to be implemented by subclasses for processing the response from the LLM and setting the reward.
        """
        return RewardResult(
            name=self.name, details=[], extra_data=response.model_dump()
        )

    def _evaluate(self, **kwargs) -> RewardResult:
        """
        Method to execute the full cycle of preparing the call, making the call to the LLM, and processing the response.
        """

        try:
            # Prepare parameters before making the call to the LLM.
            params = self._before_call(**kwargs)

            # Make the call to the LLM using the prepared parameters.
            response = self._call(**params)

            # Process the response from the LLM and set the reward.
            result = self._after_call(response=response, **kwargs)
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            result = RewardResult(
                name=self.name, details=[], extra_data={"error": str(e)}
            )
        return result


class PrinciplePairwiseReward(BaseLLMReward, BaseListWiseReward):
    principles: List[str] = Field(default=[], description="principles")
    examples: List[str] = Field(default=[], description="examples")
    template: Type[PrinciplePairwiseTemplate] = Field(
        default=PrinciplePairwiseTemplate, description="harmfulnessTemplate"
    )
    task_desc: str = Field(default="", description="task desc")

    @abstractmethod
    def _get_principle_and_desc(self):
        """
        generate principles and task desc
        """
        ...

    def _before_call(self, sample: DataSample, **kwargs) -> dict:
        self._get_principle_and_desc()
        return {
            "desc": self.task_desc,
            "principles": "\n".join(self.principles),
            "examples": "\n".join(self.examples),
            "query": sample.input[-1].content,
            "answer_a": sample.output[0].answer.content,
            "answer_b": sample.output[1].answer.content,
        }

    def _after_call(
        self, response: PrinciplePairwiseTemplate, **kwargs
    ) -> RewardResult:
        # calc score for every output,the length of scores must equal to num of output
        if response.better_answer == "A":
            scores = [1, 0]
        elif response.better_answer:
            scores = [0, 1]
        else:
            scores = [0, 0]
        return RewardResult(
            name=self.name,
            details=[
                RewardDimensionWithRank(
                    name=self.name, weight=1, reason=response.reason, rank=scores
                )
            ],
        )
