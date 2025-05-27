


from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, List, Type, TypeVar

from pydantic import Field
from src.data.base import BaseDataSet
from src.data.schema import DataSample, ChatMessage, Step
from _del.executor import BaseModule
from src.model.base import LLMClient
from src.rm.schema import LLMModuleOutput
from src.rm.template import BaseTemplate


class BaseRewardModule(BaseModule):
    # input: List[InputVar] = Field(default=[], description="input vars mapping")
    # output: Type[LLMModuleOutput] | Type[dict] = Field(default=dict, description="output schema")
    name: str = Field(default=...)

    # @classmethod
    # def parse_var(cls, var: InputVar, data: Any):
    #     """
    #     Prepare parameters for the _run method based on the module's input variable definitions.

    #     This method iterates over each input variable and resolves its value by calling `parse_var`.
    #     The resolved values are collected into a dictionary which can then be used as input to the `_run` method.

    #     Parameters:
    #     - input: InputSample object containing input data.
    #     - output: OutputSample or list of OutputSample objects representing the outputs to process.
    #     - step: Optional Step or list of Step objects if step-level processing is involved.

    #     Returns:
    #     - A dictionary mapping input variable names to their resolved values.
    #     """
    #     if isinstance(_context, list):
    #         return [cls.parse_var(var, d) for d in data]
    #     return get_value_by_path(data.context, var.path, var.default)

    # def prepare(self, input: InputSample, output: OutputSample | List[OutputSample], step: Step | List[Step] | None = None) -> dict:
    #     """
    #     Prepares a parameter dictionary based on the input, output, and step contexts.

    #     This method iterates through each input variable defined in the module and resolves its value
    #     according to its processing level (LISTWISE, POINTWISE, or STEPWISE). The resolved values are
    #     collected into a dictionary that can be used as input for further processing.

    #     Args:
    #         input (InputSample): The input context containing input variable data.
    #         output (OutputSample | List[OutputSample]): The output context(s) containing output variable data.
    #         step (Step | List[Step] | None, optional): The step context(s) containing step-level variable data,
    #             defaults to None.

    #     Returns:
    #         dict: A dictionary mapping variable names to their resolved values.
    #     """
    #     params = {}
    #     for var in self.input:
    #         if var.level == ModuleLevel.LISTWISE:
    #             params[var.name] = self.parse_var(var, input)
    #         elif var.level == ModuleLevel.POINTWISE:
    #             params[var.name] = self.parse_var(var, output)
    #         elif var.level == ModuleLevel.STEPWISE and step is not None:
    #             params[var.name] = self.parse_var(var, step)
    #     return params

    @abstractmethod
    def _run(self, **kwargs) -> LLMModuleOutput | dict:
        ...

    @abstractmethod
    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        ...

    def run_batch(self, samples: BaseDataSet, thread_pool: ThreadPoolExecutor):
        for sample in samples:
            thread_pool.submit(self.run, sample=sample, thread_pool=thread_pool)


class StepRewardModule(BaseRewardModule):
    @abstractmethod
    def _run(self, input: List[ChatMessage], output: Step, step: Step) -> dict:
        ...

    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        for output in sample.output:
            assert isinstance(output.steps, list)
            for step in output.steps:
                thread_pool.submit(self._run, input=sample.input, output=output, step=step)


class PointRewardModule(BaseRewardModule):
    @abstractmethod
    def _run(self, input: List[ChatMessage], output: Step) -> dict:
        ...

    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        for output in sample.output:
            thread_pool.submit(self._run, input=sample.input, output=output)


class ListRewardModule(BaseRewardModule):
    @abstractmethod
    def _run(self, sample: DataSample):
        ...
    
    def run(self, sample: DataSample, thread_pool: ThreadPoolExecutor):
        self._run(sample=sample)


T = TypeVar("T", StepRewardModule, PointRewardModule, ListRewardModule)


class LLMRewardModule(Generic[T]):
    llm: LLMClient = Field(default=..., description="llm client")
    desc: str | None = Field(default=None, description="evaluation task description")
    output: Type[LLMModuleOutput] = Field(default=..., description="llm output schema")
    template: Type[BaseTemplate] | str | dict = Field(default=BaseTemplate, description="prompt template")

    # @model_validator(mode="after")
    # def validate(self) -> Self:
    #     """
    #     parse template
    #     """
    #     if isinstance(self.template, dict):
    #         module = import_module(self.template["path"])
    #         self.template = getattr(module, self.template["class"])
    #     return self

    def format(self, **kwargs) -> str:
        """
        format prompt
        """
        return self.template.format(desc=self.desc, output=self.output, **kwargs)

    def _run(self, **kwargs) -> LLMModuleOutput:
        query = self.format(**kwargs)
        response = self.llm.simple_chat(query=query)
        output = self.output.parse(response)
        return output
