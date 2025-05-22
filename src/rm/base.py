
from enum import Enum
from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
from src.data.data_schema import ContentDict, EvaluationContext, EvaluationSample
from src.model.base import LLMClient
from src.pipeline.node.base import Node
from src.utils.data import get_value_by_path


class VarType(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class Var(BaseModel):
    name: str = Field(default=...)
    path: str = Field(default=...)
    vtype: VarType = Field(default=VarType.OUTPUT)
    default: Any | None = Field(default=None)


class Rule(BaseModel):
    desc: str = Field(default=..., description="rule description")
    score: float | int | str = Field(default=..., description="scorer")


class BaseTemplate(BaseModel):
    @classmethod
    def format(cls, **kwargs) -> str:
        ...


class BaseParser(Node):
    inputs: List[Var] = Field(default=[])
    output_schema: Type[EvaluationContext] | Type[dict] = Field(default=dict)
    name: str = Field(default=...)

    def prepare_params(self, sample_input: EvaluationSample, sample_output: ContentDict | List[ContentDict]) -> dict:
        """
        prepare params used by _run
        """
        params = {}
        for var in self.inputs:
            if var.vtype == VarType.INPUT:
                params[var.name] = get_value_by_path(sample_input, var.path, var.default)
            elif var.vtype == VarType.OUTPUT:
                if isinstance(sample_output, list):
                    var_list = []
                    for output in sample_output:
                        var_list.append(get_value_by_path(output, var.path, var.default))
                    params[var.name] = var_list
                else:
                    params[var.name] = get_value_by_path(sample_output, var.path, var.default)
        return params

    def _run(self, **kwargs) -> EvaluationContext | dict:
        """
        parser process
        """
        return kwargs

    def run_pointwise(self, sample: EvaluationSample):
        """
        call pointwise parser
        """
        # TODO parallel
        for output in sample.outputs:
            kwargs = self.prepare_params(sample_input=sample, sample_output=output)
            context = self._run(**kwargs)
            output.evaluation_contexts[self.name] = context
            
    def run_listwise(self, sample: EvaluationSample):
        """
        call listwise parser
        """
        kwargs = self.prepare_params(sample_input=sample, sample_output=sample.outputs)
        context = self._run(**kwargs)
        sample.evaluation_contexts[self.name] = context

    def run(self, **kwargs) -> Any:
        """
        node run api TODO
        """
        return None


class LLMParser(BaseParser):
    client: LLMClient = Field(default=..., description="llm client")
    desc: str | None = Field(default=None, description="evaluation task description")
    output_schema: Type[EvaluationContext] = Field(default=..., description="llm output schema")
    template: Type[BaseTemplate] | str | Dict = Field(default=BaseTemplate, description="prompt template")

    def format(self, **kwargs) -> str:
        """
        format prompt
        """
        return self.template.format(desc=self.desc, output_schema=self.output_schema, **kwargs)

    def _run(self, **kwargs) -> EvaluationContext:
        query = self.format(**kwargs)
        response = self.client.simple_chat(query=query)
        output = self.output_schema.parse(response)
        return output


class RuleParser(BaseParser):
    output_schema: Type[dict] = Field(default=dict)

    def _run(self, **kwargs) -> dict:
        ...
