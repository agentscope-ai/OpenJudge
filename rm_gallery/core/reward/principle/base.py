import copy
import json
import random
import re
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, List

from loguru import logger
from pydantic import BaseModel, Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.reward.template import BasePromptTemplate


class BaseGeneratorTemplate(BasePromptTemplate):
    principles: Dict[str, str] = Field(
        default=...,
        description="""```json
{
    "{phrase}": "{description}",
    ...
}
```""",
    )

    @classmethod
    def parse(cls, text: str):
        contents = cls._parse(text)

        json_pattern = r"```json(.*?)```"
        json_dict = re.findall(json_pattern, contents["principles"], re.DOTALL)
        json_dict = json_dict[0] if len(json_dict) > 0 else "{}"

        try:
            parsed_dict = json.loads(json_dict)
        except json.JSONDecodeError:
            pattern = r'"(.*?)"\s*:\s*"(.*?)"'
            matches = re.findall(pattern, json_dict)
            parsed_dict = {key: value for key, value in matches}

        return cls(
            think=contents["think"],
            principles=parsed_dict,
        )


class PrincipleGenerateTempalte(BaseGeneratorTemplate):
    @classmethod
    def format(
        cls,
        scenario: str,
        instruction: str,
        completions: List[str],
        preference: str | int,
        number: int,
        **kwargs,
    ) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += (
                f"<completion_{i + 1}>\n{completion}\n</completion_{i + 1}>\n\n"
            )

        return f"""## Overview
You will be provided with an example of instruction and completions in a task scenario.
Please propose some general principles from the scenario that can help another assistant to determine which one completion is superior to the others in the scenario.

## Requirements for Principles
(1) Principles target some general standards of the "scenario".
(2) Principles are presented from most important to least important.
(3) Principles should be as critical as possible.
(4) Each principle should consist of a brief phrase accompanied by a single sentence description.
(5) The number of principles should be LESS THAN OR EQUAL TO {number}.

## Input
### Scenario
{scenario}

### Instruction
{instruction}

### Completions
{completion_str}

### Preference
Completion {preference} is the best.

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleClusterTemplate(BaseGeneratorTemplate):
    @classmethod
    def format(cls, examples: str, scenario: str, number: int, **kwargs) -> str:
        return f"""## Overview
You will be provided with a set of examples with instruction and pre-generated principles in the scenario.
Please summarize some general principles from the examples that can help another assistant to determine which one completion is superior to the others in the scenario.

## Requirements for Principles
(1) Principles are presented from most important to least important.
(2) Principles should be as critical as possible.
(3) Each principle should consist of a brief phrase accompanied by a single sentence description.
(4) The number of principles should be LESS THAN OR EQUAL TO {number}.
(5) Focus on summarizing recurring candidate principles.

## Input
### Scenario
{scenario}

### Examples
{examples}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleGenerator(BaseModel):
    llm: BaseLLM = Field(default=..., description="llm client")
    scenario: str = Field(default=..., description="assitant scenario")
    generate_number: int = Field(
        default=10, description="number of generated principles"
    )
    cluster_number: int = Field(default=1, description="number of clustered principles")

    def generate(self, sample: DataSample):
        sample = copy.deepcopy(sample)
        instructioin: str = format_messages(sample.input)
        completions = [
            (output.answer.label["preference"], output.answer.content)
            for output in sample.output
        ]
        random.shuffle(completions)
        for i, (label, completion) in enumerate(completions):
            if label == "chosen":
                best = i + 1
        completions = [completion for _, completion in completions]

        prompt = PrincipleGenerateTempalte.format(
            instruction=instructioin,
            completions=completions,
            preference=best,
            enable_thinking=self.llm.enable_thinking,
            scenario=self.scenario,
            number=self.generate_number,
        )

        logger.info(f"prompt: {prompt}")
        response = self.llm.simple_chat(
            prompt,
            sys_prompt="You are a professional assistant skilled in extracting key insights and summarizing information.",
        )
        result = PrincipleGenerateTempalte.parse(response)
        sample.input[-1].additional_kwargs["generate"] = result.model_dump()
        return sample

    def cluster(self, samples: List[DataSample]):
        examples = []
        for i, sample in enumerate(samples):
            sample_principles = []
            for key, value in (
                sample.input[-1].additional_kwargs["generate"]["principles"].items()
            ):
                sample_principles.append(f"{key}: {value}")
            str_principles = "\n".join(sample_principles)
            str_principles = (
                f"<principles_{i+1}>\n{str_principles}\n</principles_{i+1}>"
            )
            str_instruction = f"<instruction_{i+1}>\n{format_messages(sample.input)}\n</instruction_{i+1}>"
            examples.append(
                f"<example_{i+1}>\n{str_instruction}\n{str_principles}\n</example_{i+1}>\n\n"
            )

        str_examples = "\n".join(examples)
        logger.info("===RAW EXAMPLES===\n" + str_examples)

        response = self.llm.simple_chat(
            PrincipleClusterTemplate.format(
                scenario=self.scenario,
                examples=str_examples,
                enable_thinking=self.llm.enable_thinking,
                number=self.cluster_number,
            ),
            sys_prompt="You are a skilled professional assistant focusing on induction and summarization.",
        )
        result = PrincipleClusterTemplate.parse(response)

        logger.info("===CLUSTER RESULT===\n" + result.model_dump_json())
        return result.principles

    def run_batch(
        self, samples: List[DataSample], thread_pool: ThreadPoolExecutor
    ) -> Dict[str, str]:
        futures = [thread_pool.submit(self.generate, sample) for sample in samples]
        wait(futures, return_when=ALL_COMPLETED)
        samples = [future.result() for future in futures]
        return self.cluster(samples)
