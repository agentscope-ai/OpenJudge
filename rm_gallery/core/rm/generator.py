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
from rm_gallery.core.rm.template import BasePromptTemplate


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
        desc: str,
        instruction: str,
        completions: List[str],
        preference: str | int,
        **kwargs,
    ) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += f"### Completion {i + 1}\n{completion}\n\n\n"

        return f"""## Overview
As a professional assistant specializing in extracting key insights and summarizing information, you will receive a dataset containing the following elements: scenario, instruction, completions, preference. Your objective is to distill the fundamental principles tackle the scenario, by analyzing why one completion is superior to the others.

## Process
1. **Understand the Instruction's Context**: Thoroughly analyze the scenario and requirements outlined in the instruction.
2. **Compare Completions**: Carefully examine completions, noting their differences.
3. **Assess Completions**: Utilize the given preference to reason why one completion is favored over the other, combining this with your previous analysis in steps 1 and 2.
4. **Formulate Principles**: Extract insights into a series of concise principles. Each principle should consist of a brief phrase accompanied by a single sentence description. All principles should generalize across scenarios.
5. **Check for Consistency**: Ensure the preference aligns with the formulated principles. If a discrepancy exists, return to step 1 and attempt another version of the principles.

## Input
### Scenario
{desc}
### Instruction:
{instruction}

{completion_str}
### Preference:
Completion {preference}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleClusterTemplate(BaseGeneratorTemplate):
    @classmethod
    def format(cls, principles: str, desc: str, **kwargs) -> str:
        return f"""## Overview
You are a skilled professional assistant focusing on induction and summarization. You will receive a series of principles in natural language format, each structured as: "phrase": "description, your objective is to output a set of summerized principles that tackle the given scenario.

## Process
1. Begin by focusing on the phrase of each principle, which serves as a subtitle. Use these phrases to perform a coarse-grained classification of the principles.
2. For each cluster formed in step 1, create a summary that encapsulates the essence of the cluster's principles.
3. Within each cluster, compare every individual principle to the cluster-level summary created in step 2. If a principle significantly deviates in meaning from the cluster-level summary, decide whether a new cluster should be formed; if the meanings are aligned, merge the principle with the existing cluster.
4. Repeat step 3 until all principles are consistently grouped within stable clusters, and no new clusters need to be created.
5. Read every summarized principle pair by pair obtained from step 4, and if two principles are highly similar, merge them into a single principle.
6. Generate the final set of summarized principles. Ensure the finalized number of principles is LESS THAN OR EQUAL TO 10, i.e, the number of principles has no lower bound but has upper bound as 10.

Note: Ensure that the finalized summaries do not include specific words that refer to particular instances, as these principles serve as metrics or rubrics to aid in evaluation.

## Input
### Scenario
{desc}

{principles}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleGenerator(BaseModel):
    llm: BaseLLM = Field(default=..., description="llm client")
    desc: str = Field(default=..., description="description")

    def generate(self, sample: DataSample):
        sample = copy.deepcopy(sample)
        instructioin: str = sample.input[-1].content
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
            desc=self.desc,
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
        str_principles = []
        for sample in samples:
            for key, value in (
                sample.input[-1].additional_kwargs["generate"]["principles"].items()
            ):
                str_principles.append(f"{key}: {value}")

        principles = "\n".join(str_principles)
        logger.info("===RAW PRINCIPLES===\n" + principles)

        response = self.llm.simple_chat(
            PrincipleClusterTemplate.format(
                desc=self.desc,
                principles=principles,
                enable_thinking=self.llm.enable_thinking,
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
