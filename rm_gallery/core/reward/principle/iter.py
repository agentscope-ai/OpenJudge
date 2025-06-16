import copy
import random
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Dict, List

import numpy as np
from loguru import logger
from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.model.message import format_messages
from rm_gallery.core.reward.base import BaseListWisePrincipleReward
from rm_gallery.core.reward.principle.base import (
    BaseGeneratorTemplate,
    PrincipleGenerator,
)


class PrincipleGenerateTempalte(BaseGeneratorTemplate):
    @classmethod
    def format(
        cls,
        scenario: str,
        instruction: str,
        completions: List[str],
        prediction: str | int,
        groudtruth: str | int,
        number: int,
        principles: str,
        **kwargs,
    ) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += (
                f"<completion_{i + 1}>\n{completion}\n</completion_{i + 1}>\n\n"
            )

        return f"""## Overview
Please propose additional principles about why a potential completion is qualified for a given instruction in the scenario, by completing the following analysis.
1. Compare and analyze the prediction and the ground truth, and analyze the reasons why the prediction is incorrect.
2. Summarize the points to pay attention to in order to "correctly" determin which one is the best in the same scenario, with following the requirements.

Another assistant will evaluate the completions based on these principles.

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

### Initial Principles
{principles}

### Prediction Preference
Completion {prediction} is better than others.

### Groud Truth Preference
Completion {groudtruth} is better than others

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class PrincipleClusterTemplate(BaseGeneratorTemplate):
    @classmethod
    def format(
        cls, examples: str, scenario: str, number: int, principles, **kwargs
    ) -> str:
        return f"""## Overview
As an principle aggregation and analysis expert, your task is to conduct cluster analysis on a large collection of pre-generated principles based on examples and provide the optimization principles for each category in the scenario.
**Specific Steps:**
1. Organize the initial principles and the provided improvement principles into distinct categories, ensuring that each category is unique and succinct.
2. Summarize the principles within each category into a sample set for that category, while retaining detailed information.

Another assistant will evaluate the completions in the scenario based on these principles.
When consolidating the principles, be sure to maintain the integrity, clarity, and conciseness of each category.


## Requirements for Principles
(1) Principles are presented from most important to least important.
(2) Principles should be as critical as possible.
(3) Each principle should consist of a brief phrase accompanied by a single sentence description.
(4) The number of final principles should be LESS THAN OR EQUAL TO {number}.
(5) Focus on summarizing recurring candidate principles.

## Input
### Scenario
{scenario}

### Initial Principles
{principles}

### Examples
{examples}

## Output Format Requirements
{cls.schema(**kwargs)}
"""


class IterPrincipleGenerator(PrincipleGenerator):
    reward: BaseListWisePrincipleReward = Field(
        default=..., description="reward module"
    )
    max_epochs: int = Field(default=2, description="max epochs")

    def evaluate(
        self,
        samples: List[DataSample],
        principles: Dict[str, str],
        thread_pool: ThreadPoolExecutor,
        **kwargs,
    ):
        self.reward.principles = [
            f"{key}: {value}" for key, value in principles.items()
        ]
        return self.reward.evaluate_batch(
            samples=samples,
            thread_pool=thread_pool,
            **kwargs,
        )

    def generate(self, sample: DataSample, principles: Dict[str, str]):
        sample = copy.deepcopy(sample)
        instructioin: str = format_messages(sample.input)
        completions = [
            (
                output.answer.label["preference"],
                output.answer.content,
                output.answer.reward.score,
            )
            for output in sample.output
        ]
        random.shuffle(completions)
        for i, (label, completion, pred) in enumerate(completions):
            if label == "chosen":
                groud_truth = i + 1

            if pred > 0:
                prediction = i + 1

        completions = [completion for _, completion, _ in completions]

        prompt = PrincipleGenerateTempalte.format(
            instruction=instructioin,
            completions=completions,
            enable_thinking=self.llm.enable_thinking,
            scenario=self.scenario,
            number=self.generate_number,
            groudtruth=groud_truth,
            prediction=prediction,
            principles="\n".join(
                [f"{key}: {value}" for key, value in principles.items()]
            ),
        )

        logger.info(f"prompt: {prompt}")
        response = self.llm.simple_chat(
            prompt,
            sys_prompt="You are a professional assistant skilled in extracting key insights and summarizing information.",
        )
        result = PrincipleGenerateTempalte.parse(response)
        sample.input[-1].additional_kwargs["generate"] = result.model_dump()
        return sample

    def _split_samples(self, samples: List[DataSample]):
        bad_samples = []
        for sample in samples:
            idx = np.argsort(
                np.array(
                    [
                        sum(r.score for r in output.answer.reward.details)
                        for output in sample.output
                    ]
                )
            )[-1]
            sample.output[idx].answer.reward.score = 1
            if sample.output[idx].answer.label["preference"] != "chosen":
                bad_samples.append(sample)
        return bad_samples

    def cluster(self, samples: List[DataSample], principles: Dict[str, str]):
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
                principles="\n".join(
                    [f"{key}: {value}" for key, value in principles.items()]
                ),
            ),
            sys_prompt="You are a skilled professional assistant focusing on induction and summarization.",
        )
        result = PrincipleClusterTemplate.parse(response)

        logger.info("===CLUSTER RESULT===\n" + result.model_dump_json())
        return result.principles

    def run_batch(
        self, samples: List[DataSample], thread_pool: ThreadPoolExecutor
    ) -> Dict[str, str]:
        principles = {
            "Intent Understanding": "Understand user intentions and response.."
        }
        bad_samples = samples

        for i in range(self.max_epochs):
            _samples = self.evaluate(deepcopy(samples), principles, thread_pool)
            bad_samples = self._split_samples(_samples)
            futures = [
                thread_pool.submit(self.generate, sample, principles)
                for sample in bad_samples
            ]
            wait(futures, return_when=ALL_COMPLETED)
            bad_samples = [future.result() for future in futures]
            principles = self.cluster(bad_samples, principles)

        return principles
