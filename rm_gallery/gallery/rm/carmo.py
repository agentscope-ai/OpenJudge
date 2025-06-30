# https://arxiv.org/pdf/2410.21545


from typing import List

from pydantic import Field

from rm_gallery.core.data.schema import DataSample
from rm_gallery.core.reward.base import BaseListWiseReward, BaseLLMReward
from rm_gallery.core.reward.schema import RewardDimensionWithRank, RewardResult
from rm_gallery.core.reward.template import (
    BasePromptTemplate,
    PrincipleListWiseTemplate,
)


class CriteriaGenerationPrompt(BasePromptTemplate):
    principles: List[str] = Field(
        default=...,
        description="""```json
[
    "principle 1",
    ...
]
```""",
    )

    @classmethod
    def format(
        cls,
        instruction: str,
        **kwargs,
    ) -> str:
        return f"""# Task Description
- You are an impartial judge tasked with generating principles for evaluating responses provided by AI
assistants to an instruction.
- Your job is to identify important principles, along with detailed descriptions, that a human would use
to objectively evaluate the quality of the response based on the given instruction.
- The principles should ensure that responses accurately fulfill the requirements of the instruction.
- The principles should be designed to ensure that responses are honest, helpful, and harmless (do not
contain offensive content).
- The descriptions of the principles should be framed as chain-of-thought detailed questions that assess
whether the response meets the user’s instruction.
- The length of the response should only be considered a principle if it is specified in the instruction.

# Input
# Instruction
{instruction}

# Output Requirements
{cls.schema(**kwargs)}
"""


class RelativeEvaluationPrompt(PrincipleListWiseTemplate):
    best: int = Field(
        default=...,
        description="which completion is the best? just give the number here!!!",
    )

    @classmethod
    def format(cls, instruction, principles, completions, **kwargs) -> str:
        completion_str = ""
        for i, completion in enumerate(completions):
            completion_str += f"### Completion {i + 1}\n{completion}\n\n"

        principles = "\n".join(
            [f"{i+1}. {principle}" for i, principle in enumerate(principles)]
        )

        return f"""Task Description
- Please act as an impartial judge and evaluate the quality of the responses provided by two AI
assistants to the user instruction shown below. You should choose the assistant that follows the
user’s instructions and answers the user’s instruction better.
- Your evaluation should consider the provided principles.
- Provide detailed reasons assessing the quality of the responses based on each principle individually.
Clearly specify which assistant performed better for each principle.
- After assessing all principles, provide a final verdict based on the overall performance of the
assistants.
- Don’t be influenced by the order in which the responses are presented. Do not favor certain names
of the assistants. Be as objective as possible.

# Input
## Instruction
{instruction}

## Principles
{principles}

## Completions
{completion_str}

# Output Requirements
{cls.schema(**kwargs)}
"""


class CARMO(BaseLLMReward, BaseListWiseReward):
    """Context-Aware Reward Modeling"""

    def _before_evaluate(self, sample: DataSample, **kwargs) -> dict:
        instruction = sample.input[-1].content

        query = CriteriaGenerationPrompt.format(instruction=instruction)
        response = self.llm.simple_chat(query)
        principles = CriteriaGenerationPrompt.parse(response).principles
        completions = [output.answer.content for output in sample.output]

        return dict(
            principles=principles,
            instruction=instruction,
            completions=completions,
        )

    def _after_evaluate(
        self, response: RelativeEvaluationPrompt, sample: DataSample, **kwargs
    ) -> RewardResult:
        """
        Converts LLM response to list-wise ranking metrics.

        Parameters:
            response (RelativeEvaluationPrompt): Parsed LLM comparison

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
