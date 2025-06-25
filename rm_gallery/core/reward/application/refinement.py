import re
from copy import deepcopy

from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import MessageRole, format_messages
from rm_gallery.core.reward.base import BaseReward


def filter_think(text):
    filtered_text = re.sub(r"<think>(.*?)</think>", "", text, flags=re.DOTALL)
    return filtered_text


class LLMRefinement(BaseModule):
    """
    A module implementing iterative response refinement using LLM and reward feedback.

    Attributes:
        reward_module: Reward model for evaluating response quality
        llm: Language model client for generating responses
        max_iterations: Maximum number of refinement iterations
    """

    reward_module: BaseReward = Field(default=..., description="reward module")
    llm: BaseLLM = Field(default=..., description="llm client")
    max_iterations: int = Field(default=3, description="max iterations")

    def _generate_response(
        self,
        sample: DataSample,
        feedback: str | None = None,
        **kwargs,
    ) -> DataSample:
        """
        Generate refined response based on conversation history and feedback.

        Args:
            sample: DataSample object containing input and previous responses
            feedback: Quality assessment feedback for previous responses
            **kwargs: Additional parameters for LLM generation

        Returns:
            Generated response as a DataSample object
        """
        # Construct prompt based on feedback availability
        if feedback is None:
            prompt = """# Task
Please generate a respoonse as the conversation required.

# Conversation history
{history}
""".format(
                history=format_messages(sample.input)
            )
        else:
            prompt = """# Task
Please generate a better response based on the feedback provided on candidate responses.

# Conversation history
{history}

# Responses
{responses}

# Feedback
{feedback}
""".format(
                history=format_messages(sample.input),
                responses="\n".join(
                    [
                        f"<response_{i}>{output.answer.content}</response_{i+1}>"
                        for i, output in enumerate(sample.output)
                    ]
                ),
                feedback=feedback,
            )

        respoonse = self.llm.simple_chat(prompt)
        sample.output.append(
            DataOutput(
                answer=Step(role=MessageRole.ASSISTANT, content=filter_think(respoonse))
            )
        )
        return sample

    def _generate_feedback(self, sample: DataSample, **kwargs) -> str:
        """
        Generate quality feedback for a response sample.

        Args:
            sample: Data sample containing input-response pair for evaluation
            **kwargs: Additional parameters for reward evaluation

        Returns:
            Feedback string describing response quality assessment
        """
        # Evaluate response quality using reward module
        sample = self.reward_module.evaluate(sample)
        feedback = sample.output[0].answer.reward.details[0].reason
        return feedback

    def run(self, sample: DataSample, **kwargs) -> DataSample:
        """
        Execute iterative response refinement process.

        Args:
            sample: Data sample containing input for refinement
            **kwargs: Additional parameters for generation and evaluation

        Returns:
            Final refined response as a DataSample object
        """
        sample = deepcopy(sample)
        if len(sample.output) == 0:
            # Initial response generation
            response = self.llm.chat(sample.input)
            sample.output.append(
                DataOutput(
                    answer=Step(
                        role=MessageRole.ASSISTANT,
                        content=filter_think(response.message.content),
                    )
                )
            )

        # Iterative refinement loop
        for i in range(self.max_iterations):
            # Generate feedback and create refined response
            feedback = self._generate_feedback(sample, **kwargs)
            sample = self._generate_response(sample, feedback, **kwargs)

        return sample
