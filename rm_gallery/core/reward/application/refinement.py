import uuid
from typing import List

from pydantic import Field

from rm_gallery.core.base import BaseModule
from rm_gallery.core.data.schema import DataOutput, DataSample, Step
from rm_gallery.core.model.base import BaseLLM
from rm_gallery.core.model.message import ChatMessage, MessageRole, format_messages
from rm_gallery.core.reward.base import BaseReward


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
        input: List[ChatMessage],
        candicates: List[ChatMessage] | None = None,
        feedback: str | None = None,
        **kwargs,
    ):
        """
        Generate refined response based on conversation history and feedback.

        Args:
            input: List of chat messages forming the conversation history
            candicates: Previous response attempts to be improved upon
            feedback: Quality assessment feedback for previous responses
            **kwargs: Additional parameters for LLM generation

        Returns:
            Generated response as a ChatMessage object
        """
        # Construct prompt based on feedback availability
        if feedback is None:
            prompt = """# Task
Please generate a respoonse as the conversation required.

# Conversation history
{history}
""".format(
                history=format_messages(input)
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
                history=format_messages(input),
                responses="\n".join(
                    [
                        f"<response_{i}>{msg.content}</response_{i+1}>"
                        for i, msg in enumerate(candicates)
                    ]
                ),
                feedback=feedback,
            )

        respoonse = self.llm.simple_chat(prompt)
        return ChatMessage(role=MessageRole.ASSISTANT, content=respoonse)

    def _generate_feedback(self, sample: DataSample, **kwargs):
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

    def run(self, input: List[ChatMessage], **kwargs) -> ChatMessage:
        """
        Execute iterative response refinement process.

        Args:
            input: List of chat messages forming the conversation history
            **kwargs: Additional parameters for generation and evaluation

        Returns:
            Final refined response as a ChatMessage object
        """
        # Initial response generation
        response = self.llm.chat(input)
        candicates = [response]

        # Iterative refinement loop
        for i in range(self.max_iterations):
            # Create evaluation sample with current response
            sample = DataSample(
                input=input,
                output=[
                    DataOutput(
                        answer=Step(
                            role=MessageRole.ASSISTANT, content=response.content
                        )
                    )
                ],
                unique_id=str(uuid.uuid4()),
            )

            # Generate feedback and create refined response
            feedback = self._generate_feedback(sample, **kwargs)
            response = self._generate_response(input, candicates, feedback, **kwargs)
            candicates.append(response)
        return response
