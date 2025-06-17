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
    refer: https://selfrefine.info/
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
        return respoonse

    def run(self, input: List[ChatMessage], **kwargs) -> DataSample:
        response = self.llm.chat(input)
        candicates = [response]

        for i in range(self.max_iterations):
            sample = DataSample(
                input=input,
                output=[
                    DataOutput(
                        answer=Step(role=MessageRole.ASSISTANT, content=response)
                    )
                ],
                unique_id=str(uuid.uuid4()),
            )

            sample = self._generate_feedback(sample, **kwargs)
            feedback = sample.output[0].answer.reward.details[0].reason
            sample = self._generate_response(input, candicates, feedback, **kwargs)

        return sample
