from gallery.rm.helpfulness import HelpfulnessReward
from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, MessageRole, Step
from rm_gallery.core.model.base import LLMClient


def test_helpfulness() -> None:
    sample = DataSample(
        unique_id="test_helpfulness",
        input=[
            ChatMessage(
                role=MessageRole.USER, content="今天是几号", additional_kwargs={
                    "context": "当前日期：2025年5月20号，周二"
                }
            )
        ],
        output=[
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT, content="今天是周四"
                )
            ),
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT, content="今天是周二"
                )
            ),
            DataOutput(
                answer=Step(
                    role=MessageRole.ASSISTANT, content="今天是2025年5月20号，周二"
                )
            )
        ]
    )

    llm = LLMClient(model="qwen-max")

    helpfulness = HelpfulnessReward(
        llm=llm,
        name="helpfulness"
    )
    helpfulness.run(sample)