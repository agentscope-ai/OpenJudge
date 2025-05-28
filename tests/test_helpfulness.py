from rm_gallery.core.data.schema import ChatMessage, DataOutput, DataSample, Step
from rm_gallery.core.model.message import MessageRole
from rm_gallery.core.model.openai_llm import OpenaiLLM
from rm_gallery.gallery.rm.helpfulness import HelpfulnessReward


def test_helpfulness() -> None:
    sample = DataSample(
        unique_id="test_helpfulness",
        input=[
            ChatMessage(
                role=MessageRole.USER,
                content="今天是几号",
                additional_kwargs={"context": "当前日期：2025年5月20号，周二"},
            )
        ],
        output=[
            DataOutput(answer=Step(role=MessageRole.ASSISTANT, content="今天是周四")),
            DataOutput(answer=Step(role=MessageRole.ASSISTANT, content="今天是周二")),
            DataOutput(
                answer=Step(role=MessageRole.ASSISTANT, content="今天是2025年5月20号，周二")
            ),
        ],
    )

    llm = OpenaiLLM(model="qwen-max")

    helpfulness = HelpfulnessReward(llm=llm, name="helpfulness")

    helpfulness.run(sample)
    print(sample)


test_helpfulness()
